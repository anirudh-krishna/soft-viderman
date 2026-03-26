import logging
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class Decoder(ABC):
    """
    Abstract base class for LDPC decoders.

    Subclasses must define:
        - label (class attribute): string identifier for the algorithm
        - decode(syndrome, **kwargs) -> bool
    """

    label = None  # Override in subclasses

    def __init__(self, code):
        self.code = code
        self.n = code.n
        self.m = code.m
        self.dv = code.dv
        self.dc = code.dc
        self.bit_nbhd = code.bit_nbhd
        self.check_nbhd = code.check_nbhd
        self.id = code.id

    @abstractmethod
    def decode(self, syndrome, **kwargs):
        """
        Decode the given syndrome. Returns True if decoding succeeded.
        """
        pass

    def verify_syndrome(self, recovered, syndrome):
        """
        Verify that the recovered error vector produces the given syndrome.

        Args:
            recovered: np.array of length n, binary (0/1)
            syndrome: np.array of length m, binary (0/1)

        Returns:
            True if H @ recovered mod 2 == syndrome.
        """
        recovered_syndrome = np.zeros(self.m, dtype=int)
        for c in range(self.m):
            recovered_syndrome[c] = np.sum(recovered[self.check_nbhd[c]]) % 2
        return np.array_equal(recovered_syndrome, syndrome)

    def erasure_recovery(self, L, syndrome):
        """
        Peeling decoder for erasure recovery.

        Treats all bits outside L as 0. Iteratively resolves erased bits
        by finding check nodes connected to exactly one erased variable.
        The resolved value is determined by the parity constraint and syndrome.

        Args:
            L: set of variable indices to treat as erased
            syndrome: np.array of length m, binary (0/1)

        Returns:
            (success, recovered) where success is True if all erased bits were
            resolved, and recovered is an np.array of length n with the recovered
            error vector (0 for bits outside L).
        """
        L_remaining = set(L)
        recovered = np.zeros(self.n, dtype=int)

        if len(L_remaining) == 0:
            return True, recovered

        # Effective syndrome: tracks remaining parity after accounting for resolved bits
        eff_syndrome = syndrome.copy()

        # For each check, track how many of its neighbors are still erased
        check_erased_count = np.zeros(self.m, dtype=int)
        for v in L_remaining:
            for c in self.bit_nbhd[v]:
                check_erased_count[c] += 1

        # Initialize queue of checks with exactly 1 erased neighbor
        queue = []
        for c in range(self.m):
            if check_erased_count[c] == 1:
                queue.append(c)

        while queue:
            c = queue.pop()
            if check_erased_count[c] != 1:
                continue

            # Find the single erased neighbor
            resolved_v = None
            for v in self.check_nbhd[c]:
                if v in L_remaining:
                    resolved_v = v
                    break

            if resolved_v is None:
                continue

            # The resolved bit value is the effective syndrome of this check
            recovered[resolved_v] = eff_syndrome[c]
            L_remaining.discard(resolved_v)

            # Update effective syndrome for all checks neighboring the resolved variable
            if recovered[resolved_v] == 1:
                for c2 in self.bit_nbhd[resolved_v]:
                    eff_syndrome[c2] ^= 1

            # Update erased counts
            for c2 in self.bit_nbhd[resolved_v]:
                check_erased_count[c2] -= 1
                if check_erased_count[c2] == 1:
                    queue.append(c2)

        success = len(L_remaining) == 0
        logger.debug("erasure_recovery: success=%s, stuck_vars=%d", success, len(L_remaining))
        return success, recovered


class VidermanDecoder(Decoder):
    """
    Decoder using Viderman's algorithm: greedy find + erasure recovery.
    """

    label = "viderman"

    def find(self, syndrome, h):
        """
        Identify suspect variable nodes via Viderman's greedy expansion.

        Args:
            syndrome: np.array of length m, binary (0/1)
            h: integer threshold; add v to L if |N(v) ∩ R| >= h

        Returns:
            (L, R) where L is a set of variable indices, R is a set of check indices
        """
        R = set(np.where(syndrome == 1)[0])
        L = set()

        changed = True
        while changed:
            changed = False
            for v in range(self.n):
                if v in L:
                    continue
                nbhd = self.bit_nbhd[v]
                count = sum(1 for c in nbhd if c in R)
                if count >= h:
                    L.add(v)
                    R.update(nbhd)
                    changed = True

        logger.debug("find: |L|=%d, |R|=%d, h=%d", len(L), len(R), h)
        return L, R

    def decode(self, syndrome, h):
        """
        Full Viderman decoding: find suspect set, early-fail check, then erasure recovery.

        Args:
            syndrome: np.array of length m, binary (0/1)
            h: integer threshold for find

        Returns:
            True if decoding succeeded, False otherwise.
        """
        L, R = self.find(syndrome, h)

        # Early fail: any unsatisfied check outside R means the error is not contained
        unsatisfied = set(np.where(syndrome == 1)[0])
        if not unsatisfied.issubset(R):
            logger.debug("decode: early fail, %d unsatisfied checks outside R",
                         len(unsatisfied - R))
            return False

        success, recovered = self.erasure_recovery(L, syndrome)
        if not success:
            return False

        if not self.verify_syndrome(recovered, syndrome):
            logger.debug("decode: syndrome verification failed")
            return False

        return True


class BitFlipDecoder(Decoder):
    """
    Bit-flipping decoder: iteratively flip the variable node with the most
    unsatisfied check neighbors (must exceed dv/2) until no flippable node
    remains.
    """

    label = "bitflip"

    def decode(self, syndrome, max_iters=None):
        """
        Decode by iterative bit-flipping.

        Args:
            syndrome: np.array of length m, binary (0/1)
            max_iters: maximum number of flip iterations (default: n)

        Returns:
            (e_deduced, converged) where e_deduced is np.array of length n
            and converged is True if the syndrome was fully cleared.
        """
        if max_iters is None:
            max_iters = self.n

        synd = syndrome.copy()
        e_deduced = np.zeros(self.n, dtype=int)

        # Compute initial unsat_count[v] = number of neighbors c with synd[c] == 1
        unsat_count = np.zeros(self.n, dtype=int)
        for v in range(self.n):
            for c in self.bit_nbhd[v]:
                if synd[c] == 1:
                    unsat_count[v] += 1

        threshold = self.dv // 2 + 1  # strictly more than half

        for _ in range(max_iters):
            v = np.argmax(unsat_count)
            if unsat_count[v] < threshold:
                break

            # Flip variable v
            e_deduced[v] ^= 1

            # Update syndrome and unsat_count for affected neighbors
            for c in self.bit_nbhd[v]:
                if synd[c] == 1:
                    # This check becomes satisfied: decrement unsat_count for all its variables
                    synd[c] = 0
                    for u in self.check_nbhd[c]:
                        unsat_count[u] -= 1
                else:
                    # This check becomes unsatisfied: increment unsat_count for all its variables
                    synd[c] = 1
                    for u in self.check_nbhd[c]:
                        unsat_count[u] += 1

        converged = not np.any(synd)
        logger.debug("bitflip decode: converged=%s, flips=%d", converged, np.sum(e_deduced))
        return e_deduced, converged
