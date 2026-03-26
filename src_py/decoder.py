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

    def decode(self, syndrome, p=None, max_iters=None):
        """
        Decode by iterative bit-flipping.

        Args:
            syndrome: np.array of length m, binary (0/1)
            p: unused (accepted for interface compatibility)
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


class SoftBitFlipDecoder(Decoder):
    """
    Soft bit-flipping decoder using channel log-likelihood ratio.

    Flip metric for variable v:
        E_v = -b_v * (1 - 2*x_v) + sum_{c in N(v)} (2*s_c - 1)
    where b_v = log((1-p)/p) and x_v in {0,1} is the current error estimate.

    Flips the variable with the largest E_v, provided E_v > 0.
    """

    label = "soft_bitflip"

    def decode(self, syndrome, p=None, alpha=1.0, max_iters=None):
        """
        Decode by soft iterative bit-flipping.

        E_v = -b * (1 - 2*x_v) + alpha * sum_{c in N(v)} (2*s_c - 1)

        Args:
            syndrome: np.array of length m, binary (0/1)
            p: channel error probability (used to compute LLR prior)
            alpha: weight for syndrome evidence relative to prior
            max_iters: maximum number of flip iterations (default: n)

        Returns:
            (e_deduced, converged) where e_deduced is np.array of length n
            and converged is True if the syndrome was fully cleared.
        """
        if max_iters is None:
            max_iters = self.n

        b = np.log((1 - p) / p)  # channel LLR

        synd = syndrome.copy()
        e_deduced = np.zeros(self.n, dtype=int)

        # Compute initial syndrome contribution for each variable:
        # synd_score[v] = sum_{c in N(v)} (2*s_c - 1)
        synd_score = np.zeros(self.n, dtype=float)
        for v in range(self.n):
            for c in self.bit_nbhd[v]:
                synd_score[v] += 2 * synd[c] - 1

        # E_v = -b * (1 - 2*x_v) + alpha * synd_score[v]
        # Initially all x_v = 0, so (1 - 2*x_v) = 1, prior term = -b
        E = -b + alpha * synd_score

        for _ in range(max_iters):
            v = np.argmax(E)
            if E[v] <= 0:
                break

            # Flip variable v
            e_deduced[v] ^= 1
            # Prior term for v flips sign: toggles by 2*b
            if e_deduced[v] == 1:
                E[v] += 2 * b
            else:
                E[v] -= 2 * b

            # Update syndrome and scores for affected neighbors
            for c in self.bit_nbhd[v]:
                if synd[c] == 1:
                    # Check becomes satisfied: (2*s_c - 1) changes from +1 to -1, delta = -2
                    synd[c] = 0
                    for u in self.check_nbhd[c]:
                        synd_score[u] -= 2
                        E[u] -= 2 * alpha
                else:
                    # Check becomes unsatisfied: (2*s_c - 1) changes from -1 to +1, delta = +2
                    synd[c] = 1
                    for u in self.check_nbhd[c]:
                        synd_score[u] += 2
                        E[u] += 2 * alpha

        converged = not np.any(synd)
        logger.debug("soft_bitflip decode: converged=%s, flips=%d, alpha=%.3f",
                     converged, np.sum(e_deduced), alpha)
        return e_deduced, converged


class MinSumBPDecoder(Decoder):
    """
    Min-sum belief propagation decoder for LDPC codes (syndrome-based).

    Messages are LLRs. Positive = likely no error, negative = likely error.
    Check-to-variable uses the min-sum approximation.

    All messages stored in flat arrays indexed by edge number, with
    precomputed index arrays for vectorized gather/scatter.
    """

    def __init__(self, code, max_iters=50):
        super().__init__(code)
        self.max_iters = max_iters
        self.label = f"minsum_bp_i{max_iters}"

        # Total number of edges = n * dv = m * dc
        self.n_edges = self.n * self.dv

        # Edge layout: edges 0..dv-1 belong to variable 0, dv..2*dv-1 to variable 1, etc.
        # edge_var[e] = variable of edge e
        # edge_check[e] = check of edge e
        edge_var = np.empty(self.n_edges, dtype=int)
        edge_check = np.empty(self.n_edges, dtype=int)
        for v in range(self.n):
            for i, c in enumerate(self.bit_nbhd[v]):
                e = v * self.dv + i
                edge_var[e] = v
                edge_check[e] = c
        self.edge_var = edge_var
        self.edge_check = edge_check

        # For each edge e (indexed by variable), find the corresponding edge index
        # when indexed by check. We need this to map between the two views.
        # check_edges[c] = list of edge indices (in the variable-indexed flat array)
        #                   for check c, in the order of check_nbhd[c]
        check_edges = [[] for _ in range(self.m)]
        for v in range(self.n):
            for i, c in enumerate(self.bit_nbhd[v]):
                check_edges[c].append(v * self.dv + i)

        # Reorder so check_edges[c][j] corresponds to check_nbhd[c][j]
        check_edges_ordered = []
        for c in range(self.m):
            edge_map = {}
            for e in check_edges[c]:
                edge_map[edge_var[e]] = e
            check_edges_ordered.append([edge_map[v] for v in self.check_nbhd[c]])

        # c_edges: shape (m, dc) — edge indices grouped by check
        self.c_edges = np.array(check_edges_ordered, dtype=int)

        # v_edges: shape (n, dv) — edge indices grouped by variable
        self.v_edges = np.arange(self.n_edges, dtype=int).reshape(self.n, self.dv)

    def decode(self, syndrome, p=None, max_iters=None):
        """
        Decode by min-sum belief propagation (vectorized).

        Args:
            syndrome: np.array of length m, binary (0/1)
            p: channel error probability
            max_iters: override for self.max_iters if provided

        Returns:
            (e_deduced, converged) where e_deduced is np.array of length n
            and converged is True if the syndrome was fully cleared.
        """
        if max_iters is None:
            max_iters = self.max_iters

        L = np.log((1 - p) / p)

        # Messages stored in flat arrays of length n_edges
        msg_vc = np.full(self.n_edges, L)  # variable-to-check
        msg_cv = np.zeros(self.n_edges)    # check-to-variable

        # Syndrome sign: (-1)^{s_c} for each check
        synd_sign = 1 - 2 * syndrome  # shape (m,)

        prev_decision = np.zeros(self.n, dtype=int)

        for iteration in range(max_iters):
            # --- Check-to-variable update ---
            # Gather incoming v->c messages for each check: shape (m, dc)
            incoming = msg_vc[self.c_edges]

            signs = np.sign(incoming)
            signs[signs == 0] = 1  # treat zero as positive
            mags = np.abs(incoming)

            # Product of all signs per check, adjusted by syndrome
            sign_prod = synd_sign * np.prod(signs, axis=1)  # shape (m,)

            # Min and second-min per check row
            sorted_mags = np.sort(mags, axis=1)
            min1 = sorted_mags[:, 0]  # shape (m,)
            min2 = sorted_mags[:, 1]  # shape (m,)

            # For each edge in check c: excl_sign = sign_prod[c] * sign[j] (undoes j's contribution)
            # excl_min = min1 if this edge is not the argmin, else min2
            argmin_mask = (mags == min1[:, None])
            # If multiple edges tie for min, only exclude the first one
            first_argmin = np.zeros_like(argmin_mask)
            first_argmin[np.arange(self.m), np.argmax(argmin_mask, axis=1)] = True

            excl_min = np.where(first_argmin, min2[:, None], min1[:, None])
            excl_sign = sign_prod[:, None] * signs

            outgoing_cv = excl_sign * excl_min  # shape (m, dc)

            # Scatter back to flat msg_cv array
            np.put(msg_cv, self.c_edges.ravel(), outgoing_cv.ravel())

            # --- Variable-to-check update ---
            # Gather incoming c->v messages for each variable: shape (n, dv)
            incoming_cv = msg_cv[self.v_edges]
            incoming_sum = np.sum(incoming_cv, axis=1)  # shape (n,)

            # Outgoing: L + sum of all incoming except the target
            msg_vc[self.v_edges] = L + incoming_sum[:, None] - incoming_cv

            # --- Hard decision ---
            belief = L + incoming_sum  # shape (n,)
            e_deduced = (belief < 0).astype(int)

            # Early stop if decisions stabilized
            if np.array_equal(e_deduced, prev_decision) and iteration > 0:
                break
            prev_decision = e_deduced.copy()

        converged = self.verify_syndrome(e_deduced, syndrome)
        logger.debug("minsum_bp decode: converged=%s, iters=%d, weight=%d",
                     converged, iteration + 1, np.sum(e_deduced))
        return e_deduced, converged


class SumProductBPDecoder(MinSumBPDecoder):
    """
    Sum-product belief propagation decoder for LDPC codes (syndrome-based).

    Uses the exact check-to-variable update:
        m_{c->v} = (-1)^{s_c} * 2 * atanh(prod_{u in N(c)\\v} tanh(m_{u->c} / 2))
    """

    def __init__(self, code, max_iters=50):
        super().__init__(code, max_iters=max_iters)
        self.label = f"sumproduct_bp_i{max_iters}"

    def decode(self, syndrome, p=None, max_iters=None):
        """
        Decode by sum-product belief propagation (vectorized).

        Args:
            syndrome: np.array of length m, binary (0/1)
            p: channel error probability
            max_iters: override for self.max_iters if provided

        Returns:
            (e_deduced, converged) where e_deduced is np.array of length n
            and converged is True if the syndrome was fully cleared.
        """
        if max_iters is None:
            max_iters = self.max_iters

        L = np.log((1 - p) / p)

        msg_vc = np.full(self.n_edges, L)
        msg_cv = np.zeros(self.n_edges)

        synd_sign = 1 - 2 * syndrome  # shape (m,)

        prev_decision = np.zeros(self.n, dtype=int)

        for iteration in range(max_iters):
            # --- Check-to-variable update (sum-product) ---
            # Gather incoming v->c messages for each check: shape (m, dc)
            incoming = msg_vc[self.c_edges]

            # tanh(m/2), clamped to avoid atanh(±1) = ±inf
            tanh_half = np.tanh(incoming / 2)
            tanh_half = np.clip(tanh_half, -1 + 1e-15, 1 - 1e-15)

            # Product of all tanh values per check
            tanh_prod = np.prod(tanh_half, axis=1)  # shape (m,)

            # For each edge j: exclude j by dividing out tanh_half[j]
            # tanh_half can be zero, so guard against division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                excl_prod = tanh_prod[:, None] / tanh_half  # shape (m, dc)
            excl_prod = np.clip(excl_prod, -1 + 1e-15, 1 - 1e-15)

            # m_{c->v} = (-1)^{s_c} * 2 * atanh(excl_prod)
            outgoing_cv = synd_sign[:, None] * 2 * np.arctanh(excl_prod)

            # Scatter back to flat msg_cv array
            np.put(msg_cv, self.c_edges.ravel(), outgoing_cv.ravel())

            # --- Variable-to-check update (same as min-sum) ---
            incoming_cv = msg_cv[self.v_edges]
            incoming_sum = np.sum(incoming_cv, axis=1)

            msg_vc[self.v_edges] = L + incoming_sum[:, None] - incoming_cv

            # --- Hard decision ---
            belief = L + incoming_sum
            e_deduced = (belief < 0).astype(int)

            if np.array_equal(e_deduced, prev_decision) and iteration > 0:
                break
            prev_decision = e_deduced.copy()

        converged = self.verify_syndrome(e_deduced, syndrome)
        logger.debug("sumproduct_bp decode: converged=%s, iters=%d, weight=%d",
                     converged, iteration + 1, np.sum(e_deduced))
        return e_deduced, converged


class BPFlipDecoder(Decoder):
    """
    Hybrid decoder: run min-sum BP, then clean up residual syndrome with hard bit-flip.
    """

    def __init__(self, code, bp_iters=50, flip_max_iters=None, bp_class=MinSumBPDecoder):
        super().__init__(code)
        self.bp_iters = bp_iters
        self.flip_max_iters = flip_max_iters if flip_max_iters is not None else code.n
        self.bp_decoder = bp_class(code, max_iters=bp_iters)
        bp_name = "bp" if bp_class is MinSumBPDecoder else "sp"
        self.label = f"{bp_name}{bp_iters}_flip"

    def decode(self, syndrome, p=None, max_iters=None):
        """
        Decode by running BP then hard bit-flip on the residual syndrome.

        Args:
            syndrome: np.array of length m, binary (0/1)
            p: channel error probability
            max_iters: unused (accepted for interface compatibility)

        Returns:
            (e_deduced, converged) where e_deduced is np.array of length n
            and converged is True if the syndrome was fully cleared.
        """
        # Phase 1: BP
        e_deduced, converged = self.bp_decoder.decode(syndrome, p=p)
        if converged:
            return e_deduced, True

        # Phase 2: compute residual syndrome and run hard bit-flip
        residual_synd = np.zeros(self.m, dtype=int)
        for c in range(self.m):
            residual_synd[c] = (syndrome[c] + np.sum(e_deduced[self.check_nbhd[c]])) % 2

        # Initialize unsat_count from residual syndrome
        unsat_count = np.zeros(self.n, dtype=int)
        for v in range(self.n):
            for c in self.bit_nbhd[v]:
                if residual_synd[c] == 1:
                    unsat_count[v] += 1

        threshold = self.dv // 2 + 1

        for _ in range(self.flip_max_iters):
            v = np.argmax(unsat_count)
            if unsat_count[v] < threshold:
                break

            # Flip variable v
            e_deduced[v] ^= 1

            # Update residual syndrome and unsat_count
            for c in self.bit_nbhd[v]:
                if residual_synd[c] == 1:
                    residual_synd[c] = 0
                    for u in self.check_nbhd[c]:
                        unsat_count[u] -= 1
                else:
                    residual_synd[c] = 1
                    for u in self.check_nbhd[c]:
                        unsat_count[u] += 1

        converged = not np.any(residual_synd)
        logger.debug("bp_flip decode: converged=%s, weight=%d", converged, np.sum(e_deduced))
        return e_deduced, converged
