import numpy as np
from read_ccodes import read_ccode
from decoder import VidermanDecoder, BitFlipDecoder, SoftBitFlipDecoder
from utils import compute_synd
from main import load_codes


def walk_through_decode(code, p, h, seed=42):
    """
    Generate a single error sample and walk through every step of
    Viderman's decoding algorithm, printing intermediate state.
    """
    np.random.seed(seed)
    decoder = VidermanDecoder(code)

    print("=" * 70)
    print(f"Code: {code.id}  n={code.n}  m={code.m}  dv={code.dv}  dc={code.dc}")
    print(f"h={h}")
    print("=" * 70)

    # --- Generate error ---
    err = (np.random.random(code.n) < p).astype(int)
    error_support = set(np.where(err == 1)[0])
    print(f"\nError: weight={len(error_support)}, support={sorted(error_support)}")

    # --- Compute syndrome ---
    syndrome = compute_synd(code, err)
    unsatisfied = set(np.where(syndrome == 1)[0])
    print(f"Syndrome: weight={len(unsatisfied)}, unsatisfied checks={sorted(unsatisfied)}")

    # --- find step ---
    print(f"\n--- find(h={h}) ---")
    R = set(unsatisfied)
    L = set()
    iteration = 0

    changed = True
    while changed:
        changed = False
        for v in range(code.n):
            if v in L:
                continue
            nbhd = code.bit_nbhd[v]
            count = sum(1 for c in nbhd if c in R)
            if count >= h:
                L.add(v)
                new_checks = set(nbhd) - R
                R.update(nbhd)
                iteration += 1
                in_error = "ERROR" if v in error_support else "clean"
                print(f"  iter {iteration}: add v={v} ({in_error}), "
                      f"|N(v) ∩ R|={count}/{code.dv}, "
                      f"new checks added to R: {sorted(new_checks)}, "
                      f"|L|={len(L)}, |R|={len(R)}")
                changed = True

    print(f"\nfind result: |L|={len(L)}, |R|={len(R)}")
    print(f"  L = {sorted(L)}")
    print(f"  error support ⊆ L? {error_support.issubset(L)}")
    missed = error_support - L
    if missed:
        print(f"  MISSED error bits: {sorted(missed)}")
    extra = L - error_support
    if extra:
        print(f"  Extra (clean) bits in L: {sorted(extra)}")

    # --- Early fail check ---
    print(f"\n--- Early fail check ---")
    outside_R = unsatisfied - R
    if outside_R:
        print(f"  FAIL: {len(outside_R)} unsatisfied checks outside R: {sorted(outside_R)}")
        print(f"\nDecode result: FAILURE (early fail)")
        return False
    else:
        print(f"  PASS: all unsatisfied checks are in R")

    # --- Erasure recovery ---
    print(f"\n--- erasure_recovery(L, syndrome), |L|={len(L)} ---")
    L_remaining = set(L)
    recovered = np.zeros(code.n, dtype=int)
    eff_syndrome = syndrome.copy()

    check_erased_count = np.zeros(code.m, dtype=int)
    for v in L_remaining:
        for c in code.bit_nbhd[v]:
            check_erased_count[c] += 1

    queue = []
    for c in range(code.m):
        if check_erased_count[c] == 1:
            queue.append(c)

    print(f"  Initial queue (degree-1 checks): {len(queue)} checks")

    peel_step = 0
    while queue:
        c = queue.pop()
        if check_erased_count[c] != 1:
            continue

        resolved_v = None
        for v in code.check_nbhd[c]:
            if v in L_remaining:
                resolved_v = v
                break

        if resolved_v is None:
            continue

        recovered[resolved_v] = eff_syndrome[c]
        L_remaining.discard(resolved_v)
        peel_step += 1

        if peel_step <= 20 or len(L_remaining) == 0:
            print(f"  peel {peel_step}: check c={c} resolves v={resolved_v}={recovered[resolved_v]}, "
                  f"|L_remaining|={len(L_remaining)}")
        elif peel_step == 21:
            print(f"  ... (suppressing further peel steps)")

        if recovered[resolved_v] == 1:
            for c2 in code.bit_nbhd[resolved_v]:
                eff_syndrome[c2] ^= 1

        for c2 in code.bit_nbhd[resolved_v]:
            check_erased_count[c2] -= 1
            if check_erased_count[c2] == 1:
                queue.append(c2)

    peeling_success = len(L_remaining) == 0
    print(f"\nPeeling: {peel_step} steps total")
    if not peeling_success:
        print(f"Decode result: FAILURE (peeling stuck, {len(L_remaining)} bits unresolved)")
        print(f"  Stuck bits: {sorted(L_remaining)}")
        return False

    print(f"Peeling completed: all {len(L)} erased bits resolved")
    recovered_support = set(np.where(recovered == 1)[0])
    print(f"  Recovered error: weight={len(recovered_support)}, support={sorted(recovered_support)}")
    print(f"  Actual error:    weight={len(error_support)}, support={sorted(error_support)}")
    print(f"  Match? {recovered_support == error_support}")

    # --- Syndrome verification ---
    print(f"\n--- Syndrome verification ---")
    recovered_syndrome = np.zeros(code.m, dtype=int)
    for c in range(code.m):
        recovered_syndrome[c] = np.sum(recovered[code.check_nbhd[c]]) % 2
    syndromes_match = np.array_equal(recovered_syndrome, syndrome)
    print(f"  H @ recovered mod 2 == syndrome? {syndromes_match}")

    if syndromes_match:
        print(f"\nDecode result: SUCCESS")
    else:
        mismatched = np.where(recovered_syndrome != syndrome)[0]
        print(f"  Mismatched checks: {sorted(mismatched)}")
        print(f"\nDecode result: FAILURE (syndrome mismatch)")

    return syndromes_match


def walk_through_bitflip(code, p, seed=42):
    """
    Generate a single error sample and walk through every step of
    the bit-flip decoding algorithm, printing intermediate state.
    """
    np.random.seed(seed)
    decoder = BitFlipDecoder(code)
    threshold = code.dv // 2 + 1

    print("=" * 70)
    print(f"Code: {code.id}  n={code.n}  m={code.m}  dv={code.dv}  dc={code.dc}")
    print(f"Bit-flip threshold: > dv/2, i.e. unsat_count >= {threshold}")
    print("=" * 70)

    # --- Generate error ---
    err = (np.random.random(code.n) < p).astype(int)
    error_support = set(np.where(err == 1)[0])
    print(f"\nError: weight={len(error_support)}, support={sorted(error_support)}")

    # --- Compute syndrome ---
    syndrome = compute_synd(code, err)
    unsatisfied = set(np.where(syndrome == 1)[0])
    print(f"Syndrome: weight={len(unsatisfied)}, unsatisfied checks={sorted(unsatisfied)}")

    # --- Initialize unsat_count ---
    synd = syndrome.copy()
    e_deduced = np.zeros(code.n, dtype=int)
    unsat_count = np.zeros(code.n, dtype=int)
    for v in range(code.n):
        for c in code.bit_nbhd[v]:
            if synd[c] == 1:
                unsat_count[v] += 1

    print(f"\nInitial unsat_count: max={np.max(unsat_count)}, "
          f"num flippable (>= {threshold}): {np.sum(unsat_count >= threshold)}")

    # --- Iterative flipping ---
    print(f"\n--- Bit-flip iterations ---")
    iteration = 0
    max_iters = code.n

    for iteration in range(1, max_iters + 1):
        v = np.argmax(unsat_count)
        if unsat_count[v] < threshold:
            print(f"\n  No flippable variable found (max unsat_count={unsat_count[v]}). Stopping.")
            break

        in_error = "in error" if v in error_support else "clean"
        currently_flipped = "unflip" if e_deduced[v] == 1 else "flip"

        if iteration <= 30:
            print(f"  iter {iteration}: {currently_flipped} v={v} ({in_error}), "
                  f"unsat_count={unsat_count[v]}/{code.dv}")
        elif iteration == 31:
            print(f"  ... (suppressing further iterations)")

        # Flip
        e_deduced[v] ^= 1
        for c in code.bit_nbhd[v]:
            if synd[c] == 1:
                synd[c] = 0
                for u in code.check_nbhd[c]:
                    unsat_count[u] -= 1
            else:
                synd[c] = 1
                for u in code.check_nbhd[c]:
                    unsat_count[u] += 1

        remaining_syndrome = int(np.sum(synd))
        if iteration <= 30:
            print(f"         remaining syndrome weight={remaining_syndrome}, "
                  f"max unsat_count={np.max(unsat_count)}, "
                  f"flippable={np.sum(unsat_count >= threshold)}")

    # --- Result ---
    converged = not np.any(synd)
    flipped_support = set(np.where(e_deduced == 1)[0])

    print(f"\n--- Result after {iteration} iterations ---")
    print(f"  Converged (syndrome cleared): {converged}")
    print(f"  Flipped bits: weight={len(flipped_support)}, support={sorted(flipped_support)}")
    print(f"  Actual error: weight={len(error_support)}, support={sorted(error_support)}")

    if not converged:
        remaining = set(np.where(synd == 1)[0])
        print(f"  Remaining syndrome weight: {len(remaining)}")
        print(f"\nDecode result: FAIL (syndrome not cleared)")
        return

    residual = (err + e_deduced) % 2
    residual_support = set(np.where(residual == 1)[0])
    if np.any(residual):
        print(f"  Residual (e + e_deduced) mod 2: weight={len(residual_support)}, "
              f"support={sorted(residual_support)}")
        print(f"\nDecode result: LOGICAL ERROR")
    else:
        print(f"  e_deduced == e? True")
        print(f"\nDecode result: SUCCESS")


def walk_through_soft_bitflip(code, p, seed=42):
    """
    Generate a single error sample and walk through every step of
    the soft bit-flip decoding algorithm, printing intermediate state.
    """
    np.random.seed(seed)
    decoder = SoftBitFlipDecoder(code)
    b = np.log((1 - p) / p)

    print("=" * 70)
    print(f"Code: {code.id}  n={code.n}  m={code.m}  dv={code.dv}  dc={code.dc}")
    print(f"Soft bit-flip: p={p}, b=log((1-p)/p)={b:.4f}")
    print(f"E_v = -b*(1-2*x_v) + sum_{{c in N(v)}} (2*s_c - 1)")
    print("=" * 70)

    # --- Generate error ---
    err = (np.random.random(code.n) < p).astype(int)
    error_support = set(np.where(err == 1)[0])
    print(f"\nError: weight={len(error_support)}, support={sorted(error_support)}")

    # --- Compute syndrome ---
    syndrome = compute_synd(code, err)
    unsatisfied = set(np.where(syndrome == 1)[0])
    print(f"Syndrome: weight={len(unsatisfied)}, unsatisfied checks={sorted(unsatisfied)}")

    # --- Initialize ---
    synd = syndrome.copy()
    e_deduced = np.zeros(code.n, dtype=int)
    synd_score = np.zeros(code.n, dtype=float)
    for v in range(code.n):
        for c in code.bit_nbhd[v]:
            synd_score[v] += 2 * synd[c] - 1
    E = -b + synd_score.copy()

    # --- Diagnostic: E_v breakdown for error bits vs clean bits ---
    error_mask = np.array([v in error_support for v in range(code.n)])
    clean_mask = ~error_mask

    print(f"\n--- Initial E_v diagnostics ---")
    print(f"  Prior term: -b = {-b:.4f}  (variable must have synd_score > {b:.4f} to be flippable)")
    print(f"  Hard threshold for comparison: unsat_count >= {code.dv // 2 + 1} "
          f"(synd_score >= {2 * (code.dv // 2 + 1) - code.dv})")
    print(f"  Max possible synd_score: {code.dv} (all {code.dv} neighbors unsatisfied)")

    print(f"\n  Error bits ({len(error_support)}):")
    for v in sorted(error_support):
        unsat = sum(1 for c in code.bit_nbhd[v] if synd[c] == 1)
        print(f"    v={v:3d}: synd_score={synd_score[v]:+.1f} "
              f"(unsat={unsat}/{code.dv}), "
              f"prior={-b:.4f}, E_v={E[v]:+.4f} "
              f"{'FLIPPABLE' if E[v] > 0 else ''}")

    print(f"\n  Clean bits (top 10 by E_v):")
    clean_indices = np.where(clean_mask)[0]
    top_clean = clean_indices[np.argsort(E[clean_indices])[::-1][:10]]
    for v in top_clean:
        unsat = sum(1 for c in code.bit_nbhd[v] if synd[c] == 1)
        print(f"    v={v:3d}: synd_score={synd_score[v]:+.1f} "
              f"(unsat={unsat}/{code.dv}), "
              f"prior={-b:.4f}, E_v={E[v]:+.4f} "
              f"{'FLIPPABLE' if E[v] > 0 else ''}")

    print(f"\n  Summary: E_v > 0 among error bits: {np.sum(E[error_mask] > 0)}/{len(error_support)}, "
          f"among clean bits: {np.sum(E[clean_mask] > 0)}/{np.sum(clean_mask)}")

    # --- Iterative flipping ---
    print(f"\n--- Soft bit-flip iterations ---")
    max_iters = code.n

    for iteration in range(1, max_iters + 1):
        v = np.argmax(E)
        if E[v] <= 0:
            # Show what the top candidates look like at stopping
            print(f"\n  No flippable variable found (max E={E[v]:.4f}). Stopping.")
            top5 = np.argsort(E)[::-1][:5]
            print(f"  Top 5 candidates at stop:")
            for rank, u in enumerate(top5):
                in_err = "in error" if u in error_support else "clean"
                unsat = sum(1 for c in code.bit_nbhd[u] if synd[c] == 1)
                prior = -b * (1 - 2 * e_deduced[u])
                print(f"    #{rank+1} v={u:3d} ({in_err}): E_v={E[u]:+.4f}, "
                      f"synd_score={synd_score[u]:+.1f} (unsat={unsat}/{code.dv}), "
                      f"prior={prior:+.4f}, x_v={e_deduced[u]}")
            break

        in_error = "in error" if v in error_support else "clean"
        currently_flipped = "unflip" if e_deduced[v] == 1 else "flip"

        if iteration <= 30:
            prior = -b * (1 - 2 * e_deduced[v])
            print(f"  iter {iteration}: {currently_flipped} v={v} ({in_error}), "
                  f"E_v={E[v]:.4f}, synd_score={synd_score[v]:.1f}, "
                  f"prior={prior:+.4f}, x_v={e_deduced[v]}")

        # Flip
        e_deduced[v] ^= 1
        if e_deduced[v] == 1:
            E[v] += 2 * b
        else:
            E[v] -= 2 * b

        for c in code.bit_nbhd[v]:
            if synd[c] == 1:
                synd[c] = 0
                for u in code.check_nbhd[c]:
                    synd_score[u] -= 2
                    E[u] -= 2
            else:
                synd[c] = 1
                for u in code.check_nbhd[c]:
                    synd_score[u] += 2
                    E[u] += 2

        remaining_syndrome = int(np.sum(synd))
        if iteration <= 30:
            # Show how many error vs clean bits are now flippable
            err_flippable = np.sum(E[error_mask] > 0)
            clean_flippable = np.sum(E[clean_mask] > 0)
            print(f"         remaining syndrome weight={remaining_syndrome}, "
                  f"max E={np.max(E):.4f}, "
                  f"flippable: {err_flippable} error + {clean_flippable} clean")
        elif iteration == 31:
            print(f"  ... (suppressing further iterations)")

    # --- Result ---
    converged = not np.any(synd)
    flipped_support = set(np.where(e_deduced == 1)[0])

    print(f"\n--- Result after {iteration} iterations ---")
    print(f"  Converged (syndrome cleared): {converged}")
    print(f"  Flipped bits: weight={len(flipped_support)}, support={sorted(flipped_support)}")
    print(f"  Actual error: weight={len(error_support)}, support={sorted(error_support)}")

    if not converged:
        remaining = set(np.where(synd == 1)[0])
        print(f"  Remaining syndrome weight: {len(remaining)}")
        print(f"\nDecode result: FAIL (syndrome not cleared)")
        return

    residual = (err + e_deduced) % 2
    residual_support = set(np.where(residual == 1)[0])
    if np.any(residual):
        print(f"  Residual (e + e_deduced) mod 2: weight={len(residual_support)}, "
              f"support={sorted(residual_support)}")
        print(f"\nDecode result: LOGICAL ERROR")
    else:
        print(f"  e_deduced == e? True")
        print(f"\nDecode result: SUCCESS")


if __name__ == "__main__":
    mode = "soft_bitflip"
    codes = load_codes([(120, 100)])
    if not codes:
        print("No codes found")
    else:
        if mode == "viderman":
            print("\n\n===== VIDERMAN DECODER =====\n")
            walk_through_decode(codes[0], p=0.065, h=4, seed=42)
        elif mode == "bitflip":
            print("\n\n===== BIT-FLIP DECODER =====\n")
            walk_through_bitflip(codes[0], p=0.05, seed=42)
        elif mode == "soft_bitflip":
            print("\n\n===== SOFT BIT-FLIP DECODER =====\n")
            walk_through_soft_bitflip(codes[0], p=0.03, seed=42)
