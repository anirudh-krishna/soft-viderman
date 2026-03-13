import logging
import json
import os
import numpy as np
from utils import compute_synd, wilson_ci

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def find_optimal_h(decoder, code, p, ci_target=0.01, batch_size=100,
                   min_samples=200, max_samples=100000):
    """
    Find the value of h in {0, 1, ..., dv} that minimizes failure rate for a given (code, p).

    Args:
        decoder: a Decoder subclass instance
        code: Classical_code instance
        p: error probability
        ci_target: stop when relative CI (hw / failure_rate) < ci_target
        batch_size: number of samples per batch
        min_samples: minimum number of samples before checking CI
        max_samples: hard cap on total samples per h value

    Returns:
        dict with keys: optimal_h, optimal_failure_rate, all_h_results (list of per-h dicts)
    """
    h_results = []

    for h in range(code.dv + 1):
        n_failures = 0
        n_total = 0

        log_interval = max(1000, batch_size * 10)
        next_log = n_total + log_interval

        while True:
            for _ in range(batch_size):
                err = (np.random.random(code.n) < p).astype(int)
                syndrome = compute_synd(code, err)
                success = decoder.decode(syndrome, h)
                if not success:
                    n_failures += 1
                n_total += 1

            failure_rate, ci_hw = wilson_ci(n_failures, n_total)
            rel_ci = ci_hw / failure_rate if failure_rate > 0 else float("inf")

            if n_total >= next_log:
                logger.info("    [progress] h=%d p=%.4f: n_samples=%d failure_rate=%.6f rel_CI=%.4f",
                            h, p, n_total, failure_rate, rel_ci)
                next_log = n_total + log_interval

            if n_total >= max_samples:
                logger.warning("  h=%d p=%.4f: hit max_samples=%d, rel_CI=%.4f",
                               h, p, max_samples, rel_ci)
                break
            if n_total >= min_samples and rel_ci < ci_target:
                break

        logger.info("  h=%d p=%.4f: failure_rate=%.6f rel_CI=%.4f n_samples=%d",
                    h, p, failure_rate, rel_ci, n_total)

        h_results.append({
            "h": h,
            "p": p,
            "n_samples": n_total,
            "n_failures": n_failures,
            "failure_rate": failure_rate,
            "algorithm": decoder.label,
        })

    # Find the h with minimum failure rate
    best = min(h_results, key=lambda r: r["failure_rate"])
    logger.info("  Optimal: h=%d, failure_rate=%.6f for p=%.4f", best["h"], best["failure_rate"], p)

    return {
        "optimal_h": best["h"],
        "optimal_failure_rate": best["failure_rate"],
        "all_h_results": h_results,
    }


def run_optimize_h(decoder, code, p_list, ci_target=0.01, batch_size=100,
                   min_samples=200, max_samples=100000):
    """
    Find optimal h for each p in p_list. Save results to JSON.

    Args:
        decoder: a Decoder subclass instance
        code: Classical_code instance
        p_list: list of error probabilities
        ci_target, batch_size, min_samples, max_samples: passed to find_optimal_h

    Returns:
        dict with code metadata and optimization results
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Optimizing h for code %s (n=%d, m=%d, dv=%d, dc=%d, algorithm=%s)",
                code.id, code.n, code.m, code.dv, code.dc, decoder.label)

    optimal_results = []
    all_details = []

    for p in p_list:
        logger.info("p=%.4f:", p)
        result = find_optimal_h(decoder, code, p, ci_target=ci_target,
                                batch_size=batch_size, min_samples=min_samples,
                                max_samples=max_samples)
        optimal_results.append({
            "p": p,
            "optimal_h": result["optimal_h"],
            "optimal_failure_rate": result["optimal_failure_rate"],
            "algorithm": decoder.label,
        })
        all_details.extend(result["all_h_results"])

    result_data = {
        "code_id": code.id,
        "n": code.n,
        "m": code.m,
        "dv": code.dv,
        "dc": code.dc,
        "optimal": optimal_results,
        "details": all_details,
    }

    out_path = os.path.join(OUTPUT_DIR, f"{code.id}_optimal_h.json")
    with open(out_path, "w") as f:
        json.dump(result_data, f, indent=2)
    logger.info("Saved optimal h results to %s", out_path)

    return result_data
