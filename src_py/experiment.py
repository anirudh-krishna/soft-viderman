import logging
import json
import os
import numpy as np
from utils import compute_synd, wilson_ci

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def load_existing_results(code_id):
    """
    Load existing results from JSON for a given code, if the file exists.

    Returns:
        dict with code metadata and results list, or None if no file exists.
    """
    out_path = os.path.join(OUTPUT_DIR, f"{code_id}.json")
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            return json.load(f)
    return None


def run_experiment(decoder, code, p_list, eps_list, ci_target=0.01, batch_size=100,
                   min_samples=200, max_samples=100000):
    """
    Run decoding experiments with adaptive sampling for a single decoder and code.
    Merges with existing data: accumulates n_samples and n_failures across runs.

    Args:
        decoder: a Decoder subclass instance (must have .label and .decode())
        code: Classical_code instance
        p_list: list of error probabilities
        eps_list: list of eps values (threshold h = (1-2*eps)*dv)
        ci_target: stop when relative CI (hw / failure_rate) < ci_target
        batch_size: number of samples per batch
        min_samples: minimum number of samples before checking CI
        max_samples: hard cap on total samples per (code, p, eps) point

    Returns:
        dict with code metadata and results list
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    algorithm_label = decoder.label
    logger.info("Code %s: n=%d, m=%d, dv=%d, dc=%d, algorithm=%s",
                code.id, code.n, code.m, code.dv, code.dc, algorithm_label)

    # Load existing results and index by (p, eps, algorithm)
    existing = load_existing_results(code.id)
    existing_by_key = {}
    if existing is not None:
        for r in existing.get("results", []):
            key = (r["p"], r["eps"], r["algorithm"])
            existing_by_key[key] = r

    code_results = []

    for eps in eps_list:
        h = (1.0 - 2.0 * eps) * code.dv
        for p in p_list:
            key = (p, eps, algorithm_label)

            # Start from existing counts if available
            if key in existing_by_key:
                prev = existing_by_key[key]
                n_failures = prev["n_failures"]
                n_total = prev["n_samples"]
                logger.info("  Resuming eps=%.4f p=%.4f from n_samples=%d, n_failures=%d",
                            eps, p, n_total, n_failures)
            else:
                n_failures = 0
                n_total = 0

            # Check if relative CI is already met
            failure_rate, ci_hw = wilson_ci(n_failures, n_total)
            rel_ci = ci_hw / failure_rate if failure_rate > 0 else float("inf")
            if n_total >= min_samples and rel_ci < ci_target:
                logger.info("  eps=%.4f h=%.2f p=%.4f: already converged, failure_rate=%.6f n_samples=%d",
                            eps, h, p, failure_rate, n_total)
            else:
                log_interval = max(1000, batch_size * 10)
                next_log = n_total + log_interval
                while True:
                    for _ in range(batch_size):
                        err = (np.random.random(code.n) < p).astype(int)
                        syndrome = compute_synd(code, err)
                        success = decoder.decode(syndrome, eps)
                        if not success:
                            n_failures += 1
                        n_total += 1

                    failure_rate, ci_hw = wilson_ci(n_failures, n_total)
                    rel_ci = ci_hw / failure_rate if failure_rate > 0 else float("inf")

                    if n_total >= next_log:
                        logger.info("    [progress] eps=%.4f p=%.4f: n_samples=%d failure_rate=%.6f rel_CI=%.4f",
                                    eps, p, n_total, failure_rate, rel_ci)
                        next_log = n_total + log_interval

                    if n_total >= max_samples:
                        logger.warning("  eps=%.4f p=%.4f: hit max_samples=%d, rel_CI=%.4f",
                                       eps, p, max_samples, rel_ci)
                        break
                    if n_total >= min_samples and rel_ci < ci_target:
                        break

                logger.info("  eps=%.4f h=%.2f p=%.4f: failure_rate=%.6f rel_CI=%.4f n_samples=%d",
                            eps, h, p, failure_rate, rel_ci, n_total)

            code_results.append({
                "p": p,
                "eps": eps,
                "n_samples": n_total,
                "n_failures": n_failures,
                "failure_rate": n_failures / n_total if n_total > 0 else 0.0,
                "algorithm": algorithm_label,
            })

            # Remove from existing index so leftovers can be preserved
            existing_by_key.pop(key, None)

    # Preserve any existing results for (p, eps, algorithm) combos not in this run
    for r in existing_by_key.values():
        code_results.append(r)

    result_data = {
        "code_id": code.id,
        "n": code.n,
        "m": code.m,
        "dv": code.dv,
        "dc": code.dc,
        "results": code_results,
    }

    # Save per-code JSON
    out_path = os.path.join(OUTPUT_DIR, f"{code.id}.json")
    with open(out_path, "w") as f:
        json.dump(result_data, f, indent=2)
    logger.info("Saved results to %s", out_path)

    return result_data
