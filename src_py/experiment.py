import logging
import json
import os
import sys
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


def run_experiment(decoder, code, p_list, h_list, ci_target=0.01, batch_size=100,
                   min_samples=200, max_samples=100000):
    """
    Run decoding experiments with adaptive sampling for a single decoder and code.
    Merges with existing data: accumulates n_samples and n_failures across runs.

    Args:
        decoder: a Decoder subclass instance (must have .label and .decode())
        code: Classical_code instance
        p_list: list of error probabilities
        h_list: list of integer thresholds for the decoder
        ci_target: stop when relative CI (hw / failure_rate) < ci_target
        batch_size: number of samples per batch
        min_samples: minimum number of samples before checking CI
        max_samples: hard cap on total samples per (code, p, h) point

    Returns:
        dict with code metadata and results list
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    algorithm_label = decoder.label
    logger.info("Code %s: n=%d, m=%d, dv=%d, dc=%d, algorithm=%s",
                code.id, code.n, code.m, code.dv, code.dc, algorithm_label)

    # Load existing results and index by (p, h, algorithm)
    existing = load_existing_results(code.id)
    existing_by_key = {}
    if existing is not None:
        for r in existing.get("results", []):
            key = (r["p"], r["h"], r["algorithm"])
            existing_by_key[key] = r

    code_results = []

    for h in h_list:
        for p in p_list:
            key = (p, h, algorithm_label)

            # Start from existing counts if available
            if key in existing_by_key:
                prev = existing_by_key[key]
                n_failures = prev["n_failures"]
                n_total = prev["n_samples"]
                logger.info("  Resuming h=%d p=%.4f from n_samples=%d, n_failures=%d",
                            h, p, n_total, n_failures)
            else:
                n_failures = 0
                n_total = 0

            # Check if relative CI is already met
            failure_rate, ci_hw = wilson_ci(n_failures, n_total)
            rel_ci = ci_hw / failure_rate if failure_rate > 0 else float("inf")
            if n_total >= min_samples and rel_ci < ci_target:
                logger.info("  h=%d p=%.4f: already converged, failure_rate=%.6f n_samples=%d",
                            h, p, failure_rate, n_total)
            else:
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

                    bar_width = 30
                    progress = min(n_total / max_samples, 1.0)
                    filled = int(bar_width * progress)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    status = (f"\r  h={h} p={p:.4f} [{bar}] {n_total}/{max_samples} "
                              f"fr={failure_rate:.4f} rCI={rel_ci:.4f}")
                    sys.stderr.write(status)
                    sys.stderr.flush()

                    if n_total >= max_samples:
                        sys.stderr.write("\r" + " " * 120 + "\r")
                        sys.stderr.flush()
                        logger.warning("  h=%d p=%.4f: hit max_samples=%d, rel_CI=%.4f",
                                       h, p, max_samples, rel_ci)
                        break
                    if n_total >= min_samples and rel_ci < ci_target:
                        break

                sys.stderr.write("\r" + " " * 120 + "\r")
                sys.stderr.flush()
                logger.info("  h=%d p=%.4f: failure_rate=%.6f rel_CI=%.4f n_samples=%d",
                            h, p, failure_rate, rel_ci, n_total)

            code_results.append({
                "p": p,
                "h": h,
                "n_samples": n_total,
                "n_failures": n_failures,
                "failure_rate": n_failures / n_total if n_total > 0 else 0.0,
                "algorithm": algorithm_label,
            })

            # Remove from existing index so leftovers can be preserved
            existing_by_key.pop(key, None)

    # Preserve any existing results for (p, h, algorithm) combos not in this run
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


def run_experiment_bitflip(decoder, code, p_list, ci_target=0.01, batch_size=100,
                           min_samples=200, max_samples=100000):
    """
    Run bit-flip decoding experiments with adaptive sampling.
    Tracks decode failures (nonzero syndrome) and logical errors (wrong codeword) separately.

    Args:
        decoder: BitFlipDecoder instance
        code: Classical_code instance
        p_list: list of error probabilities
        ci_target: stop when relative CI of total failure rate < ci_target
        batch_size: number of samples per batch
        min_samples: minimum samples before checking CI
        max_samples: hard cap on total samples per (code, p)

    Returns:
        dict with code metadata and results list
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    algorithm_label = decoder.label
    logger.info("Code %s: n=%d, m=%d, dv=%d, dc=%d, algorithm=%s",
                code.id, code.n, code.m, code.dv, code.dc, algorithm_label)

    # Load existing results and index by (p, algorithm)
    out_path = os.path.join(OUTPUT_DIR, f"{code.id}_bitflip.json")
    existing_by_key = {}
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            existing = json.load(f)
        for r in existing.get("results", []):
            key = (r["p"], r["algorithm"])
            existing_by_key[key] = r

    code_results = []

    for p in p_list:
        key = (p, algorithm_label)

        # Resume from existing counts if available
        if key in existing_by_key:
            prev = existing_by_key[key]
            n_decode_failures = prev["n_decode_failures"]
            n_logical_errors = prev["n_logical_errors"]
            n_total = prev["n_samples"]
            logger.info("  Resuming p=%.4f from n_samples=%d, n_decode_failures=%d, n_logical_errors=%d",
                        p, n_total, n_decode_failures, n_logical_errors)
        else:
            n_decode_failures = 0
            n_logical_errors = 0
            n_total = 0

        # Check if relative CI is already met (on total failure count)
        n_failures_total = n_decode_failures + n_logical_errors
        failure_rate, ci_hw = wilson_ci(n_failures_total, n_total)
        rel_ci = ci_hw / failure_rate if failure_rate > 0 else float("inf")
        if n_total >= min_samples and rel_ci < ci_target:
            logger.info("  p=%.4f: already converged, failure_rate=%.6f n_samples=%d",
                        p, failure_rate, n_total)
        else:
            log_interval = max(1000, batch_size * 10)
            next_log = n_total + log_interval
            while True:
                for _ in range(batch_size):
                    err = (np.random.random(code.n) < p).astype(int)
                    syndrome = compute_synd(code, err)
                    e_deduced, converged = decoder.decode(syndrome)
                    if not converged:
                        n_decode_failures += 1
                    elif np.any((err + e_deduced) % 2):
                        n_logical_errors += 1
                    n_total += 1

                n_failures_total = n_decode_failures + n_logical_errors
                failure_rate, ci_hw = wilson_ci(n_failures_total, n_total)
                rel_ci = ci_hw / failure_rate if failure_rate > 0 else float("inf")

                bar_width = 30
                progress = min(n_total / max_samples, 1.0)
                filled = int(bar_width * progress)
                bar = "█" * filled + "░" * (bar_width - filled)
                status = (f"\r  p={p:.4f} [{bar}] {n_total}/{max_samples} "
                          f"fr={failure_rate:.4f} rCI={rel_ci:.4f}")
                sys.stderr.write(status)
                sys.stderr.flush()

                if n_total >= max_samples:
                    sys.stderr.write("\r" + " " * len(status) + "\r")
                    sys.stderr.flush()
                    logger.warning("  p=%.4f: hit max_samples=%d, rel_CI=%.4f", p, max_samples, rel_ci)
                    break
                if n_total >= min_samples and rel_ci < ci_target:
                    break

            sys.stderr.write("\r" + " " * 120 + "\r")
            sys.stderr.flush()
            logger.info("  p=%.4f: failure_rate=%.6f decode_fail=%d logical_err=%d rel_CI=%.4f n_samples=%d",
                        p, failure_rate, n_decode_failures, n_logical_errors, rel_ci, n_total)

        code_results.append({
            "p": p,
            "n_samples": n_total,
            "n_decode_failures": n_decode_failures,
            "n_logical_errors": n_logical_errors,
            "decode_failure_rate": n_decode_failures / n_total if n_total > 0 else 0.0,
            "logical_error_rate": n_logical_errors / n_total if n_total > 0 else 0.0,
            "algorithm": algorithm_label,
        })

        existing_by_key.pop(key, None)

    # Preserve existing results for p values not in this run
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

    with open(out_path, "w") as f:
        json.dump(result_data, f, indent=2)
    logger.info("Saved results to %s", out_path)

    return result_data
