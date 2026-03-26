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
                   min_samples=200, max_samples=5000):
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
                           min_samples=200, max_samples=5000):
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
    out_path = os.path.join(OUTPUT_DIR, f"{code.id}_{algorithm_label}.json")
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
                    e_deduced, converged = decoder.decode(syndrome, p=p)
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


def run_experiment_soft_bitflip(decoder, code, p_list, alpha_list, ci_target=0.01,
                                batch_size=100, min_samples=200, max_samples=5000):
    """
    Run soft bit-flip decoding experiments sweeping over (p, alpha) pairs.
    Tracks decode failures and logical errors separately.

    Args:
        decoder: SoftBitFlipDecoder instance
        code: Classical_code instance
        p_list: list of error probabilities
        alpha_list: list of alpha values (syndrome weight)
        ci_target: stop when relative CI of total failure rate < ci_target
        batch_size: number of samples per batch
        min_samples: minimum samples before checking CI
        max_samples: hard cap on total samples per (code, p, alpha)

    Returns:
        dict with code metadata and results list
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    algorithm_label = decoder.label
    logger.info("Code %s: n=%d, m=%d, dv=%d, dc=%d, algorithm=%s",
                code.id, code.n, code.m, code.dv, code.dc, algorithm_label)

    # Load existing results and index by (p, alpha, algorithm)
    out_path = os.path.join(OUTPUT_DIR, f"{code.id}_{algorithm_label}.json")
    existing_by_key = {}
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            existing = json.load(f)
        for r in existing.get("results", []):
            key = (r["p"], r["alpha"], r["algorithm"])
            existing_by_key[key] = r

    code_results = []

    for alpha in alpha_list:
        for p in p_list:
            key = (p, alpha, algorithm_label)

            # Resume from existing counts if available
            if key in existing_by_key:
                prev = existing_by_key[key]
                n_decode_failures = prev["n_decode_failures"]
                n_logical_errors = prev["n_logical_errors"]
                n_total = prev["n_samples"]
                logger.info("  Resuming alpha=%.3f p=%.4f from n_samples=%d", alpha, p, n_total)
            else:
                n_decode_failures = 0
                n_logical_errors = 0
                n_total = 0

            # Check if relative CI is already met
            n_failures_total = n_decode_failures + n_logical_errors
            failure_rate, ci_hw = wilson_ci(n_failures_total, n_total)
            rel_ci = ci_hw / failure_rate if failure_rate > 0 else float("inf")
            if n_total >= min_samples and rel_ci < ci_target:
                logger.info("  alpha=%.3f p=%.4f: already converged, failure_rate=%.6f n_samples=%d",
                            alpha, p, failure_rate, n_total)
            else:
                while True:
                    for _ in range(batch_size):
                        err = (np.random.random(code.n) < p).astype(int)
                        syndrome = compute_synd(code, err)
                        e_deduced, converged = decoder.decode(syndrome, p=p, alpha=alpha)
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
                    status = (f"\r  α={alpha:.3f} p={p:.4f} [{bar}] {n_total}/{max_samples} "
                              f"fr={failure_rate:.4f} rCI={rel_ci:.4f}")
                    sys.stderr.write(status)
                    sys.stderr.flush()

                    if n_total >= max_samples:
                        sys.stderr.write("\r" + " " * 120 + "\r")
                        sys.stderr.flush()
                        logger.warning("  alpha=%.3f p=%.4f: hit max_samples=%d, rel_CI=%.4f",
                                       alpha, p, max_samples, rel_ci)
                        break
                    if n_total >= min_samples and rel_ci < ci_target:
                        break

                sys.stderr.write("\r" + " " * 120 + "\r")
                sys.stderr.flush()
                logger.info("  alpha=%.3f p=%.4f: failure_rate=%.6f decode_fail=%d logical_err=%d "
                            "rel_CI=%.4f n_samples=%d",
                            alpha, p, failure_rate, n_decode_failures, n_logical_errors, rel_ci, n_total)

            code_results.append({
                "p": p,
                "alpha": alpha,
                "n_samples": n_total,
                "n_decode_failures": n_decode_failures,
                "n_logical_errors": n_logical_errors,
                "decode_failure_rate": n_decode_failures / n_total if n_total > 0 else 0.0,
                "logical_error_rate": n_logical_errors / n_total if n_total > 0 else 0.0,
                "algorithm": algorithm_label,
            })

            existing_by_key.pop(key, None)

    # Preserve existing results not in this run
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


def run_experiment_compare(decoders, code, p_list, ci_target=0.01, batch_size=100,
                           min_samples=200, max_samples=5000):
    """
    Run multiple decoders on the same error samples for fair comparison.
    All decoders see identical errors at each (p, sample_index).

    Args:
        decoders: list of decoder instances
        code: Classical_code instance
        p_list: list of error probabilities
        ci_target: stop when the worst-converged decoder meets CI target
        batch_size: number of samples per batch
        min_samples: minimum samples before checking CI
        max_samples: hard cap on total samples per (code, p)

    Returns:
        dict mapping decoder.label -> {code metadata + results list}
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels = [d.label for d in decoders]
    logger.info("Code %s: n=%d, m=%d, dv=%d, dc=%d, comparing %s",
                code.id, code.n, code.m, code.dv, code.dc, labels)

    # Per-decoder accumulators
    accum = {label: {} for label in labels}
    for label in labels:
        for p in p_list:
            accum[label][p] = {"n_total": 0, "n_decode_failures": 0, "n_logical_errors": 0}

    for p in p_list:
        n_total = 0
        while True:
            for _ in range(batch_size):
                err = (np.random.random(code.n) < p).astype(int)
                syndrome = compute_synd(code, err)

                for decoder in decoders:
                    e_deduced, converged = decoder.decode(syndrome, p=p)
                    if not converged:
                        accum[decoder.label][p]["n_decode_failures"] += 1
                    elif np.any((err + e_deduced) % 2):
                        accum[decoder.label][p]["n_logical_errors"] += 1
                    accum[decoder.label][p]["n_total"] += 1

                n_total += 1

            # Check CI: use the worst (largest) rel_ci across decoders
            max_rel_ci = 0
            worst_fr = 0
            for label in labels:
                a = accum[label][p]
                nf = a["n_decode_failures"] + a["n_logical_errors"]
                fr, hw = wilson_ci(nf, a["n_total"])
                rel = hw / fr if fr > 0 else float("inf")
                if rel > max_rel_ci:
                    max_rel_ci = rel
                    worst_fr = fr

            bar_width = 30
            progress = min(n_total / max_samples, 1.0)
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            status = (f"\r  p={p:.4f} [{bar}] {n_total}/{max_samples} "
                      f"worst_fr={worst_fr:.4f} worst_rCI={max_rel_ci:.4f}")
            sys.stderr.write(status)
            sys.stderr.flush()

            if n_total >= max_samples:
                sys.stderr.write("\r" + " " * 120 + "\r")
                sys.stderr.flush()
                logger.warning("  p=%.4f: hit max_samples=%d", p, max_samples)
                break
            if n_total >= min_samples and max_rel_ci < ci_target:
                break

        sys.stderr.write("\r" + " " * 120 + "\r")
        sys.stderr.flush()
        for label in labels:
            a = accum[label][p]
            nf = a["n_decode_failures"] + a["n_logical_errors"]
            fr, hw = wilson_ci(nf, a["n_total"])
            rel = hw / fr if fr > 0 else float("inf")
            logger.info("  p=%.4f %s: failure_rate=%.6f decode_fail=%d logical_err=%d "
                        "rel_CI=%.4f n_samples=%d",
                        p, label, fr, a["n_decode_failures"], a["n_logical_errors"],
                        rel, a["n_total"])

    # Build per-decoder result dicts and save
    all_results = {}
    for decoder in decoders:
        label = decoder.label
        code_results = []
        for p in p_list:
            a = accum[label][p]
            nt = a["n_total"]
            code_results.append({
                "p": p,
                "n_samples": nt,
                "n_decode_failures": a["n_decode_failures"],
                "n_logical_errors": a["n_logical_errors"],
                "decode_failure_rate": a["n_decode_failures"] / nt if nt > 0 else 0.0,
                "logical_error_rate": a["n_logical_errors"] / nt if nt > 0 else 0.0,
                "algorithm": label,
            })

        result_data = {
            "code_id": code.id,
            "n": code.n,
            "m": code.m,
            "dv": code.dv,
            "dc": code.dc,
            "results": code_results,
        }

        out_path = os.path.join(OUTPUT_DIR, f"{code.id}_{label}.json")
        with open(out_path, "w") as f:
            json.dump(result_data, f, indent=2)
        logger.info("Saved results to %s", out_path)

        all_results[label] = result_data

    return all_results
