import logging
import os
import matplotlib.pyplot as plt
from utils import wilson_ci

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


def plot_results(all_results, p_list):
    """
    Plot failure rate vs p with 95% Wilson confidence intervals,
    one curve per (code, h) pair.
    Legend: "{code.id}-{h}"
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    for code_id, data in all_results.items():
        results = data["results"]

        # Group by h
        h_vals = sorted(set(r["h"] for r in results))
        for h in h_vals:
            curve_data = [r for r in results if r["h"] == h]
            curve_data.sort(key=lambda r: r["p"])

            ps = []
            frs = []
            ci_lo = []
            ci_hi = []
            for r in curve_data:
                fr, hw = wilson_ci(r["n_failures"], r["n_samples"])
                ps.append(r["p"])
                frs.append(fr)
                ci_lo.append(max(fr - hw, 0.0))
                ci_hi.append(fr + hw)

            label = f"{code_id}-h={h}"
            yerr_lo = [fr - lo for fr, lo in zip(frs, ci_lo)]
            yerr_hi = [hi - fr for fr, hi in zip(frs, ci_hi)]
            ax.errorbar(ps, frs, yerr=[yerr_lo, yerr_hi], marker="o", capsize=4, label=label)

    ax.set_xlabel("p (error probability)")
    ax.set_ylabel("Failure rate")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize="small")
    ax.set_title("Viderman Decoding: Failure Rate vs Error Probability")
    ax.grid(True, which="both", alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, "failure_rate.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to %s", plot_path)
    plt.show()
    plt.close(fig)


def plot_optimal_h(all_optimal_results):
    """
    Plot optimal h vs p, one curve per code.
    Legend: "{code.id}"

    Args:
        all_optimal_results: dict mapping code_id -> result data from run_optimize_h
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    for code_id, data in all_optimal_results.items():
        optimal = data["optimal"]
        optimal = sorted(optimal, key=lambda r: r["p"])
        ps = [r["p"] for r in optimal]
        hs = [r["optimal_h"] for r in optimal]
        ax.plot(ps, hs, marker="o", label=code_id)

    ax.set_xlabel("p (error probability)")
    ax.set_ylabel("Optimal h")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize="small")
    ax.set_title("Optimal Threshold h vs Error Probability")
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, "optimal_h.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to %s", plot_path)
    plt.show()
    plt.close(fig)


def plot_bitflip_results(all_results, show_logical=False):
    """
    Plot decode failure rate (and optionally logical error rate) vs p.

    Args:
        all_results: dict mapping code_id -> result data from run_experiment_bitflip
        show_logical: if True, also plot logical error rate curves
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    for code_id, data in all_results.items():
        results = sorted(data["results"], key=lambda r: r["p"])
        code_label = f"{code_id} (n={data['n']}, m={data['m']}, dv={data['dv']}, dc={data['dc']})"

        ps = [r["p"] for r in results]

        # Decode failure rate curve
        decode_frs = [r["decode_failure_rate"] for r in results]
        decode_cis = [wilson_ci(r["n_decode_failures"], r["n_samples"]) for r in results]
        decode_lo = [max(fr - hw, 0.0) for fr, (_, hw) in zip(decode_frs, decode_cis)]
        decode_hi = [fr + hw for fr, (_, hw) in zip(decode_frs, decode_cis)]
        ax.errorbar(ps, decode_frs,
                    yerr=[[fr - lo for fr, lo in zip(decode_frs, decode_lo)],
                          [hi - fr for fr, hi in zip(decode_frs, decode_hi)]],
                    marker="o", capsize=4, label=f"{code_label} decode fail")

        if show_logical:
            # Logical error rate curve
            logical_frs = [r["logical_error_rate"] for r in results]
            logical_cis = [wilson_ci(r["n_logical_errors"], r["n_samples"]) for r in results]
            logical_lo = [max(fr - hw, 0.0) for fr, (_, hw) in zip(logical_frs, logical_cis)]
            logical_hi = [fr + hw for fr, (_, hw) in zip(logical_frs, logical_cis)]
            ax.errorbar(ps, logical_frs,
                        yerr=[[fr - lo for fr, lo in zip(logical_frs, logical_lo)],
                              [hi - fr for fr, hi in zip(logical_frs, logical_hi)]],
                        marker="s", capsize=4, linestyle="--", label=f"{code_label} logical err")

    all_ps = [r["p"] for data in all_results.values() for r in data["results"]]
    ax.set_xlabel("p (error probability)")
    ax.set_ylabel("Rate")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlim(0, max(all_ps) * 1.05)
    ax.set_ylim(0, 1)
    ax.legend(fontsize="small")
    ax.set_title("Bit-Flip Decoding: Decode Failure & Logical Error Rate vs p")
    ax.grid(True, which="both", alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, "bitflip_results.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to %s", plot_path)
    plt.show()
    plt.close(fig)


def plot_compare_results(all_results_by_algo, show_logical=False):
    """
    Plot comparison of multiple decoders: decode failure rate (and optionally logical error rate) vs p.

    Args:
        all_results_by_algo: dict mapping algorithm_label -> {code_id -> result data}
        show_logical: if True, also plot logical error rate curves
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all code_ids across algorithms
    code_ids = set()
    for algo_results in all_results_by_algo.values():
        code_ids.update(algo_results.keys())

    for code_id in sorted(code_ids):
        fig, ax = plt.subplots(figsize=(10, 7))
        code_label = None

        for algo, algo_results in all_results_by_algo.items():
            if code_id not in algo_results:
                continue
            data = algo_results[code_id]
            if code_label is None:
                code_label = f"{code_id} (n={data['n']}, m={data['m']}, dv={data['dv']}, dc={data['dc']})"

            results = sorted(data["results"], key=lambda r: r["p"])
            ps = [r["p"] for r in results]

            # Decode failure rate
            decode_frs = [r["decode_failure_rate"] for r in results]
            decode_cis = [wilson_ci(r["n_decode_failures"], r["n_samples"]) for r in results]
            decode_lo = [max(fr - hw, 0.0) for fr, (_, hw) in zip(decode_frs, decode_cis)]
            decode_hi = [fr + hw for fr, (_, hw) in zip(decode_frs, decode_cis)]
            ax.errorbar(ps, decode_frs,
                        yerr=[[fr - lo for fr, lo in zip(decode_frs, decode_lo)],
                              [hi - fr for fr, hi in zip(decode_frs, decode_hi)]],
                        marker="o", capsize=4, label=f"{algo} decode fail")

            if show_logical:
                # Logical error rate
                logical_frs = [r["logical_error_rate"] for r in results]
                logical_cis = [wilson_ci(r["n_logical_errors"], r["n_samples"]) for r in results]
                logical_lo = [max(fr - hw, 0.0) for fr, (_, hw) in zip(logical_frs, logical_cis)]
                logical_hi = [fr + hw for fr, (_, hw) in zip(logical_frs, logical_cis)]
                ax.errorbar(ps, logical_frs,
                            yerr=[[fr - lo for fr, lo in zip(logical_frs, logical_lo)],
                                  [hi - fr for fr, hi in zip(logical_frs, logical_hi)]],
                            marker="s", capsize=4, linestyle="--", label=f"{algo} logical err")

        all_ps = []
        for algo_results in all_results_by_algo.values():
            if code_id in algo_results:
                all_ps.extend(r["p"] for r in algo_results[code_id]["results"])

        ax.set_xlabel("p (error probability)")
        ax.set_ylabel("Rate")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlim(0, max(all_ps) * 1.05)
        ax.set_ylim(0, 1)
        ax.legend(fontsize="small")
        ax.set_title(f"Decoder Comparison\n{code_label}")
        ax.grid(True, which="both", alpha=0.3)

        plot_path = os.path.join(OUTPUT_DIR, f"{code_id}_compare.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot to %s", plot_path)
        plt.show()
        plt.close(fig)


def plot_soft_bitflip_results(all_results):
    """
    Plot soft bit-flip results: failure rate vs p, one curve per alpha value.
    Two sub-curves per alpha: decode failure and logical error.

    Args:
        all_results: dict mapping code_id -> result data from run_experiment_soft_bitflip
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for code_id, data in all_results.items():
        results = data["results"]
        code_label = f"{code_id} (n={data['n']}, m={data['m']}, dv={data['dv']}, dc={data['dc']})"

        alpha_vals = sorted(set(r["alpha"] for r in results))

        # --- Plot 1: failure rate vs p, one curve per alpha ---
        fig, ax = plt.subplots(figsize=(10, 7))
        for alpha_val in alpha_vals:
            curve = sorted([r for r in results if r["alpha"] == alpha_val], key=lambda r: r["p"])
            ps = [r["p"] for r in curve]

            # Total failure rate (decode + logical)
            total_failures = [r["n_decode_failures"] + r["n_logical_errors"] for r in curve]
            total_frs = [(r["n_decode_failures"] + r["n_logical_errors"]) / r["n_samples"]
                         if r["n_samples"] > 0 else 0.0 for r in curve]
            total_cis = [wilson_ci(nf, r["n_samples"]) for nf, r in zip(total_failures, curve)]
            lo = [max(fr - hw, 0.0) for fr, (_, hw) in zip(total_frs, total_cis)]
            hi = [fr + hw for fr, (_, hw) in zip(total_frs, total_cis)]
            ax.errorbar(ps, total_frs,
                        yerr=[[fr - l for fr, l in zip(total_frs, lo)],
                              [h - fr for fr, h in zip(total_frs, hi)]],
                        marker="o", capsize=4, label=f"α={alpha_val:.3f}")

        all_ps = [r["p"] for r in results]
        ax.set_xlabel("p (error probability)")
        ax.set_ylabel("Total failure rate")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlim(0, max(all_ps) * 1.05)
        ax.set_ylim(0, 1)
        ax.legend(fontsize="small")
        ax.set_title(f"Soft Bit-Flip: Total Failure Rate vs p\n{code_label}")
        ax.grid(True, which="both", alpha=0.3)

        plot_path = os.path.join(OUTPUT_DIR, f"{code_id}_soft_bitflip_vs_p.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot to %s", plot_path)
        plt.show()
        plt.close(fig)

        # --- Plot 2: failure rate vs alpha for each p ---
        fig, ax = plt.subplots(figsize=(10, 7))
        p_vals = sorted(set(r["p"] for r in results))
        for p_val in p_vals:
            curve = sorted([r for r in results if r["p"] == p_val], key=lambda r: r["alpha"])
            alphas = [r["alpha"] for r in curve]

            decode_frs = [r["decode_failure_rate"] for r in curve]
            logical_frs = [r["logical_error_rate"] for r in curve]

            ax.plot(alphas, decode_frs, marker="o", label=f"p={p_val:.3f} decode fail")
            ax.plot(alphas, logical_frs, marker="s", linestyle="--", label=f"p={p_val:.3f} logical err")

        ax.set_xlabel("α (syndrome weight)")
        ax.set_ylabel("Rate")
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_ylim(0, 1)
        ax.legend(fontsize="small")
        ax.set_title(f"Soft Bit-Flip: Failure Rates vs α\n{code_label}")
        ax.grid(True, which="both", alpha=0.3)

        plot_path = os.path.join(OUTPUT_DIR, f"{code_id}_soft_bitflip_vs_alpha.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot to %s", plot_path)
        plt.show()
        plt.close(fig)
