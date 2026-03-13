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
