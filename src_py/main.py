import logging
import os
import sys
import glob
from read_ccodes import read_ccode
from decoder import VidermanDecoder
from experiment import run_experiment
from optimize_h import run_optimize_h
from plotting import plot_results, plot_optimal_h

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_codes(n, m):
    """
    Scan ccode/swac_* files and return all Classical_code instances matching the given n and m.

    Args:
        n: number of variable nodes (bits)
        m: number of check nodes

    Returns:
        list of Classical_code instances with code.n == n and code.m == m
    """
    ccode_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ccode")
    pattern = os.path.join(ccode_dir, "swac_*.code")
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        logger.warning("No swac_*.code files found in %s", ccode_dir)
        return []
    codes = read_ccode(file_list, [n], [m], [], [], [])
    logger.info("load_codes(n=%d, m=%d): found %d matching code(s) from %d files",
                n, m, len(codes), len(file_list))
    return codes


if __name__ == "__main__":
    # --- Configuration ---
    mode = "optimize_h"  # "sweep" or "optimize_h"

    n = 120
    m = 100
    p_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    ci_target = 0.01

    # Load codes
    codes = load_codes(n, m)
    if not codes:
        logger.error("No codes found with n=%d, m=%d.", n, m)
        sys.exit(1)

    if mode == "sweep":
        # Sweep over all h values
        dv = codes[0].dv
        h_list = list(range(dv + 1))

        all_results = {}
        for code in codes:
            decoder = VidermanDecoder(code)
            result = run_experiment(decoder, code, p_list, h_list, ci_target=ci_target)
            all_results[code.id] = result

        plot_results(all_results, p_list)

    elif mode == "optimize_h":
        # Find optimal h for each p
        all_optimal = {}
        for code in codes:
            decoder = VidermanDecoder(code)
            result = run_optimize_h(decoder, code, p_list, ci_target=ci_target)
            all_optimal[code.id] = result

        plot_optimal_h(all_optimal)
