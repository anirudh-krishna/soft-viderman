import logging
import os
import sys
import glob
from read_ccodes import read_ccode
from decoder import (VidermanDecoder, BitFlipDecoder, SoftBitFlipDecoder,
                     MinSumBPDecoder, SumProductBPDecoder, BPFlipDecoder)
from experiment import (run_experiment, run_experiment_bitflip,
                       run_experiment_soft_bitflip, run_experiment_compare)
from optimize_h import run_optimize_h
from plotting import (plot_results, plot_optimal_h, plot_bitflip_results,
                      plot_soft_bitflip_results, plot_compare_results)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_codes(nm_list):
    """
    Scan ccode/swac_* files and return all Classical_code instances matching any (n, m) pair.

    Args:
        nm_list: list of (n, m) tuples

    Returns:
        list of Classical_code instances
    """
    ccode_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ccode")
    pattern = os.path.join(ccode_dir, "swac_*.code")
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        logger.warning("No swac_*.code files found in %s", ccode_dir)
        return []
    codes = []
    for n, m in nm_list:
        found = read_ccode(file_list, [n], [m], [], [], [])
        logger.info("load_codes(n=%d, m=%d): found %d matching code(s) from %d files",
                    n, m, len(found), len(file_list))
        codes.extend(found)
    return codes


if __name__ == "__main__":
    # --- Configuration ---
    # mode: "sweep", "optimize_h", "bitflip", "soft_bitflip", "minsum_bp",
    #       or "compare" (set compare_decoders below)
    mode = "compare"
    #compare_decoders = ["bitflip", "sumproduct_bp_50", "sp50_flip"]
    compare_decoders = ["bitflip", "sumproduct_bp_25", "sp25_flip", "sumproduct_bp_100"]

    nm_list = [(120, 100),(240,200)]
    p_list = [0.04, 0.05, 0.06, 0.07]
    ci_target = 0.01

    # Load codes
    codes = load_codes(nm_list)
    if not codes:
        logger.error("No codes found for nm_list=%s.", nm_list)
        sys.exit(1)

    DECODER_MAP = {
        "bitflip": lambda code: BitFlipDecoder(code),
        "minsum_bp_50": lambda code: MinSumBPDecoder(code, max_iters=50),
        "minsum_bp_100": lambda code: MinSumBPDecoder(code, max_iters=100),
        "minsum_bp_200": lambda code: MinSumBPDecoder(code, max_iters=200),
        "sumproduct_bp_25": lambda code: SumProductBPDecoder(code, max_iters=25),
        "sumproduct_bp_50": lambda code: SumProductBPDecoder(code, max_iters=50),
        "sumproduct_bp_100": lambda code: SumProductBPDecoder(code, max_iters=100),
        "sumproduct_bp_200": lambda code: SumProductBPDecoder(code, max_iters=200),
        "sp25_flip": lambda code: BPFlipDecoder(code, bp_iters=25, bp_class=SumProductBPDecoder),
        "sp50_flip": lambda code: BPFlipDecoder(code, bp_iters=50, bp_class=SumProductBPDecoder),
        "sp200_flip": lambda code: BPFlipDecoder(code, bp_iters=200, bp_class=SumProductBPDecoder),
    }

    if mode == "compare":
        # All decoders see the same error samples per code
        all_results_by_algo = {}
        for code in codes:
            decoder_list = [DECODER_MAP[name](code) for name in compare_decoders]
            code_results = run_experiment_compare(decoder_list, code, p_list,
                                                  ci_target=ci_target)
            for label, result in code_results.items():
                if label not in all_results_by_algo:
                    all_results_by_algo[label] = {}
                all_results_by_algo[label][code.id] = result

        plot_compare_results(all_results_by_algo)

    elif mode == "bitflip":
        # Bit-flip decoder: no h parameter
        all_results = {}
        for code in codes:
            decoder = BitFlipDecoder(code)
            result = run_experiment_bitflip(decoder, code, p_list, ci_target=ci_target)
            all_results[code.id] = result

        plot_bitflip_results(all_results)

    elif mode == "soft_bitflip":
        # Soft bit-flip decoder: coarse sweep over alpha values
        import numpy as np
        alpha_list = list(np.linspace(0.5, 3.0, 11))

        all_results = {}
        for code in codes:
            decoder = SoftBitFlipDecoder(code)
            result = run_experiment_soft_bitflip(decoder, code, p_list, alpha_list,
                                                ci_target=0.1, min_samples=100,
                                                max_samples=5000)
            all_results[code.id] = result

        plot_soft_bitflip_results(all_results)

    elif mode == "minsum_bp":
        # Min-sum BP decoder
        all_results = {}
        for code in codes:
            decoder = MinSumBPDecoder(code, max_iters = 200)
            result = run_experiment_bitflip(decoder, code, p_list, ci_target=ci_target)
            all_results[code.id] = result

        plot_bitflip_results(all_results)

    elif mode == "sweep":
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
