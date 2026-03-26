# Error Correcting Codes Simulation Project

## Overview
Numerical simulation of classical LDPC codes using Viderman's algorithm.

## Core Principles
- **Performance:** Utilize vectorized operations (NumPy).
- **Correctness:** Python implementation must be fully validated before any C++ migration

## Project Structure
- `ccode/` — classical error correcting codes
- `src_py/` — Python source code (development & testing)
  - `read_ccodes.py` — loads classical codes from `ccode/`
  - `decoder.py` — `Decoder` ABC + concrete subclasses (e.g. `VidermanDecoder`)
  - `utils.py` — shared helpers (`compute_synd`, `wilson_ci`)
  - `experiment.py` — experiment orchestration and adaptive sampling
  - `optimize_h.py` — finds optimal h per (code, p) by sweeping h in {0..dv}
  - `plotting.py` — failure rate plots + optimal h vs p plots
  - `main.py` — code loading (`load_codes`), entry point with `mode`: `"sweep"`, `"optimize_h"`, `"bitflip"`, or `"soft_bitflip"`
  - `test_explicit.py` — single-sample decode walkthrough for verification
- `src_cpp/` — C++ source code (Used only after Python version is validated)
- `outputs/` — all results, plots, and data exports

## Stack
- **Language** Python (Numpy/ Scipy) and C++ (BLAS/LAPACK)
- **Visualization** Matplotlib for plots

## Conventions
- Use verbose, readable code with `logging` module
- Output data saved as per-code JSON: `outputs/{code.id}.json` (sweep), `outputs/{code.id}_optimal_h.json` (optimization), `outputs/{code.id}_{algorithm_label}.json` (bitflip/soft_bitflip)
- Decoder threshold `h` is an integer in `{0, 1, ..., dv}` — passed directly (not via eps). Only `dv+1` distinct behaviors exist. (Viderman only)
- New decoders: subclass `Decoder` ABC, set `label` class attribute, implement `decode(...)`. Shared methods (`erasure_recovery`, `verify_syndrome`) live on the base class.
- `BitFlipDecoder`: iterative bit-flipping. `decode(syndrome)` returns `(e_deduced, converged)`. Flips the variable with the most unsatisfied neighbors each iteration (threshold: `> dv/2`). No `h` parameter. Experiment tracks decode failures and logical errors separately.
- `SoftBitFlipDecoder`: soft bit-flipping with channel LLR prior. `E_v = -b*(1-2*x_v) + α*Σ(2*s_c-1)` where `b=log((1-p)/p)`. `decode(syndrome, p, alpha)` returns `(e_deduced, converged)`. Experiment sweeps over `alpha_list`.
- `MinSumBPDecoder`: min-sum belief propagation (syndrome-based). `decode(syndrome, p, max_iters=50)` returns `(e_deduced, converged)`. Precomputes edge index mappings in `__init__`. Uses two-smallest trick for efficient min-excluding-one.
- Code loading: `load_codes(nm_list)` scans `ccode/swac_*.code` filtered by list of `(n, m)` tuples

## Workflow
- Always develop and validate in `src_py/` first.
- NumPy vectorized operations preferred over loops
- Always save results and plots to `outputs/`

## Meta
After completing any significant task, suggest updates to CLAUDE.md or 
relevant docs/ files if new conventions, structures, or workflows were established.