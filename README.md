# Viderman's Algorithm for LDPC Codes

Numerical simulation of Viderman's decoding algorithm on classical LDPC codes.

## Classical Codes (`ccode/swac_*`)

All `swac_*` code files, grouped by parameters. Each row may correspond to multiple code instances (different Tanner graphs with the same parameters).

| n | m | dv | dc | # codes |
|-----|-----|----|----|---------|
| 30  | 20  | 2  | 3  | 1       |
| 30  | 30  | 3  | 3  | 1       |
| 40  | 30  | 3  | 4  | 1       |
| 40  | 40  | 4  | 4  | 1       |
| 50  | 25  | 5  | 10 | 1       |
| 50  | 40  | 4  | 5  | 2       |
| 60  | 30  | 3  | 6  | 1       |
| 60  | 40  | 2  | 3  | 1       |
| 60  | 50  | 5  | 6  | 1       |
| 60  | 60  | 3  | 3  | 1       |
| 70  | 60  | 6  | 7  | 1       |
| 80  | 40  | 4  | 8  | 1       |
| 80  | 60  | 3  | 4  | 1       |
| 80  | 70  | 7  | 8  | 1       |
| 80  | 80  | 4  | 4  | 1       |
| 90  | 60  | 2  | 3  | 1       |
| 90  | 90  | 3  | 3  | 1       |
| 100 | 50  | 5  | 10 | 1       |
| 100 | 80  | 4  | 5  | 2       |
| 120 | 60  | 3  | 6  | 1       |
| 120 | 60  | 6  | 12 | 1       |
| 120 | 80  | 2  | 3  | 1       |
| 120 | 90  | 3  | 4  | 1       |
| 120 | 100 | 5  | 6  | 1       |
| 120 | 120 | 3  | 3  | 1       |
| 120 | 120 | 4  | 4  | 1       |
| 140 | 120 | 6  | 7  | 1       |
| 150 | 75  | 5  | 10 | 1       |
| 150 | 120 | 4  | 5  | 2       |
| 160 | 80  | 4  | 8  | 1       |
| 160 | 120 | 3  | 4  | 1       |
| 160 | 140 | 7  | 8  | 1       |
| 160 | 160 | 4  | 4  | 1       |
| 180 | 90  | 3  | 6  | 1       |
| 180 | 150 | 5  | 6  | 1       |
| 200 | 100 | 5  | 10 | 1       |
| 200 | 160 | 4  | 5  | 2       |
| 210 | 180 | 6  | 7  | 1       |
| 240 | 120 | 3  | 6  | 1       |
| 240 | 120 | 4  | 8  | 1       |
| 240 | 120 | 6  | 12 | 1       |
| 240 | 200 | 5  | 6  | 1       |
| 240 | 210 | 7  | 8  | 1       |
| 240 | 240 | 3  | 3  | 1       |
| 250 | 125 | 5  | 10 | 1       |
| 250 | 200 | 4  | 5  | 2       |
| 280 | 240 | 6  | 7  | 1       |
| 300 | 150 | 5  | 10 | 1       |
| 300 | 240 | 4  | 5  | 2       |
| 300 | 250 | 5  | 6  | 1       |
| 320 | 160 | 4  | 8  | 1       |
| 320 | 240 | 3  | 4  | 1       |
| 320 | 280 | 7  | 8  | 1       |
| 360 | 180 | 6  | 12 | 1       |
| 360 | 300 | 5  | 6  | 1       |
| 480 | 240 | 3  | 6  | 1       |

**Total: 62 code files across 56 parameter sets.**

## Running Experiments

Following scripts are in `src_py/` and should be run from that directory.

### `main.py`

Entry point for experiments. Set `mode`, `n`, `m`, `p_list`, and `ci_target` in the configuration block.

**`mode = "sweep"`** — Sweep over all threshold values

Runs the decoder for every `h` in `{0, 1, ..., dv}` and every `p` in `p_list`. Uses adaptive sampling with relative 95% Wilson confidence intervals OR until some max number of samples is collected (determined by variable max_samples, arg to run_experiment, whose default is 100000).
Produces a failure rate vs. p plot with one curve per h value.

- Results saved to `outputs/{code.id}.json`
- Plot saved to `outputs/failure_rate.png`

```python
mode = "sweep"
n = 120
m = 100
p_list = [0.01, 0.02, 0.03, 0.04, 0.05]
ci_target = 0.01
```

**`mode = "optimize_h"`** — Find optimal threshold

For each `p` in `p_list`, evaluates all `h` in `{0, 1, ..., dv}` and selects the one that minimizes failure rate. Produces an optimal h vs. p plot.

- Results saved to `outputs/{code.id}_optimal_h.json`
- Plot saved to `outputs/optimal_h.png`

```python
mode = "optimize_h"
n = 120
m = 100
p_list = [0.01, 0.02, 0.03, 0.04, 0.05]
ci_target = 0.01
```

### `test_explicit.py`

Generates a single error sample and walks through every step of the decoding algorithm with full diagnostic output:

1. **Error generation** — shows the error support
2. **Syndrome computation** — shows unsatisfied checks
3. **`find` step** — prints each variable node added to L, whether it is an actual error bit or a clean bit, and how R grows
4. **Early fail check** — verifies all unsatisfied checks are contained in R
5. **Erasure recovery** — shows each peeling step and the resolved bit value
6. **Syndrome verification** — confirms the recovered error matches the syndrome

Configure by editing the `__main__` block:

```python
walk_through_decode(codes[0], p=0.065, h=4, seed=42)
```
