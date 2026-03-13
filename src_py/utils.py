import numpy as np


def compute_synd(code, err):
    """
    Compute the syndrome H @ err mod 2 using the sparse check_nbhd representation.

    Args:
        code: Classical_code instance
        err: np.array of length code.n, binary (0/1)

    Returns:
        np.array of length code.m, binary (0/1)
    """
    syndrome = np.zeros(code.m, dtype=int)
    for c in range(code.m):
        syndrome[c] = np.sum(err[code.check_nbhd[c]]) % 2
    return syndrome


def wilson_ci(n_failures, n_total, z=1.96):
    """
    Wilson score interval for a binomial proportion.

    Args:
        n_failures: number of failures
        n_total: total number of trials
        z: z-score for confidence level (1.96 for 95%)

    Returns:
        (p_hat, half_width) where p_hat is the raw proportion and
        half_width is the Wilson CI half-width.
    """
    if n_total == 0:
        return 0.0, 1.0
    p_hat = n_failures / n_total
    denom = 1.0 + z * z / n_total
    half_width = (z * np.sqrt(p_hat * (1 - p_hat) / n_total + z * z / (4.0 * n_total * n_total))) / denom
    return p_hat, half_width
