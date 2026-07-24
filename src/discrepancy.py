"""Discrepancy measures: MMD (U and V statistics) and HSIC.

Following Appendix B.1 of arXiv 2603.11907 and Gretton et al. (2005).
"""
from __future__ import annotations
import numpy as np


def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """RBF (Gaussian) kernel matrix between X and Y."""
    sq_dist = (
        np.sum(X ** 2, axis=1)[:, None]
        + np.sum(Y ** 2, axis=1)[None, :]
        - 2.0 * X @ Y.T
    )
    sq_dist = np.maximum(sq_dist, 0.0)
    return np.exp(-sq_dist / (2.0 * sigma ** 2))


def median_heuristic(X: np.ndarray) -> float:
    """Median pairwise distance for bandwidth selection."""
    n = min(X.shape[0], 500)
    idx = np.random.default_rng(0).choice(X.shape[0], n, replace=False)
    Xs = X[idx]
    sq_dist = (
        np.sum(Xs ** 2, axis=1)[:, None]
        + np.sum(Xs ** 2, axis=1)[None, :]
        - 2.0 * Xs @ Xs.T
    )
    sq_dist = np.maximum(sq_dist, 0.0)
    off_diag = sq_dist[np.triu_indices(n, k=1)]
    return float(np.sqrt(np.median(off_diag[off_diag > 0]))) if np.any(off_diag > 0) else 1.0


def mmd2_u_statistic(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """Unbiased MMD^2 U-statistic (Gretton et al. 2012, Eq. 3).

    MMD^2 = (1/(m(m-1))) sum_{i!=j} k(x_i,x_j) + (1/(n(n-1))) sum_{i!=j} k(y_i,y_j)
            - (2/(mn)) sum_{i,j} k(x_i,y_j)
    """
    m, n = len(X), len(Y)
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)

    # Zero out diagonals for unbiased estimate
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term1 = Kxx.sum() / (m * (m - 1)) if m > 1 else 0.0
    term2 = Kyy.sum() / (n * (n - 1)) if n > 1 else 0.0
    term3 = 2.0 * Kxy.sum() / (m * n)

    return float(max(term1 + term2 - term3, 0.0))


def mmd2_v_statistic(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """Biased MMD^2 V-statistic (Gretton et al. 2012, Eq. 5)."""
    m, n = len(X), len(Y)
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)

    term1 = Kxx.sum() / (m ** 2)
    term2 = Kyy.sum() / (n ** 2)
    term3 = 2.0 * Kxy.sum() / (m * n)

    return float(max(term1 + term2 - term3, 0.0))


def mmd2_ipm(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """MMD as an IPM: sqrt(max(MMD^2, 0)). Uses U-statistic."""
    return float(np.sqrt(mmd2_u_statistic(X, Y, sigma)))


def hsic_v_statistic(
    X: np.ndarray,
    E: np.ndarray,
    sigma_x: float = 1.0,
    sigma_e: float = 1.0,
) -> float:
    """HSIC V-statistic estimator (Gretton et al. 2005, Eq. 5).

    HSIC = (1/(n-1)^2) tr(K_x H L H)
    where H = I - (1/n)11^T is the centering matrix,
    K_x = kernel matrix of X, L = kernel matrix of E.
    """
    n = len(X)
    Kx = rbf_kernel(X, X, sigma_x)
    Le = rbf_kernel(E, E, sigma_e)

    # Centered: H K H L or equivalently tr(KHLH)
    # HSIC^b = (1/(n-1)^2) tr(KHLH) where H = I - 11^T/n
    # Equivalently: (1/n^2) sum_{i,j} (Kx_ij - rowmean - colmean + grandmean)(Le_ij - ...)
    Kx_centered = Kx - Kx.mean(axis=0, keepdims=True) - Kx.mean(axis=1, keepdims=True) + Kx.mean()
    Le_centered = Le - Le.mean(axis=0, keepdims=True) - Le.mean(axis=1, keepdims=True) + Le.mean()

    hsic = (Kx_centered * Le_centered).sum() / ((n - 1) ** 2)
    return float(max(hsic, 0.0))


def count_pairwise_ops(K: int) -> int:
    """Number of pairwise discrepancy terms: C(K,2)."""
    return K * (K - 1) // 2


def count_ova_ops(K: int) -> int:
    """Number of one-vs-all discrepancy terms: K."""
    return K


def count_agg_ops(K: int) -> int:
    """Number of aggregation (HSIC) terms: 1 (independent of K)."""
    return 1
