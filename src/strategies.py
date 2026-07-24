"""Balancing strategies: pairwise, one-vs-all (OVA), and treatment aggregation.

Each strategy defines a population imbalance functional R_S(Phi) and its
empirical estimator R_hat_S(Phi), per Section 2 of arXiv 2603.11907.
"""
from __future__ import annotations
import time
import numpy as np
from .discrepancy import (
    mmd2_ipm,
    mmd2_u_statistic,
    hsic_v_statistic,
    median_heuristic,
    count_pairwise_ops,
    count_ova_ops,
    count_agg_ops,
)


class BalancingStrategy:
    """Base class for balancing strategies."""

    name: str = "base"
    theoretical_cost: str = ""

    def imbalance(self, Z: np.ndarray, T: np.ndarray, K: int, **kwargs) -> float:
        raise NotImplementedError

    def timed_imbalance(self, Z: np.ndarray, T: np.ndarray, K: int, **kwargs) -> tuple[float, float]:
        """Compute imbalance and return (value, wall-clock seconds)."""
        t0 = time.perf_counter()
        val = self.imbalance(Z, T, K, **kwargs)
        elapsed = time.perf_counter() - t0
        return val, elapsed

    def n_ops(self, K: int) -> int:
        raise NotImplementedError


class PairwiseStrategy(BalancingStrategy):
    """Pairwise balancing: R_pair = sum_{j<k} IPM(P_j, P_k). Cost O(K^2)."""

    name = "pair"
    theoretical_cost = "O(K^2)"

    def imbalance(self, Z: np.ndarray, T: np.ndarray, K: int, sigma: float = 1.0, **kwargs) -> float:
        # Vectorized: compute full kernel matrix once, extract pairwise MMDs
        from .discrepancy import rbf_kernel
        N = len(Z)
        K_full = rbf_kernel(Z, Z, sigma)
        T_oh = np.eye(K)[T]
        n_k = T_oh.sum(axis=0)
        # Cross-group kernel means: A[j,k] = sum K(x_i,x_l) for T_i=j, T_l=k
        cross = T_oh.T @ K_full @ T_oh
        norm = np.outer(n_k, n_k)
        norm = np.maximum(norm, 1.0)
        A = cross / norm
        # MMD^2(j,k) = A[j,j] + A[k,k] - 2*A[j,k]
        diag = np.diag(A)
        mmd2_mat = diag[:, None] + diag[None, :] - 2.0 * A
        mmd2_mat = np.maximum(mmd2_mat, 0.0)
        mmd_mat = np.sqrt(mmd2_mat)
        # Sum upper triangle
        mask = np.triu(np.ones((K, K)), k=1).astype(bool)
        return float((mmd_mat * mask).sum())

    def n_ops(self, K: int) -> int:
        return count_pairwise_ops(K)


class OVAStrategy(BalancingStrategy):
    """One-vs-All balancing: R_ova = sum_k IPM(P_k, P_{-k}). Cost O(K)."""

    name = "ova"
    theoretical_cost = "O(K)"

    def imbalance(self, Z: np.ndarray, T: np.ndarray, K: int, sigma: float = 1.0, **kwargs) -> float:
        # Vectorized using full kernel matrix
        from .discrepancy import rbf_kernel
        N = len(Z)
        K_full = rbf_kernel(Z, Z, sigma)
        T_oh = np.eye(K)[T]
        n_k = T_oh.sum(axis=0)
        n_neg = N - n_k
        T_neg = 1.0 - T_oh

        cross = T_oh.T @ K_full @ T_oh
        cross_neg = T_oh.T @ K_full @ T_neg
        cross_neg_neg = T_neg.T @ K_full @ T_neg

        diag_kk = np.diag(cross) / np.maximum(n_k ** 2, 1.0)
        diag_nn = np.diag(cross_neg_neg) / np.maximum(n_neg ** 2, 1.0)
        cross_kn = np.diag(cross_neg) / np.maximum(n_k * n_neg, 1.0)

        mmd2_vec = np.maximum(diag_kk + diag_nn - 2.0 * cross_kn, 0.0)
        return float(np.sqrt(mmd2_vec).sum())

    def n_ops(self, K: int) -> int:
        return count_ova_ops(K)


class AggregationStrategy(BalancingStrategy):
    """Treatment aggregation: R_agg = HSIC(Phi(X), E_T). Cost O(1) w.r.t. K."""

    name = "agg"
    theoretical_cost = "O(1)"

    def imbalance(
        self,
        Z: np.ndarray,
        T: np.ndarray,
        K: int,
        sigma: float = 1.0,
        embeddings: np.ndarray | None = None,
        **kwargs,
    ) -> float:
        if embeddings is None:
            # Use one-hot encoding as default embedding
            embeddings = np.eye(K)[T]
        sigma_e = 1.0
        return hsic_v_statistic(Z, embeddings, sigma_x=sigma, sigma_e=sigma_e)

    def n_ops(self, K: int) -> int:
        return count_agg_ops(K)


def get_strategy(name: str) -> BalancingStrategy:
    strategies = {
        "pair": PairwiseStrategy,
        "ova": OVAStrategy,
        "agg": AggregationStrategy,
    }
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Choose from {list(strategies.keys())}")
    return strategies[name]()


def compute_all_imbalances(
    Z: np.ndarray, T: np.ndarray, K: int, sigma: float = 1.0,
    embeddings: np.ndarray | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute imbalance for all three strategies with timing.

    Returns {strategy_name: (imbalance_value, elapsed_seconds)}.
    """
    results = {}
    for name in ["pair", "ova", "agg"]:
        strat = get_strategy(name)
        kw = {}
        if name == "agg" and embeddings is not None:
            kw["embeddings"] = embeddings
        val, elapsed = strat.timed_imbalance(Z, T, K, sigma=sigma, **kw)
        results[name] = (val, elapsed)
    return results
