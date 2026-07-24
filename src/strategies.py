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
        total = 0.0
        for j in range(K):
            Zj = Z[T == j]
            if len(Zj) < 2:
                continue
            for k in range(j + 1, K):
                Zk = Z[T == k]
                if len(Zk) < 2:
                    continue
                total += mmd2_ipm(Zj, Zk, sigma)
        return total

    def n_ops(self, K: int) -> int:
        return count_pairwise_ops(K)


class OVAStrategy(BalancingStrategy):
    """One-vs-All balancing: R_ova = sum_k IPM(P_k, P_{-k}). Cost O(K)."""

    name = "ova"
    theoretical_cost = "O(K)"

    def imbalance(self, Z: np.ndarray, T: np.ndarray, K: int, sigma: float = 1.0, **kwargs) -> float:
        total = 0.0
        for k in range(K):
            Zk = Z[T == k]
            Znk = Z[T != k]
            if len(Zk) < 2 or len(Znk) < 2:
                continue
            total += mmd2_ipm(Zk, Znk, sigma)
        return total

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
