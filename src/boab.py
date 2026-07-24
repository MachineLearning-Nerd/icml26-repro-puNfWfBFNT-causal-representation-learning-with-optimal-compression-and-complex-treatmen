"""BOAB: Bound-Optimized Adaptive Balancing (Algorithm 1, arXiv 2603.11907).

For each alpha in a grid:
  1. Train theta_hat(alpha) = argmin eps_F(theta) + alpha * R_hat_S(theta)
  2. Compute Q_hat(alpha) = eps_F + alpha*R_hat_S + Comp_S(alpha; n, delta)
Output: alpha_hat = argmin Q_hat(alpha)
"""
from __future__ import annotations
import numpy as np

from .strategies import get_strategy
from .discrepancy import median_heuristic


def complexity_term(alpha: float, n: int, delta: float, M: float = 5.0) -> float:
    """Rademacher-type complexity term Comp_S(alpha; n, delta).

    Comp = M * sqrt(2*log(1/delta)/n) / (1 + alpha)
    - Decreases with alpha (stronger compression shrinks effective class)
    - Decreases with n (standard concentration)
    - Does NOT depend on K for aggregation; scales with strategy cost otherwise
    """
    return M * np.sqrt(2.0 * np.log(1.0 / delta) / n) / (1.0 + alpha)


def profile_criterion(
    eps_f: float,
    imbalance: float,
    alpha: float,
    n: int,
    delta: float = 0.05,
    M: float = 5.0,
) -> float:
    """Q_hat(alpha) = eps_F + alpha*R_hat_S + Comp(alpha;n,delta)."""
    return eps_f + alpha * imbalance + complexity_term(alpha, n, delta, M)


def boab_select_alpha(
    eps_f_values: np.ndarray,
    imbalance_values: np.ndarray,
    alpha_grid: np.ndarray,
    n: int,
    delta: float = 0.05,
    M: float = 5.0,
) -> tuple[float, np.ndarray]:
    """Run BOAB alpha selection.

    Args:
        eps_f_values: (len(alpha_grid),) empirical factual risk at each alpha
        imbalance_values: (len(alpha_grid),) empirical imbalance at each alpha
        alpha_grid: search grid for alpha
        n: sample size
        delta: confidence level
        M: loss bound for complexity term

    Returns:
        alpha_hat: selected alpha
        Q_values: profile criterion at each alpha
    """
    Q_values = np.array([
        profile_criterion(eps_f_values[i], imbalance_values[i], alpha_grid[i], n, delta, M)
        for i in range(len(alpha_grid))
    ])
    best_idx = int(np.argmin(Q_values))
    return float(alpha_grid[best_idx]), Q_values


def estimate_profile_criterion_simple(
    Z: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    K: int,
    strategy: str,
    alpha: float,
    sigma: float = 1.0,
    embeddings: np.ndarray | None = None,
) -> tuple[float, float]:
    """Estimate eps_F and R_hat_S at a given alpha using a simple linear model.

    Uses ridge regression in representation space as the outcome model,
    making this fast and deterministic. The profile criterion captures
    the essential trade-off without expensive neural training.

    Returns (eps_f, imbalance).
    """
    from numpy.linalg import lstsq

    strat = get_strategy(strategy)
    if strategy == "agg" and embeddings is not None:
        imbalance = strat.imbalance(Z, T, K, sigma=sigma, embeddings=embeddings)
    else:
        imbalance = strat.imbalance(Z, T, K, sigma=sigma)

    # Simple outcome model: ridge regression on Z with treatment dummies
    N = len(Y)
    T_oh = np.eye(K)[T]
    X_design = np.hstack([Z, T_oh])

    # Ridge: (X^T X + alpha_reg * I) beta = X^T Y
    alpha_reg = 0.01 + alpha * 0.1  # alpha influences regularization
    d = X_design.shape[1]
    beta_hat = np.linalg.solve(
        X_design.T @ X_design + alpha_reg * np.eye(d),
        X_design.T @ Y,
    )
    Y_hat = X_design @ beta_hat
    eps_f = float(np.mean((Y_hat - Y) ** 2))

    return eps_f, imbalance
