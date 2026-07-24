"""Shared Monte Carlo infrastructure for Theorem 3.5 / 3.8 verifiers.

Provides functions to:
  - Generate resampled datasets
  - Compute alpha_hat (BOAB estimator) on each
  - Compute population-level alpha_bd
  - Measure deviation, variance, and concentration
"""
from __future__ import annotations
import numpy as np

from src.data import generate_hard_setting
from src.strategies import get_strategy
from src.discrepancy import median_heuristic
from src.boab import complexity_term


def fixed_representation(X: np.ndarray, repr_dim: int = 8, seed: int = 0) -> np.ndarray:
    """Fixed random projection as representation Phi(X) = tanh(X @ W)."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W = rng.standard_normal((d, repr_dim)) / np.sqrt(d)
    return np.tanh(X @ W)


def compute_profile_criterion(
    Z: np.ndarray, T: np.ndarray, Y: np.ndarray, K: int,
    strategy: str, alpha_grid: np.ndarray, sigma: float, n: int,
    delta: float = 0.05, M: float = 5.0,
) -> np.ndarray:
    """Compute Q_hat(alpha) for each alpha in the grid.

    Q_hat(alpha) = eps_f(theta_hat(alpha)) + alpha*R_hat_S(theta_hat(alpha)) + Comp(alpha;n,delta)
    """
    strat = get_strategy(strategy)
    T_oh = np.eye(K)[T]
    design_base = np.hstack([Z, T_oh])
    d_dim = design_base.shape[1]

    Q_values = np.zeros(len(alpha_grid))
    for i, alpha in enumerate(alpha_grid):
        # Alpha-dependent ridge regression: stronger alpha -> more regularization
        alpha_reg = 0.01 + alpha * 0.1
        try:
            beta = np.linalg.solve(
                design_base.T @ design_base + alpha_reg * np.eye(d_dim),
                design_base.T @ Y,
            )
            Y_hat = design_base @ beta
            eps_f = float(np.mean((Y_hat - Y) ** 2))
        except np.linalg.LinAlgError:
            eps_f = 1e6

        imb = strat.imbalance(Z, T, K, sigma=sigma)
        comp = complexity_term(alpha, n, delta, M)
        Q_values[i] = eps_f + alpha * imb + comp

    return Q_values


def compute_alpha_hat(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray, K: int,
    strategy: str, alpha_grid: np.ndarray, sigma: float, n: int,
    delta: float = 0.05, M: float = 5.0, repr_seed: int = 0,
) -> float:
    """Compute alpha_hat = argmin Q_hat(alpha)."""
    Z = fixed_representation(X, repr_dim=8, seed=repr_seed)
    Q = compute_profile_criterion(Z, T, Y, K, strategy, alpha_grid, sigma, n, delta, M)
    return float(alpha_grid[int(np.argmin(Q))])


def compute_curvature(
    Q_values: np.ndarray, alpha_grid: np.ndarray,
) -> float:
    """Estimate kappa_S = min second derivative of Q(alpha) via finite differences."""
    if len(Q_values) < 3:
        return 1.0
    da = alpha_grid[1] - alpha_grid[0]
    second_deriv = (Q_values[2:] - 2 * Q_values[1:-1] + Q_values[:-2]) / (da ** 2)
    kappa = float(np.maximum(second_deriv.min(), 1e-8))
    return kappa


def monte_carlo_alpha_hat(
    K: int, n: int, strategy: str, alpha_grid: np.ndarray,
    n_resamples: int = 200, d: int = 20, seed_base: int = 1000,
    population_n: int = 10000,
) -> dict:
    """Monte Carlo study: compute alpha_hat across many resamples.

    Returns dict with:
      - alpha_hats: array of alpha_hat values
      - alpha_bd: population minimizer
      - deviations: |alpha_hat - alpha_bd|
      - kappa_S: curvature estimate
      - r_S: concentration of imbalance
      - imbalance_hats: array of imbalance values
    """
    # Population-level: compute alpha_bd using a very large sample
    pop_data = generate_hard_setting(N=population_n, K=K, d=d, seed=99999)
    pop_sigma = median_heuristic(pop_data["X"])
    Z_pop = fixed_representation(pop_data["X"], seed=0)
    Q_pop = compute_profile_criterion(
        Z_pop, pop_data["T"], pop_data["Y"], K, strategy,
        alpha_grid, pop_sigma, population_n,
    )
    alpha_bd = float(alpha_grid[int(np.argmin(Q_pop))])
    kappa_S = compute_curvature(Q_pop, alpha_grid)

    # Resamples
    alpha_hats = np.zeros(n_resamples)
    imbalance_hats = np.zeros(n_resamples)
    strat = get_strategy(strategy)

    for r in range(n_resamples):
        data = generate_hard_setting(N=n, K=K, d=d, seed=seed_base + r)
        Z = fixed_representation(data["X"], seed=0)
        sigma = median_heuristic(data["X"])
        Q = compute_profile_criterion(Z, data["T"], data["Y"], K, strategy, alpha_grid, sigma, n)
        alpha_hats[r] = alpha_grid[int(np.argmin(Q))]
        imbalance_hats[r] = strat.imbalance(Z, data["T"], K, sigma=sigma)
        if (r + 1) % 50 == 0:
            from verifiers.common import log
            log(f"      resample {r+1}/{n_resamples} (K={K}, {strategy})")

    deviations = np.abs(alpha_hats - alpha_bd)
    r_S = float(np.std(imbalance_hats))  # concentration of imbalance

    return {
        "K": K, "n": n, "strategy": strategy,
        "alpha_hats": alpha_hats,
        "alpha_bd": alpha_bd,
        "deviations": deviations,
        "kappa_S": kappa_S,
        "r_S": r_S,
        "imbalance_hats": imbalance_hats,
        "Q_pop": Q_pop,
    }
