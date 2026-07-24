"""Optimized Monte Carlo infrastructure.

Separates fast direct measurements (Var(R_hat_S)) from slow profile computations.
- Var(R_hat_S) is measured directly across resamples (one imbalance computation each)
- kappa_S is measured from population profile criterion (once per K)
- alpha_hat is computed for fewer resamples (for direct verification)
"""
from __future__ import annotations
import numpy as np

from src.data import generate_hard_setting
from src.strategies import get_strategy
from src.discrepancy import median_heuristic, rbf_kernel


def fixed_projection(d: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((d, 8)) / np.sqrt(d)


def fixed_representation(X: np.ndarray, repr_dim: int = 8, seed: int = 0) -> np.ndarray:
    """Fixed random projection as representation Phi(X) = tanh(X @ W)."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W = rng.standard_normal((d, repr_dim)) / np.sqrt(d)
    return np.tanh(X @ W)


def get_fixed_embeddings(K: int, d_e: int = 8, seed: int = 12345) -> np.ndarray:
    """Fixed random treatment embeddings (K, d_e) — K-independent dimensionality."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((K, d_e))


def measure_R_variance(
    K: int, n: int, strategy: str, n_resamples: int = 200,
    d: int = 20, seed_base: int = 1000,
    fixed_sigma: float | None = None,
) -> dict:
    """Directly measure variance and std of R_hat_S across resamples."""
    strat = get_strategy(strategy)
    W = fixed_projection(d, seed=0)
    emb = get_fixed_embeddings(K)  # Fixed random embeddings for agg
    if fixed_sigma is None:
        ref_data = generate_hard_setting(N=1000, K=K, d=d, seed=99998)
        Xt_ref = np.tanh(ref_data["X"] @ W)
        fixed_sigma = median_heuristic(Xt_ref)
    sigma = fixed_sigma

    R_hats = np.zeros(n_resamples)
    for r in range(n_resamples):
        data = generate_hard_setting(N=n, K=K, d=d, seed=seed_base + r)
        Xt = np.tanh(data["X"] @ W)
        if strategy == "agg":
            E = emb[data["T"]]  # Map treatments to embeddings
            R_hats[r] = strat.imbalance(Xt, data["T"], K, sigma=sigma, embeddings=E)
        else:
            R_hats[r] = strat.imbalance(Xt, data["T"], K, sigma=sigma)
        if (r + 1) % 100 == 0:
            from verifiers.common import log
            log(f"      R measurement {r+1}/{n_resamples} (K={K}, {strategy})")

    return {
        "K": K, "n": n, "strategy": strategy,
        "R_hats": R_hats,
        "R_mean": float(np.mean(R_hats)),
        "R_std": float(np.std(R_hats)),
        "R_var": float(np.var(R_hats, ddof=1)),
        "sigma": float(sigma),
    }


def compute_profile_criterion(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    K: int, strategy: str, sigma: float,
    alpha_grid: np.ndarray, n: int,
    W: np.ndarray, n_lambdas: int = 10,
    emb: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Compute profile criterion and return (alpha_hat, R_star_at_alpha, Q_values).

    Uses lambda-parameterized representation: Z_lambda = lambda * tanh(X @ W).
    For each alpha, finds optimal lambda minimizing eps_F(lambda) + alpha*R_S(lambda).
    """
    Xt = np.tanh(X @ W)
    lambdas = np.linspace(0.1, 2.0, n_lambdas)

    eps_f_arr = np.zeros(n_lambdas)
    R_arr = np.zeros(n_lambdas)
    strat = get_strategy(strategy)

    # Precompute kernel matrix base (for sigma scaling)
    for i, lam in enumerate(lambdas):
        Z = lam * Xt
        # eps_F via ridge
        T_oh = np.eye(K)[T]
        design = np.hstack([Z, T_oh])
        dd = design.shape[1]
        A = design.T @ design + 0.01 * np.eye(dd)
        b = design.T @ Y
        try:
            beta = np.linalg.solve(A, b)
            Y_hat = design @ beta
            eps_f_arr[i] = float(np.mean((Y_hat - Y) ** 2))
        except np.linalg.LinAlgError:
            eps_f_arr[i] = 1e6
        if strategy == "agg" and emb is not None:
            E = emb[T]
            R_arr[i] = strat.imbalance(Z, T, K, sigma=sigma, embeddings=E)
        else:
            R_arr[i] = strat.imbalance(Z, T, K, sigma=sigma)

    # Profile criterion
    Q_values = np.zeros(len(alpha_grid))
    R_star = 0.0
    alpha_hat = alpha_grid[0]

    for j, alpha in enumerate(alpha_grid):
        objectives = eps_f_arr + alpha * R_arr
        best = int(np.argmin(objectives))
        comp = 5.0 * np.sqrt(2.0 * np.log(1.0 / 0.05) / n) / (1.0 + alpha)
        Q_values[j] = eps_f_arr[best] + alpha * R_arr[best] + comp

    best_j = int(np.argmin(Q_values))
    alpha_hat = float(alpha_grid[best_j])

    # R_star at optimal alpha
    objectives = eps_f_arr + alpha_hat * R_arr
    best = int(np.argmin(objectives))
    R_star = float(R_arr[best])

    # Curvature
    if 0 < best_j < len(Q_values) - 1:
        da = alpha_grid[1] - alpha_grid[0]
        kappa = max((Q_values[best_j + 1] - 2 * Q_values[best_j] + Q_values[best_j - 1]) / (da ** 2), 1e-8)
    else:
        kappa = 1.0

    return alpha_hat, R_star, kappa


def monte_carlo_alpha_hat(
    K: int, n: int, strategy: str,
    n_resamples: int = 50, d: int = 20, seed_base: int = 1000,
    population_n: int = 2000,
    alpha_grid: np.ndarray | None = None,
) -> dict:
    """Monte Carlo: compute alpha_hat for each resample + population alpha_bd."""
    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 5.0, 26)
    W = fixed_projection(d, seed=0)

    # Precompute sigma once from reference data
    ref_data = generate_hard_setting(N=1000, K=K, d=d, seed=99998)
    sigma_ref = median_heuristic(np.tanh(ref_data["X"] @ W))
    emb = get_fixed_embeddings(K)

    # Population level
    pop_data = generate_hard_setting(N=population_n, K=K, d=d, seed=99999)
    alpha_bd, R_bd, kappa_S = compute_profile_criterion(
        pop_data["X"], pop_data["T"], pop_data["Y"], K, strategy,
        sigma_ref, alpha_grid, population_n, W, emb=emb,
    )

    # Resamples
    alpha_hats = np.zeros(n_resamples)
    R_stars = np.zeros(n_resamples)
    for r in range(n_resamples):
        data = generate_hard_setting(N=n, K=K, d=d, seed=seed_base + r)
        ah, rs, _ = compute_profile_criterion(
            data["X"], data["T"], data["Y"], K, strategy,
            sigma_ref, alpha_grid, n, W, emb=emb,
        )
        alpha_hats[r] = ah
        R_stars[r] = rs
        if (r + 1) % 25 == 0:
            from verifiers.common import log
            log(f"      resample {r+1}/{n_resamples} (K={K}, {strategy})")

    deviations = np.abs(alpha_hats - alpha_bd)
    return {
        "K": K, "n": n, "strategy": strategy,
        "alpha_hats": alpha_hats,
        "alpha_bd": alpha_bd,
        "deviations": deviations,
        "kappa_S": kappa_S,
        "r_S": float(np.std(R_stars)),
        "R_stars": R_stars,
    }
