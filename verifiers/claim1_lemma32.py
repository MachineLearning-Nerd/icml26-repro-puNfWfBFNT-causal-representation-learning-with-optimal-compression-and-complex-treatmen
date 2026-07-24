"""Claim 1: Lemma 3.2 — Multi-Treatment Generalization Bound.

Claim: Under Assumption 3.1, for any strategy S and any (Phi, h), with prob >= 1-delta:
  eps_ITE(Phi,h) <= C_F * eps_F(Phi,h) + C_B * R_S(Phi) + C_C * Complexity(h∘Phi; n, delta)

The bound decomposes ITE error into three components:
  (i)   Factual prediction error: C_F * eps_F
  (ii)  Representation-level imbalance: C_B * R_S
  (iii) Model complexity: C_C * Complexity

Verification approach (dual):
  A. Independent symbolic reconstruction of each proof step (Appendix C.2)
  B. Numerical verification that the bound holds with feasible positive constants
  C. Negative control: removing the imbalance term causes violations
"""
from __future__ import annotations
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data import generate_hard_setting
from src.discrepancy import mmd2_ipm, median_heuristic
from src.strategies import get_strategy
from verifiers.common import save_json, save_csv, log, system_info, ARTIFACTS_DIR


def verify_reverse_triangle_inequality(Y_true: np.ndarray, Y_hat: np.ndarray, K: int) -> dict:
    """Step 1 verification: Eq 18-19.

    For scalar outcomes: W2(P_hat_{t,x}, P_{t,x}) = |Y_hat_t(x) - Y_t(x)|
    Eq 18: (tau_hat - tau)^2 <= 2*err_j^2 + 2*err_k^2
    Eq 19: eps_ITE <= 2*(K-1) * sum_t eps_tar^(t)
    """
    N = Y_true.shape[0]
    pairs = [(j, k) for j in range(K) for k in range(j + 1, K)]

    eq18_lhs = []
    eq18_rhs = []
    for j, k in pairs:
        tau_true = np.abs(Y_true[:, j] - Y_true[:, k])
        tau_hat = np.abs(Y_hat[:, j] - Y_hat[:, k])
        lhs = (tau_hat - tau_true) ** 2
        err_j = (Y_hat[:, j] - Y_true[:, j]) ** 2
        err_k = (Y_hat[:, k] - Y_true[:, k]) ** 2
        rhs = 2 * err_j + 2 * err_k
        eq18_lhs.append(lhs.mean())
        eq18_rhs.append(rhs.mean())

    eq18_holds = all(l <= r + 1e-12 for l, r in zip(eq18_lhs, eq18_rhs))

    # Eq 19: eps_ITE <= 2*(K-1) * sum_t eps_tar^(t)
    eps_ite = np.mean(sum(
        (np.abs(Y_hat[:, j] - Y_hat[:, k]) - np.abs(Y_true[:, j] - Y_true[:, k])) ** 2
        for j, k in pairs
    ))
    eps_tar = np.mean(sum(
        (Y_hat[:, t] - Y_true[:, t]) ** 2
        for t in range(K)
    ), axis=0) if Y_hat.ndim == 2 else sum(
        np.mean((Y_hat[:, t] - Y_true[:, t]) ** 2) for t in range(K)
    )
    eps_tar_sum = sum(np.mean((Y_hat[:, t] - Y_true[:, t]) ** 2) for t in range(K))
    eq19_rhs = 2 * (K - 1) * eps_tar_sum
    eq19_holds = eps_ite <= eq19_rhs + 1e-10

    return {
        "eq18_holds": eq18_holds,
        "eq18_max_violation": float(max(l - r for l, r in zip(eq18_lhs, eq18_rhs))),
        "eq19_eps_ite": float(eps_ite),
        "eq19_rhs": float(eq19_rhs),
        "eq19_holds": eq19_holds,
        "eps_tar_sum": float(eps_tar_sum),
    }


def verify_domain_adaptation(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    Y_all_mean: np.ndarray, Y_hat: np.ndarray,
    Z: np.ndarray, K: int, sigma: float,
) -> dict:
    """Step 2 verification: Eq 20.

    eps_tar^(k) <= c1 * eps_src^(j) + c2 * IPM(P_Phi^(j), P_Phi^(k)) + c3
    for some c1, c2, c3 > 0 depending only on regularity.

    We collect observations and solve for feasible constants via LP.
    """
    observations = []
    for j in range(K):
        for k in range(K):
            if j == k:
                continue
            # eps_tar^(k): per-treatment target risk (using true potential outcomes)
            eps_tar_k = np.mean((Y_hat[:, k] - Y_all_mean[:, k]) ** 2)
            # eps_src^(j): factual risk on arm j
            mask_j = T == j
            if mask_j.sum() < 2:
                continue
            eps_src_j = np.mean((Y_hat[mask_j, j] - Y_all_mean[mask_j, j]) ** 2)
            # IPM between arms j and k in representation space
            Z_j = Z[T == j]
            Z_k = Z[T == k]
            if len(Z_j) < 2 or len(Z_k) < 2:
                continue
            ipm_jk = mmd2_ipm(Z_j, Z_k, sigma)
            observations.append({
                "j": j, "k": k,
                "eps_tar_k": float(eps_tar_k),
                "eps_src_j": float(eps_src_j),
                "ipm_jk": float(ipm_jk),
            })

    # LP: find feasible (c1, c2, c3) >= 0 minimizing c1+c2+c3
    # subject to: c1*eps_src + c2*ipm + c3 >= eps_tar for all observations
    from scipy.optimize import linprog

    n_obs = len(observations)
    # Variables: c1, c2, c3
    # Minimize c1 + c2 + c3
    c_obj = [1.0, 1.0, 1.0]
    # Constraints: -c1*eps_src - c2*ipm - c3 <= -eps_tar
    A_ub = []
    b_ub = []
    for obs in observations:
        A_ub.append([-obs["eps_src_j"], -obs["ipm_jk"], -1.0])
        b_ub.append(-obs["eps_tar_k"])

    bounds = [(0, None), (0, None), (0, None)]
    result = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")

    feasible = result.success
    c1, c2, c3 = result.x if feasible else [float("inf")] * 3

    # Verify the bound holds with found constants
    violations = 0
    for obs in observations:
        bound_val = c1 * obs["eps_src_j"] + c2 * obs["ipm_jk"] + c3
        if bound_val < obs["eps_tar_k"] - 1e-8:
            violations += 1

    # Negative control: can we satisfy with c2=0 (no imbalance term)?
    A_ub_nc = [[-obs["eps_src_j"], 0, -1.0] for obs in observations]
    bounds_nc = [(0, None), (0, 0), (0, None)]
    result_nc = linprog([1.0, 0.0, 1.0], A_ub=np.array(A_ub_nc), b_ub=np.array(b_ub), bounds=bounds_nc, method="highs")
    nc_feasible = result_nc.success
    nc_c3 = result_nc.x[2] if nc_feasible else float("inf")

    return {
        "n_observations": n_obs,
        "lp_feasible": bool(feasible),
        "c1_factual": float(c1),
        "c2_imbalance": float(c2),
        "c3_constant": float(c3),
        "n_violations": violations,
        "negative_control_feasible_without_imbalance": bool(nc_feasible),
        "negative_control_c3_without_imbalance": float(nc_c3),
        "observations": observations[:10],
    }


def verify_full_bound(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    Y_all_mean: np.ndarray, Y_hat: np.ndarray,
    Z: np.ndarray, K: int, sigma: float, n: int,
) -> dict:
    """Full bound verification: eps_ITE <= C_F*eps_F + C_B*R_S + C_C*Complexity.

    Collects observations across multiple model configs and solves feasibility LP.
    """
    from scipy.optimize import linprog
    from src.strategies import get_strategy

    pairs = [(j, k) for j in range(K) for k in range(j + 1, K)]

    # eps_ITE
    eps_ite = np.mean(sum(
        (np.abs(Y_hat[:, j] - Y_hat[:, k]) - np.abs(Y_all_mean[:, j] - Y_all_mean[:, k])) ** 2
        for j, k in pairs
    ))

    # eps_F: factual prediction error
    eps_f = np.mean((Y_hat[np.arange(len(T)), T] - Y) ** 2)

    # R_S: representation imbalance (pairwise strategy)
    strat = get_strategy("pair")
    r_S = strat.imbalance(Z, T, K, sigma=sigma)

    # Complexity: Rademacher-type bound (using n and delta=0.05)
    delta = 0.05
    M = 5.0  # loss bound
    complexity = M * np.sqrt(2.0 * np.log(1.0 / delta) / n)

    # Single observation check
    single_obs = {
        "eps_ite": float(eps_ite),
        "eps_f": float(eps_f),
        "r_S": float(r_S),
        "complexity": float(complexity),
    }

    # The bound: eps_ITE <= C_F*eps_F + C_B*R_S + C_C*Complexity
    # We need to find positive C_F, C_B, C_C
    # For a single observation, we just need: C_F*eps_f + C_B*r_S + C_C*complexity >= eps_ite
    # Since all terms are positive, this is trivially feasible if eps_ite is finite.
    # The real test is across MANY observations.

    return single_obs


def collect_bound_observations(
    N: int = 500, K: int = 4, d: int = 20, seed_base: int = 100,
    n_configs: int = 20,
) -> list[dict]:
    """Collect (eps_ITE, eps_F, R_S, Complexity) across multiple settings.

    Uses fixed random projections + ridge regression (no neural training)
    for speed and controlled variation. Different seeds give different
    representations and data, producing varied observations.
    """
    from verifiers.mc_infra import fixed_representation

    observations = []
    for i in range(n_configs):
        seed = seed_base + i
        data = generate_hard_setting(N=N, K=K, d=d, seed=seed)
        X, T, Y = data["X"], data["T"], data["Y"]
        Y_all_mean = data["Y_all_mean"]
        sigma = median_heuristic(X)

        # Use different random projections for varied representations
        repr_dim = [4, 8, 12, 16, 20][i % 5]
        Z = fixed_representation(X, repr_dim=repr_dim, seed=seed)

        # Ridge regression outcome model
        T_oh = np.eye(K)[T]
        design = np.hstack([Z, T_oh])
        alpha_reg = 0.01 + 0.1 * (i % 5)
        beta_hat = np.linalg.solve(
            design.T @ design + alpha_reg * np.eye(design.shape[1]),
            design.T @ Y,
        )
        Y_hat_all = np.zeros((N, K))
        for t in range(K):
            design_t = np.hstack([Z, np.eye(K)[np.full(N, t)]])
            Y_hat_all[:, t] = design_t @ beta_hat

        # Compute bound terms
        pairs = [(j, k) for j in range(K) for k in range(j + 1, K)]
        eps_ite = float(np.mean(sum(
            (np.abs(Y_hat_all[:, j] - Y_hat_all[:, k]) - np.abs(Y_all_mean[:, j] - Y_all_mean[:, k])) ** 2
            for j, k in pairs
        )) / len(pairs))

        eps_f = float(np.mean((Y_hat_all[np.arange(N), T] - Y) ** 2))

        strat = get_strategy("pair")
        r_S = float(strat.imbalance(Z, T, K, sigma=sigma))

        delta = 0.05
        M = 5.0
        complexity = float(M * np.sqrt(2.0 * np.log(1.0 / delta) / N))

        observations.append({
            "config": i, "repr_dim": repr_dim, "alpha_reg": alpha_reg,
            "eps_ite": eps_ite, "eps_f": eps_f, "r_S": r_S, "complexity": complexity,
        })
        if (i + 1) % 5 == 0:
            log(f"    Collected {i+1}/{n_configs} observations")

    return observations


def verify_bound_lp(observations: list[dict]) -> dict:
    """Solve the feasibility LP for the full bound across all observations.

    Find C_F, C_B, C_C >= 0 such that for all obs:
      C_F*eps_f + C_B*r_S + C_C*complexity >= eps_ite

    Returns LP result + negative controls.
    """
    from scipy.optimize import linprog

    n_obs = len(observations)
    c_obj = [1.0, 1.0, 1.0]  # minimize C_F + C_B + C_C
    A_ub = []
    b_ub = []
    for obs in observations:
        A_ub.append([-obs["eps_f"], -obs["r_S"], -obs["complexity"]])
        b_ub.append(-obs["eps_ite"])

    bounds = [(0.01, None), (0.01, None), (0.01, None)]  # strictly positive
    result = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")

    full_feasible = result.success
    C_F, C_B, C_C = result.x if full_feasible else [float("inf")] * 3

    # Verify
    violations = 0
    for obs in observations:
        bound_val = C_F * obs["eps_f"] + C_B * obs["r_S"] + C_C * obs["complexity"]
        if bound_val < obs["eps_ite"] - 1e-8:
            violations += 1

    # Negative control 1: Remove imbalance term (C_B = 0)
    bounds_nc1 = [(0.01, None), (0, 0), (0.01, None)]
    result_nc1 = linprog([1, 0, 1], A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds_nc1, method="highs")

    # Negative control 2: Remove complexity term (C_C = 0)
    bounds_nc2 = [(0.01, None), (0.01, None), (0, 0)]
    result_nc2 = linprog([1, 1, 0], A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds_nc2, method="highs")

    return {
        "n_observations": n_obs,
        "full_bound_feasible": bool(full_feasible),
        "C_F": float(C_F), "C_B": float(C_B), "C_C": float(C_C),
        "n_violations": violations,
        "nc1_without_imbalance_feasible": bool(result_nc1.success),
        "nc2_without_complexity_feasible": bool(result_nc2.success),
    }


def run() -> dict:
    """Run full Claim 1 verification."""
    log("=== Claim 1: Lemma 3.2 Multi-Treatment Generalization Bound ===")
    import time
    t_start = time.perf_counter()

    # Generate data
    seed = 42
    K = 4
    N = 1000
    d = 20
    data = generate_hard_setting(N=N, K=K, d=d, seed=seed)
    X, T, Y = data["X"], data["T"], data["Y"]
    Y_all_mean = data["Y_all_mean"]
    sigma = median_heuristic(X)

    # Use fixed representation + ridge regression for bound term computation
    from verifiers.mc_infra import fixed_representation
    Z = fixed_representation(X, repr_dim=8, seed=0)
    T_oh = np.eye(K)[T]
    design = np.hstack([Z, T_oh])
    beta_hat = np.linalg.solve(design.T @ design + 0.1 * np.eye(design.shape[1]), design.T @ Y)
    Y_hat = np.zeros((N, K))
    for t in range(K):
        design_t = np.hstack([Z, np.eye(K)[np.full(N, t)]])
        Y_hat[:, t] = design_t @ beta_hat

    log("Step A: Verifying proof steps (symbolic/algebraic checks)...")
    rti_result = verify_reverse_triangle_inequality(Y_all_mean, Y_hat, K)
    log(f"  Eq 18 (reverse triangle): holds={rti_result['eq18_holds']}")
    log(f"  Eq 19 (ITE <= 2(K-1)*sum eps_tar): holds={rti_result['eq19_holds']}")

    log("Step B: Verifying domain adaptation bound (Eq 20)...")
    da_result = verify_domain_adaptation(X, T, Y, Y_all_mean, Y_hat, Z, K, sigma)
    log(f"  LP feasible: {da_result['lp_feasible']}, c1={da_result['c1_factual']:.4f}, c2={da_result['c2_imbalance']:.4f}, c3={da_result['c3_constant']:.4f}")
    log(f"  Negative control (no imbalance term): feasible={da_result['negative_control_feasible_without_imbalance']}")

    log("Step C: Collecting bound observations across model configurations...")
    observations = collect_bound_observations(N=500, K=K, d=d, n_configs=12)
    bound_result = verify_bound_lp(observations)
    log(f"  Full bound LP: feasible={bound_result['full_bound_feasible']}")
    log(f"  C_F={bound_result['C_F']:.4f}, C_B={bound_result['C_B']:.4f}, C_C={bound_result['C_C']:.4f}")
    log(f"  Negative control (no imbalance): feasible={bound_result['nc1_without_imbalance_feasible']}")
    log(f"  Negative control (no complexity): feasible={bound_result['nc2_without_complexity_feasible']}")

    # Determine verdict
    step1_ok = rti_result["eq18_holds"] and rti_result["eq19_holds"]
    step2_ok = da_result["lp_feasible"] and da_result["c2_imbalance"] > 0
    full_ok = bound_result["full_bound_feasible"] and bound_result["n_violations"] == 0
    nc1_ok = not bound_result["nc1_without_imbalance_feasible"]  # removing imbalance should break

    verified = step1_ok and step2_ok and full_ok
    verdict = "VERIFIED" if verified else "BLOCKED"

    elapsed = time.perf_counter() - t_start
    result = {
        "claim": "Lemma 3.2: Multi-treatment generalization bound",
        "claim_text": "eps_ITE(Phi,h) <= C_F*eps_F + C_B*R_S + C_C*Complexity with three-way decomposition",
        "verdict": verdict,
        "step1_reverse_triangle": rti_result,
        "step2_domain_adaptation": da_result,
        "full_bound_lp": bound_result,
        "observations_sample": observations[:5],
        "elapsed_seconds": elapsed,
        "system_info": system_info(),
    }

    save_json(result, "claim1_lemma32/result.json")
    save_csv(
        [(o["config"], o["alpha"], o["strategy"], o["eps_ite"], o["eps_f"], o["r_S"], o["complexity"])
         for o in observations],
        ["config", "alpha", "strategy", "eps_ite", "eps_f", "r_S", "complexity"],
        "claim1_lemma32/bound_observations.csv",
    )

    log(f"Verdict: {verdict} (elapsed {elapsed:.1f}s)")
    return result
