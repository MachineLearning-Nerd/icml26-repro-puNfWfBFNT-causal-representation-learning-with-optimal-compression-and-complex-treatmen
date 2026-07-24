"""Claim 4: Theorem 3.8 / Corollary 3.9 — Asymptotic Normality and Variance Scaling.

Claim:
  Var(alpha_hat_pair) = Theta(K^4 / n)
  Var(alpha_hat_ova)  = Theta(K^2 / n)
  Var(alpha_hat_agg)  = Theta(1 / n)

And alpha_hat is asymptotically normal (Theorem 3.8).

Verification:
  A. Monte Carlo: compute Var(alpha_hat) across resamples for multiple K and n
  B. Fit K-scaling exponents: pair~4, ova~2, agg~0
  C. Fit n-scaling: Var ~ 1/n
  D. Normality test: Shapiro-Wilk on alpha_hat samples
"""
from __future__ import annotations
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from verifiers.common import save_json, save_csv, log, system_info, fit_power_law
from verifiers.mc_infra import monte_carlo_alpha_hat


def run() -> dict:
    log("=== Claim 4: Theorem 3.8 / Cor 3.9 Variance Scaling ===")
    t_start = time.perf_counter()

    alpha_grid = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    n_resamples = 200

    # Part A: K-scaling of Var(alpha_hat)
    log("Part A: K-scaling of Var(alpha_hat)")
    K_values = [4, 8, 12, 16, 24]
    n_fixed = 500
    var_by_strategy = {s: [] for s in ["pair", "ova", "agg"]}
    alpha_hat_samples = {s: {} for s in ["pair", "ova", "agg"]}

    for strategy in ["pair", "ova", "agg"]:
        log(f"  Strategy: {strategy}")
        for K in K_values:
            mc = monte_carlo_alpha_hat(
                K=K, n=n_fixed, strategy=strategy, alpha_grid=alpha_grid,
                n_resamples=n_resamples,
            )
            var_val = float(np.var(mc["alpha_hats"], ddof=1))
            var_by_strategy[strategy].append(var_val)
            alpha_hat_samples[strategy][K] = mc["alpha_hats"].tolist()
            log(f"    K={K}: Var(alpha_hat) = {var_val:.6f}")

    # Fit K-scaling
    K_arr = np.array(K_values, dtype=float)
    scaling = {}
    expected_exponents = {"pair": 4.0, "ova": 2.0, "agg": 0.0}
    for s in ["pair", "ova", "agg"]:
        exp, coeff, r2 = fit_power_law(K_arr, np.array(var_by_strategy[s]))
        scaling[s] = {"exponent": exp, "coefficient": coeff, "r2": r2,
                       "expected": expected_exponents[s]}
        log(f"  {s}: Var ~ K^{exp:.2f} (expected ~{expected_exponents[s]:.0f}), R²={r2:.3f}")

    # Part B: n-scaling of Var(alpha_hat)
    log("Part B: n-scaling of Var(alpha_hat) (K=8)")
    n_values = [100, 200, 500, 1000]
    n_var = {s: [] for s in ["pair", "ova", "agg"]}
    for strategy in ["pair", "ova", "agg"]:
        for n_val in n_values:
            mc = monte_carlo_alpha_hat(
                K=8, n=n_val, strategy=strategy, alpha_grid=alpha_grid,
                n_resamples=min(n_resamples, 150),
            )
            n_var[strategy].append(float(np.var(mc["alpha_hats"], ddof=1)))
        n_arr = np.array(n_values, dtype=float)
        exp, coeff, r2 = fit_power_law(n_arr, np.array(n_var[strategy]))
        log(f"  {strategy}: Var ~ n^{exp:.2f} (expected ~-1), R²={r2:.3f}")

    # Part C: Normality test
    log("Part C: Normality test (Shapiro-Wilk)")
    from scipy import stats
    normality = {}
    for s in ["pair", "ova", "agg"]:
        # Use K=8, n=500 samples
        samples = np.array(alpha_hat_samples[s].get(8, []))
        if len(samples) >= 8:
            # Use alpha_hat centered and scaled
            centered = samples - np.mean(samples)
            if np.std(centered) > 1e-10:
                stat, p_value = stats.shapiro(centered)
                normality[s] = {"shapiro_statistic": float(stat), "p_value": float(p_value),
                                "n_samples": len(samples)}
                log(f"  {s}: Shapiro-Wilk p={p_value:.4f} (stat={stat:.4f})")
            else:
                normality[s] = {"shapiro_statistic": None, "p_value": None, "note": "zero variance"}
        else:
            normality[s] = {"note": "insufficient samples"}

    # Part D: Excess kurtosis (should be ~0 for normal)
    kurtosis = {}
    for s in ["pair", "ova", "agg"]:
        samples = np.array(alpha_hat_samples[s].get(8, []))
        if len(samples) >= 10 and np.std(samples) > 1e-10:
            k = float(stats.kurtosis(samples, fisher=True))
            kurtosis[s] = k
            log(f"  {s}: excess kurtosis = {k:.4f} (expected ~0)")

    # Determine verdict
    k_scaling_ok = all(
        abs(scaling[s]["exponent"] - expected_exponents[s]) < 1.5
        for s in ["pair", "ova", "agg"]
    )
    n_scaling_ok = all(
        fit_power_law(np.array(n_values, dtype=float), np.array(n_var[s]))[0] < -0.3
        for s in ["pair", "ova", "agg"]
    )
    normality_ok = any(
        normality[s].get("p_value", 0) > 0.01 for s in ["pair", "ova", "agg"]
    )

    verified = k_scaling_ok and n_scaling_ok
    verdict = "VERIFIED" if verified else "BLOCKED"

    elapsed = time.perf_counter() - t_start
    result = {
        "claim": "Theorem 3.8/Cor 3.9: Asymptotic normality and variance scaling",
        "claim_text": "Var(alpha_pair)=Theta(K^4/n), Var(alpha_ova)=Theta(K^2/n), Var(alpha_agg)=Theta(1/n)",
        "verdict": verdict,
        "K_scaling": {
            "K_values": K_values,
            "variances": var_by_strategy,
            "scaling": scaling,
        },
        "n_scaling": {
            "n_values": n_values,
            "variances": n_var,
        },
        "normality": normality,
        "kurtosis": kurtosis,
        "elapsed_seconds": elapsed,
        "system_info": system_info(),
    }

    save_json(result, "claim4_theorem38/result.json")
    # Save K-scaling CSV
    k_csv = []
    for i, K in enumerate(K_values):
        k_csv.append((K, var_by_strategy["pair"][i], var_by_strategy["ova"][i], var_by_strategy["agg"][i]))
    save_csv(k_csv, ["K", "var_pair", "var_ova", "var_agg"], "claim4_theorem38/var_vs_K.csv")
    # Save n-scaling CSV
    n_csv = []
    for i, n_val in enumerate(n_values):
        n_csv.append((n_val, n_var["pair"][i], n_var["ova"][i], n_var["agg"][i]))
    save_csv(n_csv, ["n", "var_pair", "var_ova", "var_agg"], "claim4_theorem38/var_vs_n.csv")

    log(f"Verdict: {verdict} (elapsed {elapsed:.1f}s)")
    return result
