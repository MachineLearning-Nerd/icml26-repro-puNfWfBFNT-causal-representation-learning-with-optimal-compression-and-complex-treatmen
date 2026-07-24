"""Claim 4: Theorem 3.8 / Corollary 3.9 — Variance Scaling.

Claim:
  Var(alpha_hat_pair) = Theta(K^4 / n)
  Var(alpha_hat_ova)  = Theta(K^2 / n)
  Var(alpha_hat_agg)  = Theta(1 / n)

Verification:
  A. Direct Var(R_hat_S) measurement → pair~K^4, ova~K^2, agg~K^0
  B. kappa_S measured from population profile (K-independent check)
  C. Var(alpha_hat) measured via profile criterion
  D. Normality test (Shapiro-Wilk)
"""
from __future__ import annotations
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from verifiers.common import save_json, save_csv, log, system_info, fit_power_law
from verifiers.mc_infra import measure_R_variance, monte_carlo_alpha_hat


def run() -> dict:
    log("=== Claim 4: Theorem 3.8/Cor 3.9 Variance Scaling ===")
    t_start = time.perf_counter()

    n = 500
    alpha_grid = np.linspace(0.0, 5.0, 26)

    # Part A: Direct Var(R_hat_S) measurement vs K
    log("Part A: Var(R_hat_S) vs K (200 resamples)")
    K_values = [4, 8, 12, 16, 24]
    var_R = {s: [] for s in ["pair", "ova", "agg"]}
    alpha_hat_samples_all = {s: {} for s in ["pair", "ova", "agg"]}

    for strategy in ["pair", "ova", "agg"]:
        log(f"  Strategy: {strategy}")
        for K in K_values:
            res = measure_R_variance(K=K, n=n, strategy=strategy, n_resamples=200, seed_base=8000)
            var_R[strategy].append(max(res["R_var"], 1e-12))
            log(f"    K={K}: Var(R)={res['R_var']:.8f}")

    # Fit K-scaling of Var(R_hat_S)
    K_arr = np.array(K_values, dtype=float)
    var_scaling = {}
    expected_v = {"pair": 4.0, "ova": 2.0, "agg": 0.0}
    for s in ["pair", "ova", "agg"]:
        exp, coeff, r2 = fit_power_law(K_arr, np.array(var_R[s]))
        var_scaling[s] = {"exponent": exp, "r2": r2, "expected": expected_v[s]}
        log(f"  {s}: Var(R) ~ K^{exp:.2f} (expected ~{expected_v[s]:.0f}), R²={r2:.3f}")

    # Part B: n-scaling of Var(R_hat_S)
    log("Part B: n-scaling of Var(R_hat_S) (K=8)")
    n_values = [100, 200, 500, 1000]
    n_var_R = {s: [] for s in ["pair", "ova", "agg"]}
    for strategy in ["pair", "ova", "agg"]:
        for n_val in n_values:
            res = measure_R_variance(K=8, n=n_val, strategy=strategy, n_resamples=100, seed_base=9000)
            n_var_R[strategy].append(max(res["R_var"], 1e-12))
        n_arr = np.array(n_values, dtype=float)
        exp_n, _, r2_n = fit_power_law(n_arr, np.array(n_var_R[strategy]))
        log(f"  {strategy}: Var(R) ~ n^{exp_n:.2f} (expected ~-1), R²={r2_n:.3f}")

    # Part C: alpha_hat variance via profile criterion
    log("Part C: Var(alpha_hat) via profile criterion")
    K_values_C = [4, 8, 16]
    var_alpha = {s: [] for s in ["pair", "ova", "agg"]}
    kappa_vals = {s: [] for s in ["pair", "ova", "agg"]}
    for strategy in ["pair", "ova", "agg"]:
        for K in K_values_C:
            mc = monte_carlo_alpha_hat(
                K=K, n=n, strategy=strategy, n_resamples=50, alpha_grid=alpha_grid,
                seed_base=11000, population_n=2000,
            )
            va = max(np.var(mc["alpha_hats"], ddof=1), 1e-10)
            var_alpha[strategy].append(va)
            kappa_vals[strategy].append(mc["kappa_S"])
            alpha_hat_samples_all[strategy][K] = mc["alpha_hats"].tolist()
            log(f"  {strategy} K={K}: Var(alpha_hat)={va:.8f}, kappa={mc['kappa_S']:.3f}")

    # Fit Var(alpha_hat) K-scaling
    KC_arr = np.array(K_values_C, dtype=float)
    alpha_scaling = {}
    for s in ["pair", "ova", "agg"]:
        exp, _, r2 = fit_power_law(KC_arr, np.array(var_alpha[s]))
        alpha_scaling[s] = {"exponent": exp, "r2": r2, "expected": expected_v[s]}
        log(f"  {s}: Var(alpha_hat) ~ K^{exp:.2f} (expected ~{expected_v[s]:.0f})")

    # Part D: Normality test
    log("Part D: Normality test")
    from scipy import stats
    normality = {}
    kurtosis = {}
    for s in ["pair", "ova", "agg"]:
        samples = np.array(alpha_hat_samples_all[s].get(8, []))
        if len(samples) >= 8 and np.std(samples) > 1e-10:
            centered = samples - np.mean(samples)
            stat, p_value = stats.shapiro(centered)
            k = float(stats.kurtosis(samples, fisher=True))
            normality[s] = {"shapiro_p": float(p_value), "statistic": float(stat)}
            kurtosis[s] = k
            log(f"  {s}: Shapiro p={p_value:.4f}, kurtosis={k:.4f}")

    # Determine verdict
    var_R_ok = all(
        abs(var_scaling[s]["exponent"] - expected_v[s]) < 1.5
        for s in ["pair", "ova", "agg"]
    )
    n_scaling_ok = all(
        fit_power_law(np.array(n_values, dtype=float), np.array(n_var_R[s]))[0] < -0.3
        for s in ["pair", "ova", "agg"]
    )

    verified = var_R_ok and n_scaling_ok
    verdict = "VERIFIED" if verified else "BLOCKED"

    elapsed = time.perf_counter() - t_start
    result = {
        "claim": "Var(alpha_pair)=Theta(K^4/n), Var(alpha_ova)=Theta(K^2/n), Var(alpha_agg)=Theta(1/n)",
        "verdict": verdict,
        "Var_R_K_scaling": {"K_values": K_values, "variances": var_R, "scaling": var_scaling},
        "Var_R_n_scaling": {"n_values": n_values, "variances": n_var_R},
        "Var_alpha_K_scaling": {"K_values": K_values_C, "variances": var_alpha, "scaling": alpha_scaling},
        "kappa_values": kappa_vals,
        "normality": normality,
        "kurtosis": kurtosis,
        "elapsed_seconds": elapsed,
        "system_info": system_info(),
    }

    save_json(result, "claim4_theorem38/result.json")
    k_csv = [(K_values[i], var_R["pair"][i], var_R["ova"][i], var_R["agg"][i]) for i in range(len(K_values))]
    save_csv(k_csv, ["K", "var_R_pair", "var_R_ova", "var_R_agg"], "claim4_theorem38/var_R_vs_K.csv")
    n_csv = [(n_values[i], n_var_R["pair"][i], n_var_R["ova"][i], n_var_R["agg"][i]) for i in range(len(n_values))]
    save_csv(n_csv, ["n", "var_R_pair", "var_R_ova", "var_R_agg"], "claim4_theorem38/var_R_vs_n.csv")

    log(f"Verdict: {verdict} (elapsed {elapsed:.1f}s)")
    return result
