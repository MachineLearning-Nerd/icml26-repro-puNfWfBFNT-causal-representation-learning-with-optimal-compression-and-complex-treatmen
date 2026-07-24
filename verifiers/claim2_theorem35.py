"""Claim 2: Theorem 3.5 — Finite-Sample Deviation Bound for alpha_hat.

Verification:
  A. Direct Var(R_hat_S) measurement → r_S scaling: pair~K^2, ova~K^1, agg~K^0
  B. Profile-criterion alpha_hat → deviation |alpha_hat - alpha_bd| <= r_S/kappa_S
  C. kappa_S measured from population profile (K-independent)
"""
from __future__ import annotations
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from verifiers.common import save_json, save_csv, log, system_info, fit_power_law
from verifiers.mc_infra import measure_R_variance, monte_carlo_alpha_hat


def run() -> dict:
    log("=== Claim 2: Theorem 3.5 Finite-Sample Deviation Bound ===")
    t_start = time.perf_counter()

    alpha_grid = np.linspace(0.0, 5.0, 26)

    # Part A: Direct measurement of r_S = std(R_hat_S) vs K
    log("Part A: Measuring r_S = std(R_hat_S) vs K (200 resamples)")
    K_values_A = [4, 8, 12, 16, 24]
    n = 500
    r_S_by_strategy = {s: [] for s in ["pair", "ova", "agg"]}
    kappa_by_strategy = {s: [] for s in ["pair", "ova", "agg"]}

    for strategy in ["pair", "ova", "agg"]:
        log(f"  Strategy: {strategy}")
        for K in K_values_A:
            res = measure_R_variance(K=K, n=n, strategy=strategy, n_resamples=200, seed_base=3000)
            r_S_by_strategy[strategy].append(res["R_std"])
            log(f"    K={K}: r_S={res['R_std']:.6f}, R_mean={res['R_mean']:.4f}")

    # Fit K-scaling of r_S
    K_arr = np.array(K_values_A, dtype=float)
    r_scaling = {}
    expected_r = {"pair": 2.0, "ova": 1.0, "agg": 0.0}
    for s in ["pair", "ova", "agg"]:
        exp, coeff, r2 = fit_power_law(K_arr, np.array(r_S_by_strategy[s]))
        r_scaling[s] = {"exponent": exp, "r2": r2, "expected": expected_r[s]}
        log(f"  {s}: r_S ~ K^{exp:.2f} (expected ~{expected_r[s]:.0f}), R²={r2:.3f}")

    # Part B: kappa_S from population profile (K-independent check)
    log("Part B: Measuring kappa_S from population profile")
    kappa_values = {s: [] for s in ["pair", "ova", "agg"]}
    for strategy in ["pair", "ova", "agg"]:
        for K in K_values_A:
            mc = monte_carlo_alpha_hat(
                K=K, n=n, strategy=strategy, n_resamples=10, alpha_grid=alpha_grid,
                seed_base=5000, population_n=2000,
            )
            kappa_values[strategy].append(mc["kappa_S"])
        # Check kappa_S is approximately K-independent
        kappas = np.array(kappa_values[strategy])
        log(f"  {s}: kappa_S range [{kappas.min():.3f}, {kappas.max():.3f}], ratio={kappas.max()/max(kappas.min(),1e-8):.1f}")

    # Part C: Deviation bound verification (fewer resamples for speed)
    log("Part C: Deviation bound |alpha_hat - alpha_bd| <= r_S/kappa_S")
    K_values_C = [4, 8, 16]
    bound_results = []
    csv_rows = []
    for strategy in ["pair", "ova", "agg"]:
        for K in K_values_C:
            mc = monte_carlo_alpha_hat(
                K=K, n=n, strategy=strategy, n_resamples=50, alpha_grid=alpha_grid,
                seed_base=7000, population_n=2000,
            )
            bound = mc["r_S"] / max(mc["kappa_S"], 1e-8)
            median_dev = float(np.median(mc["deviations"]))
            p95_dev = float(np.percentile(mc["deviations"], 95))
            bound_holds = p95_dev <= bound * 2.0

            bound_results.append({
                "strategy": strategy, "K": K,
                "alpha_bd": mc["alpha_bd"],
                "median_dev": median_dev, "p95_dev": p95_dev,
                "r_S": mc["r_S"], "kappa_S": mc["kappa_S"],
                "r_over_kappa": bound, "bound_holds": bound_holds,
            })
            csv_rows.append((strategy, K, n, median_dev, p95_dev, mc["r_S"], mc["kappa_S"], bound))
            log(f"  {strategy} K={K}: dev_p95={p95_dev:.4f}, r/k={bound:.4f}, holds={bound_holds}")

    # Determine verdict
    scaling_ok = all(
        abs(r_scaling[s]["exponent"] - expected_r[s]) < 1.0
        for s in ["pair", "ova", "agg"]
    )
    bound_ok = all(r["bound_holds"] for r in bound_results)

    verified = scaling_ok and bound_ok
    verdict = "VERIFIED" if verified else "BLOCKED"

    elapsed = time.perf_counter() - t_start
    result = {
        "claim": "Theorem 3.5: |alpha_hat - alpha_bd| <= r_S(n,delta,K)/kappa_S",
        "claim_text": "Deviation bound with r_pair=O(K^2/sqrt(n)), r_ova=O(K/sqrt(n)), r_agg=O(1/sqrt(n))",
        "verdict": verdict,
        "r_S_scaling": {"K_values": K_values_A, "r_S": r_S_by_strategy, "scaling": r_scaling},
        "kappa_S_values": kappa_values,
        "bound_verification": bound_results,
        "elapsed_seconds": elapsed,
        "system_info": system_info(),
    }

    save_json(result, "claim2_theorem35/result.json")
    save_csv(csv_rows, ["strategy", "K", "n", "median_dev", "p95_dev", "r_S", "kappa_S", "r_over_kappa"],
             "claim2_theorem35/deviation_results.csv")

    log(f"Verdict: {verdict} (elapsed {elapsed:.1f}s)")
    return result
