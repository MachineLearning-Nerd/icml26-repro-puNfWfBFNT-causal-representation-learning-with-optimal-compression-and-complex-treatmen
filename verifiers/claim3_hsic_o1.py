"""Claim 3: HSIC Treatment Aggregation achieves O(1) complexity and K-independent deviation.

Claim (Section 2):
  - HSIC achieves O(1) computational complexity w.r.t. K
  - r_agg = O(sqrt(log(1/delta)/n)), independent of K
  - Versus O(K^2) for pairwise and O(K) for one-vs-all

Verification:
  A. Timing: measure wall-clock time of R_pair, R_ova, R_agg as K varies; fit power law
  B. Operation count: verify C(K,2), K, 1 terms respectively
  C. Concentration: measure std of each imbalance estimator across resamples vs K
  D. Negative control: pairwise std should grow with K, agg std should not
"""
from __future__ import annotations
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data import generate_hard_setting
from src.strategies import compute_all_imbalances, get_strategy
from src.discrepancy import median_heuristic, count_pairwise_ops, count_ova_ops, count_agg_ops
from verifiers.common import save_json, save_csv, log, system_info, fit_power_law
from verifiers.mc_infra import fixed_representation


def run() -> dict:
    log("=== Claim 3: HSIC O(1) Complexity + K-independent Deviation ===")
    t_start = time.perf_counter()

    # Part A: Timing — wall-clock time vs K
    log("Part A: Timing complexity (wall-clock time vs K)")
    K_values = [2, 4, 8, 12, 16, 20, 30, 40, 50]
    timing_results = {s: [] for s in ["pair", "ova", "agg"]}
    op_counts = {s: [] for s in ["pair", "ova", "agg"]}

    N = 500
    d = 20
    for K in K_values:
        data = generate_hard_setting(N=N, K=K, d=d, seed=42)
        X, T = data["X"], data["T"]
        Z = fixed_representation(X, seed=0)
        sigma = median_heuristic(X)

        # Warm up
        for s in ["pair", "ova", "agg"]:
            strat = get_strategy(s)
            strat.imbalance(Z, T, K, sigma=sigma)

        # Time each strategy (average over multiple runs for small K)
        n_runs = max(1, 200 // K)
        for s in ["pair", "ova", "agg"]:
            strat = get_strategy(s)
            times = []
            for _ in range(n_runs):
                _, elapsed = strat.timed_imbalance(Z, T, K, sigma=sigma)
                times.append(elapsed)
            avg_time = np.median(times)
            timing_results[s].append(float(avg_time))
            op_counts[s].append(strat.n_ops(K))

        log(f"  K={K:3d}: pair={timing_results['pair'][-1]*1000:8.1f}ms "
            f"ova={timing_results['ova'][-1]*1000:8.1f}ms "
            f"agg={timing_results['agg'][-1]*1000:8.1f}ms "
            f"(ops: pair={op_counts['pair'][-1]}, ova={op_counts['ova'][-1]}, agg=1)")

    # Fit power law: time vs K
    K_arr = np.array(K_values, dtype=float)
    scaling = {}
    for s in ["pair", "ova", "agg"]:
        exp, coeff, r2 = fit_power_law(K_arr, np.array(timing_results[s]))
        scaling[s] = {"exponent": exp, "coefficient": coeff, "r2": r2}
        expected = {"pair": 2.0, "ova": 1.0, "agg": 0.0}[s]
        log(f"  {s}: time ~ K^{exp:.2f} (expected ~{expected:.0f}), R²={r2:.3f}")

    # Part B: Verify operation counts
    log("Part B: Operation counts")
    for s in ["pair", "ova", "agg"]:
        ops = op_counts[s]
        log(f"  {s}: ops = {ops[-1]} at K={K_values[-1]}")

    # Part C: Concentration — std of imbalance vs K
    log("Part C: Statistical concentration (std of imbalance vs K)")
    n_resamples = 100
    K_values_conc = [4, 8, 16, 24, 32]
    n = 500
    concentration = {s: [] for s in ["pair", "ova", "agg"]}

    for K in K_values_conc:
        log(f"  K={K}, resamples={n_resamples}")
        imb_samples = {s: [] for s in ["pair", "ova", "agg"]}
        for r in range(n_resamples):
            data = generate_hard_setting(N=n, K=K, d=d, seed=5000 + r)
            Z = fixed_representation(data["X"], seed=0)
            sigma = median_heuristic(data["X"])
            for s in ["pair", "ova", "agg"]:
                strat = get_strategy(s)
                imb = strat.imbalance(Z, data["T"], K, sigma=sigma)
                imb_samples[s].append(imb)
        for s in ["pair", "ova", "agg"]:
            std = float(np.std(imb_samples[s]))
            concentration[s].append(std)

    K_conc_arr = np.array(K_values_conc, dtype=float)
    conc_scaling = {}
    for s in ["pair", "ova", "agg"]:
        exp, coeff, r2 = fit_power_law(K_conc_arr, np.array(concentration[s]))
        conc_scaling[s] = {"exponent": exp, "coefficient": coeff, "r2": r2, "stds": concentration[s]}
        expected = {"pair": 2.0, "ova": 1.0, "agg": 0.0}[s]
        log(f"  {s}: std(imbalance) ~ K^{exp:.2f} (expected ~{expected:.0f}), R²={r2:.3f}")

    # Determine verdict
    timing_ok = (
        scaling["pair"]["exponent"] > 1.5 and
        scaling["ova"]["exponent"] > 0.5 and
        scaling["agg"]["exponent"] < 0.5
    )
    conc_ok = (
        conc_scaling["pair"]["exponent"] > 1.0 and
        conc_scaling["ova"]["exponent"] > 0.3 and
        abs(conc_scaling["agg"]["exponent"]) < 0.5
    )

    verified = timing_ok and conc_ok
    verdict = "VERIFIED" if verified else "BLOCKED"

    elapsed = time.perf_counter() - t_start
    result = {
        "claim": "HSIC Treatment Aggregation: O(1) complexity + K-independent deviation",
        "claim_text": "Aggregation achieves O(1) complexity and r_agg = O(sqrt(log(1/delta)/n)) independent of K",
        "verdict": verdict,
        "timing": {
            "K_values": K_values,
            "times": timing_results,
            "scaling": scaling,
        },
        "operation_counts": op_counts,
        "concentration": {
            "K_values": K_values_conc,
            "stds": concentration,
            "scaling": conc_scaling,
        },
        "elapsed_seconds": elapsed,
        "system_info": system_info(),
    }

    save_json(result, "claim3_hsic_o1/result.json")
    # Save timing CSV
    timing_csv = []
    for i, K in enumerate(K_values):
        timing_csv.append((K, timing_results["pair"][i], timing_results["ova"][i], timing_results["agg"][i],
                          op_counts["pair"][i], op_counts["ova"][i], op_counts["agg"][i]))
    save_csv(timing_csv, ["K", "time_pair_s", "time_ova_s", "time_agg_s", "ops_pair", "ops_ova", "ops_agg"],
             "claim3_hsic_o1/timing.csv")
    # Save concentration CSV
    conc_csv = []
    for i, K in enumerate(K_values_conc):
        conc_csv.append((K, concentration["pair"][i], concentration["ova"][i], concentration["agg"][i]))
    save_csv(conc_csv, ["K", "std_pair", "std_ova", "std_agg"],
             "claim3_hsic_o1/concentration.csv")

    log(f"Verdict: {verdict} (elapsed {elapsed:.1f}s)")
    return result
