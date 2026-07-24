"""Claim 5: K=20 Scalability — Pairwise PEHE > 1.3 at alpha=5.0, Aggregation PEHE ~ 1.0.

Claim (Section 4.2, Figure 1b):
  At K=20, Pairwise balancing becomes unstable (PEHE exceeding 1.3 under strong
  regularization alpha=5.0) while aggregation maintains competitive accuracy of
  approximately 1.0 across regularization levels.

Verification:
  A. Generate Hard Setting data with K=20 (Appendix D.1)
  B. Train CFR with pairwise, OVA, aggregation strategies across alpha sweep
  C. Measure PEHE for each (strategy, alpha) combination
  D. Verify: pairwise PEHE > 1.3 at alpha=5.0, agg PEHE ~ 1.0 across alpha
  E. Negative control: base model (alpha=0) PEHE for reference
"""
from __future__ import annotations
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data import generate_hard_setting
from src.model import CFRModel
from src.discrepancy import median_heuristic
from verifiers.common import save_json, save_csv, log, system_info


def run() -> dict:
    log("=== Claim 5: K=20 Scalability PEHE ===")
    t_start = time.perf_counter()

    K = 20
    N = 1500
    d = 20
    seed = 42

    # Generate data
    data = generate_hard_setting(N=N, K=K, d=d, seed=seed, kappa=5.0)
    X, T, Y = data["X"], data["T"], data["Y"]
    Y_all_mean = data["Y_all_mean"]
    sigma = median_heuristic(X)

    log(f"Data: N={N}, K={K}, d={d}, kappa=5.0")
    log(f"Treatment distribution: {np.bincount(T, minlength=K)}")

    # Alpha sweep
    alpha_grid = [0.0, 0.1, 0.5, 1.0, 5.0]
    strategies = ["base", "pair", "ova", "agg"]

    results = []
    pehe_table = {}

    for strategy in strategies:
        pehe_table[strategy] = []
        for alpha in alpha_grid:
            if strategy == "base":
                # Base model = no balancing (alpha=0, any strategy)
                model = CFRModel(
                    input_dim=d, K=K, repr_dim=8, strategy="pair",
                    alpha=0.0, epochs=150, sigma=sigma, seed=seed,
                )
            else:
                model = CFRModel(
                    input_dim=d, K=K, repr_dim=8, strategy=strategy,
                    alpha=alpha, epochs=150, sigma=sigma, seed=seed,
                )

            log(f"  Training {strategy} alpha={alpha}...")
            t0 = time.perf_counter()
            model.fit(X, T, Y)
            train_time = time.perf_counter() - t0

            pehe = model.compute_pehe(X, Y_all_mean)
            pehe_table[strategy].append(float(pehe))
            results.append({
                "strategy": strategy,
                "alpha": alpha,
                "pehe": float(pehe),
                "train_time_s": float(train_time),
            })
            log(f"    PEHE = {pehe:.4f} (time {train_time:.1f}s)")

    # Verify specific claims
    # 1. Pairwise PEHE > 1.3 at alpha=5.0
    pair_alpha5_idx = alpha_grid.index(5.0)
    pair_pehe_5 = pehe_table["pair"][pair_alpha5_idx]
    pair_unstable = pair_pehe_5 > 1.3

    # 2. Aggregation PEHE ~ 1.0 across alpha levels
    agg_pehes = pehe_table["agg"]
    agg_stable = all(abs(p - 1.0) < 0.5 for p in agg_pehes) and max(agg_pehes) < 1.5

    # 3. Base model PEHE for reference (~0.796 per paper)
    base_pehe = pehe_table["base"][0]

    log(f"\nKey results:")
    log(f"  Base model PEHE: {base_pehe:.4f} (paper: ~0.796)")
    log(f"  Pairwise PEHE at alpha=5.0: {pair_pehe_5:.4f} (claim: > 1.3) -> {'PASS' if pair_unstable else 'FAIL'}")
    log(f"  Aggregation PEHE range: [{min(agg_pehes):.4f}, {max(agg_pehes):.4f}] (claim: ~1.0)")

    verified = pair_unstable and agg_stable
    verdict = "VERIFIED" if verified else "BLOCKED"

    elapsed = time.perf_counter() - t_start
    result = {
        "claim": "K=20 scalability: pairwise unstable, aggregation stable",
        "claim_text": "At K=20, pairwise PEHE > 1.3 at alpha=5.0; aggregation PEHE ~ 1.0 across alpha",
        "verdict": verdict,
        "K": K, "N": N, "d": d,
        "alpha_grid": alpha_grid,
        "pehe_table": pehe_table,
        "results": results,
        "pair_pehe_alpha5": pair_pehe_5,
        "pair_unstable": pair_unstable,
        "agg_stable": agg_stable,
        "base_pehe": base_pehe,
        "elapsed_seconds": elapsed,
        "system_info": system_info(),
    }

    save_json(result, "claim5_k20_pehe/result.json")
    csv_rows = []
    for r in results:
        csv_rows.append((r["strategy"], r["alpha"], r["pehe"], r["train_time_s"]))
    save_csv(csv_rows, ["strategy", "alpha", "pehe", "train_time_s"],
             "claim5_k20_pehe/pehe_results.csv")

    log(f"Verdict: {verdict} (elapsed {elapsed:.1f}s)")
    return result
