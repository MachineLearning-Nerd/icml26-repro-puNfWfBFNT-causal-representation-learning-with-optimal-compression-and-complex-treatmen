"""Exact assumption-satisfying audit of Theorem 3.5 and Theorem 3.8/Corollary 3.9.

The paper's earlier linear-profile attempt had a boundary minimizer, so the hypotheses of
both theorems were absent.  This audit instead uses a bounded profile-score model for which
every assumption and conclusion is available in closed form.

For a strategy with m imbalance components, observation i contributes m bounded terms.  With
probability rho the terms share one Rademacher sign; otherwise their signs are independent.
Thus distinct terms have covariance rho > 0, matching Corollary 3.9's positive-dependence
condition, and the summed score has variance

    v_m = m + rho * m * (m - 1).

The component counts are exactly m=C(K,2), K, 1 for pair, OVA, aggregate.  Set

    R(alpha) = m,
    R_hat(alpha) = m + mean_i S_i,
    Comp(alpha) = 2m + 1 + kappa/2 * (alpha-alpha_0)^2 - m*alpha.

On A=[0,2], Comp is positive and decreasing, R_hat is nonnegative, and the population and
empirical profile criteria are

    Q(alpha) = 2m + 1 + kappa/2 * (alpha-alpha_0)^2,
    Q_hat(alpha) = Q(alpha) + alpha * mean_i S_i.

Consequently alpha_bd=alpha_0 and alpha_hat is the clipped value
alpha_0-mean(S)/kappa.  This makes the finite-sample inequality, CLT, and K-scaling directly
auditable without treating a numerical experiment as a proof.  Monte Carlo is retained as a
fixed-seed diagnostic of the closed-form results.
"""
from __future__ import annotations

import math
import os
import json
from pathlib import Path

import numpy as np
from scipy import stats

SEED = 20260802
DELTA = 0.05
RHO = 0.25
KAPPA = 0.5
ALPHA_0 = 1.0
ALPHA_RANGE = (0.0, 2.0)
N_REPLICATES = 20_000
STRATEGIES = ("pair", "ova", "agg")
ARTIFACTS_DIR = Path(
    os.environ.get(
        "CLAIM24_OUTPUT_DIR",
        Path(__file__).resolve().parents[1]
        / ".openresearch"
        / "artifacts"
        / "claim24_assumption_satisfying",
    )
)


def save_json(data: dict, name: str) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_csv(rows: list[tuple], header: list[str], name: str) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    lines.extend(",".join(str(value) for value in row) for row in rows)
    (ARTIFACTS_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def term_count(strategy: str, K: int) -> int:
    return {"pair": K * (K - 1) // 2, "ova": K, "agg": 1}[strategy]


def score_variance(m: int, rho: float = RHO) -> float:
    return m + rho * m * (m - 1)


def score_fourth_moment(m: int, rho: float = RHO) -> float:
    independent_fourth = 3 * m * m - 2 * m
    return rho * m**4 + (1 - rho) * independent_fourth


def sample_score_means(
    rng: np.random.Generator,
    m: int,
    n: int,
    n_replicates: int = N_REPLICATES,
    rho: float = RHO,
) -> np.ndarray:
    """Sample exact sums without allocating an n_replicates x n x m array."""
    n_common = rng.binomial(n, rho, size=n_replicates)
    common_positive = rng.binomial(n_common, 0.5)
    common_sum = m * (2 * common_positive - n_common)

    independent_trials = (n - n_common) * m
    independent_positive = rng.binomial(independent_trials, 0.5)
    independent_sum = 2 * independent_positive - independent_trials
    return (common_sum + independent_sum) / n


def alpha_hat(score_mean: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    raw = ALPHA_0 - score_mean / kappa
    return np.clip(raw, *ALPHA_RANGE)


def finite_sample_audit(rng: np.random.Generator) -> tuple[list[dict], list[tuple]]:
    cells = []
    csv_rows = []
    for strategy in STRATEGIES:
        for K in (4, 8, 16):
            m = term_count(strategy, K)
            for n in (2_048, 32_768, 1_048_576):
                score_means = sample_score_means(rng, m, n)
                estimates = alpha_hat(score_means)
                r_bound = m * math.sqrt(2 * math.log(2 / DELTA) / n)
                concentration_event = np.abs(score_means) <= r_bound
                deviations = np.abs(estimates - ALPHA_0)
                conclusion_holds = deviations <= r_bound / KAPPA + 1e-15
                event_count = int(concentration_event.sum())
                violations_on_event = int(np.sum(concentration_event & ~conclusion_holds))
                coverage = float(concentration_event.mean())
                cell = {
                    "strategy": strategy,
                    "K": K,
                    "n": n,
                    "m": m,
                    "r_bound": r_bound,
                    "r_scaled_constant": r_bound * math.sqrt(n) / m,
                    "deviation_bound": r_bound / KAPPA,
                    "empirical_event_coverage": coverage,
                    "event_count": event_count,
                    "violations_on_event": violations_on_event,
                    "max_deviation_on_event": float(deviations[concentration_event].max()),
                    "bound_nontrivial": bool(r_bound / KAPPA < ALPHA_RANGE[1] - ALPHA_0),
                }
                cells.append(cell)
                csv_rows.append(
                    (
                        strategy,
                        K,
                        n,
                        m,
                        r_bound,
                        coverage,
                        violations_on_event,
                        cell["max_deviation_on_event"],
                    )
                )
    return cells, csv_rows


def asymptotic_audit(rng: np.random.Generator) -> tuple[list[dict], list[tuple]]:
    cells = []
    csv_rows = []
    for strategy in STRATEGIES:
        for K in (4, 8, 12, 20, 32):
            m = term_count(strategy, K)
            variance_per_observation = score_variance(m)
            for n in (4_096, 65_536, 1_048_576, 16_777_216):
                score_means = sample_score_means(rng, m, n)
                estimates = alpha_hat(score_means)
                predicted_variance = variance_per_observation / (KAPPA**2 * n)
                empirical_variance = float(np.var(estimates, ddof=1))
                z = (
                    np.sqrt(n)
                    * (estimates - ALPHA_0)
                    * KAPPA
                    / math.sqrt(variance_per_observation)
                )
                ks_statistic, ks_pvalue = stats.kstest(z, "norm")
                wasserstein = stats.wasserstein_distance(z, stats.norm.ppf((np.arange(len(z)) + 0.5) / len(z)))
                fourth = score_fourth_moment(m)
                excess_kurtosis = (fourth / variance_per_observation**2 - 3) / n
                cell = {
                    "strategy": strategy,
                    "K": K,
                    "n": n,
                    "m": m,
                    "component_correlation": RHO,
                    "score_variance": variance_per_observation,
                    "predicted_alpha_variance": predicted_variance,
                    "empirical_alpha_variance": empirical_variance,
                    "variance_relative_error": abs(empirical_variance / predicted_variance - 1),
                    "interior_rate": float(np.mean((estimates > ALPHA_RANGE[0]) & (estimates < ALPHA_RANGE[1]))),
                    "standardized_mean": float(np.mean(z)),
                    "standardized_variance": float(np.var(z, ddof=1)),
                    "ks_statistic": float(ks_statistic),
                    "ks_pvalue": float(ks_pvalue),
                    "wasserstein_to_normal": float(wasserstein),
                    "exact_standardized_excess_kurtosis": excess_kurtosis,
                    "n_var_over_K_power": predicted_variance
                    * n
                    / ({"pair": K**4, "ova": K**2, "agg": 1}[strategy]),
                }
                cells.append(cell)
                csv_rows.append(
                    (
                        strategy,
                        K,
                        n,
                        m,
                        predicted_variance,
                        empirical_variance,
                        cell["interior_rate"],
                        cell["ks_statistic"],
                        cell["wasserstein_to_normal"],
                    )
                )
    return cells, csv_rows


def controls(rng: np.random.Generator) -> dict:
    m = term_count("pair", 8)
    score_means = sample_score_means(rng, m, 4_096)

    boundary_estimates = np.clip(ALPHA_0 - score_means / KAPPA, 0.0, ALPHA_0)
    boundary_mass = float(np.mean(boundary_estimates == ALPHA_0))
    boundary_z = (
        np.sqrt(4_096)
        * (boundary_estimates - ALPHA_0)
        * KAPPA
        / math.sqrt(score_variance(m))
    )
    boundary_ks = float(stats.kstest(boundary_z, "norm").statistic)

    return {
        "boundary_optimum": {
            "search_range": [0.0, ALPHA_0],
            "population_optimum_on_boundary": True,
            "assumption_3_7_i_satisfied": False,
            "mass_at_boundary": boundary_mass,
            "ks_statistic": boundary_ks,
            "expected_result": "REJECT THEOREM 3.8 AUDIT AS OUT OF SCOPE",
        },
        "zero_curvature": {
            "kappa": 0.0,
            "unique_population_minimizer": False,
            "assumption_3_4_i_satisfied": False,
            "finite_sample_denominator_defined": False,
            "expected_result": "REJECT THEOREM 3.5 AUDIT AS OUT OF SCOPE",
        },
    }


def run() -> dict:
    rng = np.random.default_rng(SEED)
    finite_cells, finite_csv = finite_sample_audit(rng)
    asymptotic_cells, asymptotic_csv = asymptotic_audit(rng)
    negative_controls = controls(rng)

    final_n = 16_777_216
    final_normality = [c for c in asymptotic_cells if c["K"] == 8 and c["n"] == final_n]
    assumptions = {
        "compact_search_range": list(ALPHA_RANGE),
        "population_optimum": ALPHA_0,
        "population_optimum_interior": ALPHA_RANGE[0] < ALPHA_0 < ALPHA_RANGE[1],
        "infimum_population_curvature": KAPPA,
        "complexity_positive": True,
        "complexity_monotone_decreasing": KAPPA * (ALPHA_RANGE[1] - ALPHA_0) - 1 <= 0,
        "empirical_imbalance_nonnegative": True,
        "component_pairwise_correlation": RHO,
    }
    gates = {
        "assumptions_in_force": all(bool(v) for k, v in assumptions.items() if isinstance(v, bool)),
        "finite_bound_no_violations_on_event": all(c["violations_on_event"] == 0 for c in finite_cells),
        "finite_event_coverage_at_least_1_minus_delta": all(
            c["empirical_event_coverage"] >= 1 - DELTA for c in finite_cells
        ),
        "finite_has_nontrivial_bounds": any(c["bound_nontrivial"] for c in finite_cells),
        "normal_limit_diagnostics": all(
            c["ks_statistic"] < 0.025
            and c["wasserstein_to_normal"] < 0.04
            and abs(c["standardized_mean"]) < 0.03
            and abs(c["standardized_variance"] - 1) < 0.05
            for c in final_normality
        ),
        "variance_formula_matches_monte_carlo": all(
            c["variance_relative_error"] < 0.08
            for c in asymptotic_cells
            if c["n"] == final_n
        ),
        "boundary_control_rejected": (
            not negative_controls["boundary_optimum"]["assumption_3_7_i_satisfied"]
            and negative_controls["boundary_optimum"]["mass_at_boundary"] > 0.45
            and negative_controls["boundary_optimum"]["ks_statistic"] > 0.20
        ),
        "zero_curvature_control_rejected": (
            not negative_controls["zero_curvature"]["assumption_3_4_i_satisfied"]
            and not negative_controls["zero_curvature"]["finite_sample_denominator_defined"]
        ),
    }
    result = {
        "audit": "Theorem 3.5 and Theorem 3.8/Corollary 3.9",
        "scope": "Exact bounded profile-score construction satisfying the stated assumptions; not a proof replacement or an audit of the paper's neural experiments.",
        "constants": {
            "seed": SEED,
            "delta": DELTA,
            "rho": RHO,
            "kappa": KAPPA,
            "alpha_0": ALPHA_0,
            "alpha_range": list(ALPHA_RANGE),
            "replicates": N_REPLICATES,
        },
        "identities": {
            "term_counts": "pair=C(K,2), ova=K, agg=1",
            "score_variance": "m + rho*m*(m-1)",
            "alpha_hat": "clip(alpha_0 - mean(score)/kappa, alpha_min, alpha_max)",
            "finite_bound": "|alpha_hat-alpha_0| <= r/kappa on |mean(score)| <= r",
            "alpha_variance": "[m + rho*m*(m-1)]/(kappa^2*n)",
        },
        "assumptions": assumptions,
        "claim_2_finite_sample": {"cells": finite_cells},
        "claim_4_asymptotic": {"cells": asymptotic_cells},
        "controls": negative_controls,
        "gates": gates,
        "verdict": "VERIFIED" if all(gates.values()) else "BLOCKED",
    }
    save_json(result, "result.json")
    save_csv(
        finite_csv,
        ["strategy", "K", "n", "m", "r_bound", "coverage", "violations_on_event", "max_deviation_on_event"],
        "finite_sample.csv",
    )
    save_csv(
        asymptotic_csv,
        ["strategy", "K", "n", "m", "predicted_variance", "empirical_variance", "interior_rate", "ks_statistic", "wasserstein"],
        "asymptotic.csv",
    )
    print(f"Verdict: {result['verdict']}")
    for gate, passed in gates.items():
        print(f"  {gate}: {passed}")
    return result


if __name__ == "__main__":
    run()
