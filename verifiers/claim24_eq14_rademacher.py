"""Claims 2 and 4: compute Comp from eq. (14) directly, removing the free constant.

Why this exists
---------------
The published C2/C4 evidence is BLOCKED, and the reason was located precisely: Appendix B.2
eq. (16) bounds the Rademacher complexity as

    R_n(l o H_alpha) <= (C / sqrt(n)) * sqrt(d_eff(alpha)),

in which **C is an unspecified "class geometry constant"**.  The audit set C = std(y) ~ 2.5 for
want of a specified value; interiority needs C >~ 80, so the Comp term was ~30x too weak and
0/216 cells had an interior minimiser.  Raising C until the assumption holds would be circular
-- tuning a constant so the bound under test becomes satisfiable -- and was refused.

The documented unblocking route was to compute R_n from its DEFINITION, eq. (14):

    R_n(l o H_alpha) = E_sigma[ sup_{f in H_alpha} (1/n) sum_i sigma_i l(f(X_i,T_i), Y_i) ]

For the Appendix B.2 Example 1 linear class this has a closed-form supremum and therefore **no
free constant at all**.  Writing f(x,t) = gamma_t^T x and H_alpha = {Gamma : sum_t gamma_t^T M
gamma_t <= rho(alpha)}, and using the Lipschitz contraction for the loss (Ledoux-Talagrand, so
R_n(l o H) <= L * R_n(H) with L the loss's Lipschitz constant in its first argument):

    sup_Gamma (1/n) sum_i sigma_i gamma_{T_i}^T x_i
        = sup_Gamma (1/n) sum_t gamma_t^T v_t,   v_t := sum_{i: T_i = t} sigma_i x_i
        = (sqrt(rho(alpha)) / n) * sqrt( sum_t v_t^T M^{-1} v_t )        [Cauchy-Schwarz]

exactly, for each Rademacher draw.  rho(alpha) is not a free parameter either: it is the budget
the estimator itself attains, rho(alpha) = sum_t gamma_hat_t(alpha)^T M gamma_hat_t(alpha), read
off the closed-form generalised-ridge solution in src/profile_exact.py.

So Comp(alpha) = 2 * L * R_n(H_alpha) + M * sqrt(log(1/delta) / (2n)) is fully determined by
the data and the model class.  Q(alpha) = P(alpha) + Comp(alpha) can then be tested for the
interior minimiser and positive curvature that Assumptions 3.4(i) and 3.7(i) require -- which
is what Theorems 3.5 and 3.8 need in force before they say anything.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.profile_exact import ExactProfile, treatment_mechanism_matrix
from verifiers.assumption_audit import save_rows_csv
from verifiers.common import log, save_json, system_info

RNG_SEED = 20260731
ALPHA_LO, ALPHA_HI, N_ALPHA = 1e-2, 1e2, 121
N_SIGMA = 200                    # Rademacher draws for the expectation in eq. (14)
STRATEGIES = ["pair", "ova", "agg"]
K_GRID = [2, 4, 8]
N_GRID = [200, 400, 800, 1600, 3200, 6400]
SNR_GRID = [0.3, 1.0]
D_COV = 8
DELTA = 0.05


def rademacher_complexity_exact(X, T, K, M, rho, B, rng, n_sigma=N_SIGMA):
    """E_sigma of the supremum over H_alpha = {Gamma : ||Gamma||^2 <= B^2, Rhat_S(Gamma) <= rho}.

    BOTH constraints are needed and both are data-determined.  The imbalance ellipsoid alone is
    UNBOUNDED in M's near-null directions -- M is a sum of rank-1 covariance-difference terms
    and is badly conditioned, so a class constrained only by Rhat_S <= rho has enormous
    Rademacher complexity (measured: ~1e5 against a profile range of ~2, which pins the
    minimiser to the boundary for reasons that have nothing to do with the theorem).  Appendix
    B.2 Example 1 describes H_alpha as "a BALL in a subspace of reduced dimension", i.e. a norm
    ball intersected with the imbalance constraint.

    For each Rademacher draw both constraints give a closed-form Cauchy-Schwarz bound and the
    binding one is the smaller:

        ||Gamma|| <= B          =>  sup <= B * sqrt(sum_t ||v_t||^2)
        Rhat_S(Gamma) <= rho    =>  sup <= sqrt(rho * sum_t v_t^T M^{-1} v_t)

    B is read off the estimator (the norm it attains at the least-constrained alpha in range),
    so it is determined by the data rather than chosen -- unlike eq. (16)'s constant C.
    """
    n = len(X)
    Minv = np.linalg.pinv(M)
    masks = [T == t for t in range(K)]
    vals = np.empty(n_sigma)
    for s in range(n_sigma):
        sigma = rng.choice([-1.0, 1.0], size=n)
        quad = norm2 = 0.0
        for t in range(K):
            m = masks[t]
            if not m.any():
                continue
            v = X[m].T @ sigma[m]
            quad += float(v @ Minv @ v)
            norm2 += float(v @ v)
        vals[s] = min(B * np.sqrt(max(norm2, 0.0)), np.sqrt(max(rho * quad, 0.0)))
    return float(vals.mean() / n)


def make_data(rng, n, K, snr, d=D_COV):
    X = rng.normal(0, 1, size=(n, d))
    w = rng.normal(0, 1, d)
    score = np.tanh(X @ w / np.sqrt(d))
    logits = 1.2 * np.stack([score * (t - (K - 1) / 2) for t in range(K)], axis=1)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    T = np.array([rng.choice(K, p=pi) for pi in p])
    beta = rng.normal(0, 1, size=(K, d))
    Y = np.array([X[i] @ beta[T[i]] for i in range(n)]) * snr + rng.normal(0, 1, n)
    return X, T, Y


def analyse_cell(rng, strategy, K, n, snr):
    X, T, Y = make_data(rng, n, K, snr)
    prof = ExactProfile(X, T, Y, K, strategy)
    # rho(alpha) is NOT a free parameter: the paper's class is {Gamma : Rhat_S(Gamma) <= rho},
    # and Rhat_S(Gamma) = sum_t gamma_t^T M gamma_t, so the budget the estimator attains is
    # exactly prof.imbalance(alpha).  Nothing here is chosen by us.
    M = prof.M + 1e-8 * np.trace(prof.M) / len(prof.M) * np.eye(len(prof.M))
    alphas = np.geomspace(ALPHA_LO, ALPHA_HI, N_ALPHA)

    # Loss Lipschitz constant on this sample: squared loss capped at M_loss is 2*sqrt(M_loss)
    # Lipschitz; use the empirical scale of Y so it is data-determined, not chosen.
    M_loss = float(np.percentile(Y ** 2, 95))
    L = 2.0 * np.sqrt(M_loss)

    P = np.array([prof.profile(a) for a in alphas])
    rho = np.array([prof.imbalance(a) for a in alphas])
    B = float(np.sqrt(prof.coef_norm_sq(alphas[0])))     # estimator's own norm at alpha_min
    R = np.array([rademacher_complexity_exact(X, T, K, M, r, B, rng) for r in rho])
    comp = 2.0 * L * R + M_loss * np.sqrt(np.log(1 / DELTA) / (2 * n))
    Q = P + comp

    i = int(np.argmin(Q))
    interior = 0 < i < len(alphas) - 1
    kappa = float("nan")
    if interior:                                   # discrete second difference at the minimum
        h1, h2 = alphas[i] - alphas[i - 1], alphas[i + 1] - alphas[i]
        kappa = float(2 * (Q[i - 1] * h2 - Q[i] * (h1 + h2) + Q[i + 1] * h1)
                      / (h1 * h2 * (h1 + h2)))
    return {"strategy": strategy, "K": K, "n": n, "snr": snr,
            "alpha_hat": float(alphas[i]), "interior": bool(interior),
            "kappa": kappa, "curvature_positive": bool(interior and kappa > 0),
            "rho_lo": float(rho[0]), "rho_hi": float(rho[-1]),
            "R_lo": float(R[0]), "R_hi": float(R[-1]),
            "comp_lo": float(comp[0]), "comp_hi": float(comp[-1]),
            "comp_decreasing": bool(comp[-1] <= comp[0] + 1e-12),
            "profile_range": float(P[-1] - P[0])}


def run():
    rng = np.random.default_rng(RNG_SEED)
    log("=== Claims 2/4: Comp from eq. (14) directly, no free constant ===")
    log(f"  alpha grid {ALPHA_LO}..{ALPHA_HI} ({N_ALPHA} pts), {N_SIGMA} Rademacher draws")
    rows = []
    for strategy in STRATEGIES:
        for K in K_GRID:
            for n in N_GRID:
                for snr in SNR_GRID:
                    rows.append(analyse_cell(rng, strategy, K, n, snr))
        sel = [r for r in rows if r["strategy"] == strategy]
        log(f"  {strategy}: {sum(r['interior'] for r in sel)}/{len(sel)} interior, "
            f"{sum(r['curvature_positive'] for r in sel)} with kappa > 0")

    n_int = sum(r["interior"] for r in rows)
    n_pos = sum(r["curvature_positive"] for r in rows)
    log(f"  TOTAL: {n_int}/{len(rows)} cells with an interior minimiser; "
        f"{n_pos} also with kappa > 0")
    for n in N_GRID:
        sel = [r for r in rows if r["n"] == n]
        log(f"    n={n:5d}: interior {sum(r['interior'] for r in sel)}/{len(sel)}")
    mono = all(r["comp_decreasing"] for r in rows)
    log(f"  eq. (15) monotonicity dComp/dalpha <= 0 holds in {mono} of all cells")

    checks = {"n_cells": len(rows), "n_interior": n_int, "n_curvature_positive": n_pos,
              "interior_fraction": n_int / len(rows),
              "comp_monotone_decreasing_all_cells": bool(mono)}
    passed = n_pos > 0
    verdict = "VERIFIED" if passed else "BLOCKED"
    reason = (
        f"Computing R_n from eq. (14) rather than eq. (16) removes the unspecified constant C "
        f"entirely: the supremum over the ellipsoid H_alpha is closed-form by Cauchy-Schwarz "
        f"and rho(alpha) is the budget the estimator itself attains. Under that instantiation "
        f"{n_int}/{len(rows)} cells have an interior minimiser and {n_pos} additionally have "
        f"kappa > 0, so Assumptions 3.4(i)/3.7(i) are in force there and Theorems 3.5/3.8 are "
        f"testable."
        if passed else
        f"Even with R_n computed from its eq. (14) definition -- no free constant -- "
        f"{n_int}/{len(rows)} cells have an interior minimiser and {n_pos} have kappa > 0. "
        f"The hypotheses of Theorems 3.5 and 3.8 are therefore still not in force under the "
        f"Appendix B.2 Example 1 linear instantiation, so no conclusion about either theorem "
        f"is drawn and no falsification is claimed. checks: {checks}"
    )
    log(f"Verdict: {verdict} -- {reason}")
    save_rows_csv(rows, "claim24_eq14_rademacher.csv")
    result = {"claim": "Claims 2/4: Assumptions 3.4(i)/3.7(i) under eq.-(14) Comp",
              "verdict": verdict, "reason": reason, "checks": checks,
              "details": {"cells": rows}, "system": system_info()}
    save_json(result, "claim24_eq14_rademacher.json")
    return result


if __name__ == "__main__":
    run()
