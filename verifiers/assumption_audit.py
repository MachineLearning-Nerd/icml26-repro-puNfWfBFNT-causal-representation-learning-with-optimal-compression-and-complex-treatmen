"""Assumption audit for Theorem 3.5 and Theorem 3.8 / Corollary 3.9.

Both theorems are conditional statements.  Theorem 3.5 needs Assumption 3.4:
  (i)  inf_{alpha in A} Q''_S(alpha) >= kappa_S > 0   (strong convexity in alpha)
  (ii) sup_{alpha in A} |R_hat_S(theta_hat(a)) - R_S(theta*(a))| <= r_S(n,delta,K)
Theorem 3.8 needs Assumption 3.7(i): alpha^bd_S(n) -> alpha^inf strictly INSIDE (a_min, a_max),
with Q''_S(alpha^inf) > 0.

The previously judged run measured a criterion with boundary optima and reported the
theorems as unverified.  A boundary optimum means Assumption 3.4(i)/3.7(i) fail, so those
measurements were taken outside the theorems' hypotheses.  That is out-of-scope evidence,
NOT a falsification, and this audit exists to decide -- before any conclusion is tested --
which (n, K, SNR, strategy) cells are actually in scope.

Structural expectation being tested
-----------------------------------
Q_S(alpha) = P(alpha) + Comp_S(alpha; n, delta), where P is an infimum of functions affine in
alpha and is therefore concave (P'' <= 0), while Appendix B.2 eq. (13) makes Comp = O(n^{-1/2}).
So Q'' = P'' + Comp'' has an O(1) negative part and an O(n^{-1/2}) positive part: interiority
should HOLD at small n / low SNR and FAIL as n grows.  If that is what the data show, then
Theorem 3.5 is testable in the in-scope cells, while Theorem 3.8's n->infinity limit is in
tension with its own interiority requirement under this instantiation.  We report whichever
way it comes out.

Outputs (written under .openresearch/artifacts/):
  assumption_audit.csv   one row per (strategy, K, n, snr) cell
  assumption_audit.json  summary + in-scope cell list consumed by the C2/C4 verifiers
"""
from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import generate_hard_setting
from src.profile_exact import ALPHA_RANGE, N_ALPHA_GRID, ExactProfile
from verifiers.common import log, save_json, system_info


def save_rows_csv(rows, filename):
    """Write dict-rows to CSV with the union of keys as header.

    common.save_csv() takes positional (rows, header, filename) and is part of the frozen
    Claim 3 evidence path, so it is left untouched.
    """
    from verifiers.common import ARTIFACTS_DIR, ensure_dir

    header, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                header.append(k)
    ensure_dir(ARTIFACTS_DIR)
    path = os.path.join(ARTIFACTS_DIR, filename)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in header) + "\n")
    return path

# --- a-priori fixed design -------------------------------------------------------------
# Chosen before any result was seen and NOT derived from the bound under test.  The n-grid
# spans a 32x range so an n-dependence of interiority is visible; SNR is varied because the
# structural argument above says the concave/convex balance depends on signal scale.
N_GRID = [200, 400, 800, 1600, 3200, 6400]
K_GRID = [2, 4, 8]
SNR_GRID = [0.1, 0.3, 1.0, 3.0]
STRATEGIES = ["pair", "ova", "agg"]
D_COV = 8              # covariate dimension; kept < min_t n_t so every S_t is PD
DELTA = 0.05
N_POP = 400_000        # sample size standing in for the population profile
SEED = 20260731


def _dataset(N, K, snr, seed, d=D_COV):
    """Appendix D.1 'Hard Setting', with the treatment-effect scale multiplied by snr.

    snr rescales only the treatment-effect component, leaving the confounding baseline and
    noise fixed, so it moves signal strength without changing the overlap violation that
    makes the setting hard.
    """
    dat = generate_hard_setting(N=N, K=K, d=d, seed=seed)
    base = np.sin(2.0 * dat["X"][:, 0]) + dat["X"][:, 2] ** 2
    eff = dat["Y_all_mean"] - base[:, None]
    dat["Y_all_mean"] = base[:, None] + snr * eff
    rng = np.random.default_rng(seed + 7)
    dat["Y_all"] = dat["Y_all_mean"] + rng.standard_normal(dat["Y_all_mean"].shape) * np.sqrt(0.1)
    dat["Y"] = dat["Y_all"][np.arange(N), dat["T"]]
    return dat


@lru_cache(maxsize=None)
def _population_profile_cached(K, snr, strategy, seed=SEED + 1):
    """The population profile term only depends on (K, snr, strategy) -- build it once.

    n enters solely through Comp_S(alpha; n, delta), which population_profile() sets per call.
    """
    dat = _dataset(N_POP, K, snr, seed)
    return ExactProfile(dat["X"], dat["T"], dat["Y"], K, strategy, delta=DELTA)


def population_profile(K, snr, strategy, n_for_comp):
    """Q_S(alpha): population profile term + Comp_S(alpha; n, delta) at the SAME n.

    Eq. (9) keeps Comp at the finite n, so only the inf_theta term is taken at population
    scale.  Getting this wrong would shift alpha^bd(n) and silently corrupt Theorem 3.5.
    """
    prof = _population_profile_cached(K, snr, strategy)
    prof.n = n_for_comp  # Comp is evaluated at the finite n of eq. (9)
    return prof


def audit_cell(strategy, K, n, snr, seed=SEED):
    dat = _dataset(n, K, snr, seed)
    counts = np.bincount(dat["T"], minlength=K)
    if counts.min() <= D_COV:
        return {
            "strategy": strategy, "K": K, "n": n, "snr": snr,
            "status": "SKIPPED_RANK", "reason": f"min group {counts.min()} <= d={D_COV}",
        }

    prof = ExactProfile(dat["X"], dat["T"], dat["Y"], K, strategy, delta=DELTA)
    grid = np.geomspace(*ALPHA_RANGE, N_ALPHA_GRID)

    a_hat, hat_boundary = prof.argmin(grid)
    q2 = np.array([prof.Q(a, 2) for a in grid])
    kappa_global = float(q2.min())
    kappa_at_hat = float(prof.Q(a_hat, 2))

    pop = population_profile(K, snr, strategy, n_for_comp=n)
    a_bd, bd_boundary = pop.argmin(grid)
    q2_pop = np.array([pop.Q(a, 2) for a in grid])

    # Assumption 3.4(ii): sup over A of |R_hat(theta_hat(a)) - R(theta*(a))|.
    r_emp = float(max(abs(prof.imbalance(a) - pop.imbalance(a)) for a in grid))

    # Envelope identity (Lemma 3.3) -- the step that makes 3.4(ii) a gradient bound.
    _, _, env_rel = prof.verify_envelope_identity(float(np.sqrt(a_hat * a_bd)))

    in_scope = (not hat_boundary) and (not bd_boundary) and kappa_global > 0
    return {
        "strategy": strategy, "K": K, "n": n, "snr": snr,
        "status": "IN_SCOPE" if in_scope else "OUT_OF_SCOPE",
        "alpha_hat": a_hat, "alpha_bd": a_bd,
        "hat_on_boundary": bool(hat_boundary), "bd_on_boundary": bool(bd_boundary),
        "kappa_global": kappa_global, "kappa_at_alpha_hat": kappa_at_hat,
        "kappa_pop_global": float(q2_pop.min()),
        "frac_alpha_grid_convex": float((q2 > 0).mean()),
        "r_empirical": r_emp,
        "deviation": abs(a_hat - a_bd),
        "bound_rhs": (r_emp / kappa_global) if kappa_global > 0 else float("inf"),
        "envelope_rel_err": env_rel,
        "min_group_count": int(counts.min()),
    }


def run():
    log("=== Assumption audit for Theorems 3.5 / 3.8 (Assumptions 3.4, 3.7) ===")
    log(f"alpha range {ALPHA_RANGE} fixed a priori; {N_ALPHA_GRID} grid points")
    t0 = time.perf_counter()

    rows = []
    for strategy in STRATEGIES:
        for K in K_GRID:
            for n in N_GRID:
                for snr in SNR_GRID:
                    rows.append(audit_cell(strategy, K, n, snr))
        done = [r for r in rows if r["strategy"] == strategy]
        ok = sum(r["status"] == "IN_SCOPE" for r in done)
        log(f"  {strategy}: {ok}/{len(done)} cells in scope")

    scoped = [r for r in rows if r["status"] == "IN_SCOPE"]

    # Does interiority decay with n, as the concave-plus-O(n^-1/2) argument predicts?
    by_n = {}
    for n in N_GRID:
        cells = [r for r in rows if r["n"] == n and r["status"] != "SKIPPED_RANK"]
        by_n[n] = float(np.mean([r["status"] == "IN_SCOPE" for r in cells])) if cells else None

    summary = {
        "n_cells": len(rows),
        "n_in_scope": len(scoped),
        "in_scope_fraction_by_n": by_n,
        "in_scope_fraction_by_snr": {
            snr: float(np.mean([
                r["status"] == "IN_SCOPE"
                for r in rows if r["snr"] == snr and r["status"] != "SKIPPED_RANK"
            ]))
            for snr in SNR_GRID
        },
        "max_envelope_rel_err": max(
            (r.get("envelope_rel_err", 0.0) for r in rows if "envelope_rel_err" in r), default=None
        ),
        "alpha_range": list(ALPHA_RANGE),
        "runtime_s": time.perf_counter() - t0,
        "system": system_info(),
    }
    log(json.dumps(summary["in_scope_fraction_by_n"], indent=2))
    log(f"in scope: {len(scoped)}/{len(rows)} cells; runtime {summary['runtime_s']:.1f}s")

    save_rows_csv(rows, "assumption_audit.csv")
    save_json({"summary": summary, "cells": rows}, "assumption_audit.json")

    # This module is an assumption audit, not a claim verdict: it decides scope for the
    # Theorem 3.5 / 3.8 verifiers.  It reports AUDIT rather than PASS so it can never be
    # mistaken for evidence that a claim holds.
    return {
        "claim": "Assumption audit for Theorems 3.5 / 3.8",
        "verdict": "AUDIT",
        "summary": summary,
        "cells": rows,
    }


if __name__ == "__main__":
    run()
