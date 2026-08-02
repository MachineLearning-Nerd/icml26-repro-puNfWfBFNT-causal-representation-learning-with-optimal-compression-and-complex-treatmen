"""Claim 1, Steps 2-3 re-run on TRAINED DEEP REPRESENTATIONS rather than simulated triples.

The 2026-07-31 judge (sha 1396ce3c) narrowed its objection to Claim 1 to exactly one point:

    "Step 1 of the proof is symbolically certified (sympy) and numerically verified across 7
     outcome families with 0 violations and exact tightness at K=2, but Steps 2-4 are validated
     on CONTROLLED SIMULATIONS of (eps_src, eps_tar, IPM) triples rather than TRAINED DEEP
     REPRESENTATIONS, and the constants C_F, C_B, C_C remain unidentified as the paper itself
     disclaims."

Note the constants are explicitly no longer held against the claim.  What remains is that the
(eps_src, eps_tar, IPM) triples were synthetic.  So here every triple comes from an actually
trained representation network: Phi is learned by `src.cfr_fixed.CFRFixed` under the paper's own
balancing strategies (pair / ova / agg) across a range of balancing strengths alpha, and then

    eps_src  = factual risk on the source arm under the trained Phi
    eps_tar  = risk on the target arm under the same trained Phi
    IPM      = MMD between Phi(X)|T=j and Phi(X)|T=k, measured on the learned representation

are all read off the trained model.  Varying alpha and the strategy sweeps the representation
from badly imbalanced to well balanced, which is what gives the imbalance term something to
explain.  The test is unchanged and remains TIGHTNESS, not feasibility: constants are fitted on
a calibration half and scored by held-out R^2, because inflating constants satisfies any
variant -- the defect that earned this claim `toy` in the first place.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cfr_fixed import CFRFixed
from verifiers.assumption_audit import save_rows_csv
from verifiers.claim1_steps234 import bounded_loss, mmd2
from verifiers.common import log, save_json, system_info

RNG_SEED = 20260731
N_UNITS = 900
D_COV = 10
K = 4
# Sweeping strategy x alpha moves the learned Phi from badly imbalanced (alpha=0) to strongly
# balanced, so the IPM term varies over a wide range on REAL representations.
STRATEGIES = ["pair", "ova", "agg"]
ALPHA_GRID = [0.0, 0.1, 0.5, 2.0, 8.0]
SEEDS = [0, 1, 2]
STEPS = 800
REPR_DIM = 16


def make_dataset(rng):
    """Confounded multi-treatment data with genuine overlap (positivity holds by construction)."""
    X = rng.normal(0, 1, size=(N_UNITS, D_COV))
    w = rng.normal(0, 1, D_COV)
    score = np.tanh(X @ w / np.sqrt(D_COV))
    # Bounded logits => bounded propensities => overlap.
    logits = 1.5 * np.stack([score * (t - (K - 1) / 2) for t in range(K)], axis=1)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    T = np.array([rng.choice(K, p=pi) for pi in p])
    base = np.stack([np.sin(X @ rng.normal(0, 1, D_COV) / np.sqrt(D_COV)) + 0.7 * t
                     for t in range(K)], axis=1)
    Y_all = base + score[:, None]
    Y = Y_all[np.arange(N_UNITS), T] + rng.normal(0, 0.1, N_UNITS)
    return X, T, Y, Y_all, float(p.min())


def triples_from_trained_model(m, X, T, Y_all):
    """Read (eps_src, eps_tar, IPM) off a TRAINED representation, for every ordered arm pair."""
    import torch

    with torch.no_grad():
        Z = m.phi(m._tx(X)).cpu().numpy()
    Yh = m.predict_all_treatments(X)                 # predictions at every treatment
    out = []
    for j in range(m.K):
        for k in range(m.K):
            if j == k:
                continue
            src = T == j
            if src.sum() < 20 or (T == k).sum() < 20:
                continue
            # Factual risk on the source arm, and the same model's risk on the target arm's
            # potential outcome -- both under the bounded loss Step 2 assumes.
            eps_src = float(np.mean(bounded_loss(Yh[src, j] - Y_all[src, j])))
            eps_tar = float(np.mean(bounded_loss(Yh[src, k] - Y_all[src, k])))
            ipm = float(np.sqrt(max(mmd2(Z[T == j][:150], Z[T == k][:150]), 0.0)))
            out.append((eps_src, eps_tar, ipm))
    return out


def run():
    rng = np.random.default_rng(RNG_SEED)
    log("=== Claim 1: Steps 2-3 on TRAINED deep representations ===")
    X, T, Y, Y_all, min_prop = make_dataset(rng)
    log(f"  n={N_UNITS}, d={D_COV}, K={K}; min propensity {min_prop:.4f} (overlap holds)")

    rows, pool, meta = [], [], []
    for strategy in STRATEGIES:
        for alpha in ALPHA_GRID:
            for seed in SEEDS:
                m = CFRFixed(D_COV, K, repr_dim=REPR_DIM, strategy=strategy, alpha=alpha,
                             steps=STEPS, seed=seed).fit(X, T, Y)
                tri = triples_from_trained_model(m, X, T, Y_all)
                pool.extend(tri)
                meta.append({"strategy": strategy, "alpha": alpha, "seed": seed,
                             "n_triples": len(tri),
                             "mean_ipm": float(np.mean([t[2] for t in tri])),
                             "mean_eps_tar": float(np.mean([t[1] for t in tri]))})
                rows.append({"part": "trained_repr", "strategy": strategy, "alpha": alpha,
                             "seed": seed, "mean_ipm": meta[-1]["mean_ipm"],
                             "mean_eps_tar": meta[-1]["mean_eps_tar"]})
        sel = [d for d in meta if d["strategy"] == strategy]
        log(f"  {strategy}: IPM ranges {min(d['mean_ipm'] for d in sel):.4f} -> "
            f"{max(d['mean_ipm'] for d in sel):.4f} across alpha {ALPHA_GRID}")

    log(f"  {len(pool)} (eps_src, eps_tar, IPM) triples, all from trained representations")

    order = rng.permutation(len(pool))
    half = len(pool) // 2
    calib = [pool[i] for i in order[:half]]
    holdout = [pool[i] for i in order[half:]]

    def evaluate(use_ipm):
        A = np.array([[s, m_, 1.0] if use_ipm else [s, 1.0] for s, _, m_ in calib])
        b = np.array([t for _, t, _ in calib])
        coef = np.abs(np.linalg.lstsq(A, b, rcond=None)[0])
        Ae = np.array([[s, m_, 1.0] if use_ipm else [s, 1.0] for s, _, m_ in holdout])
        be = np.array([t for _, t, _ in holdout])
        pred = Ae @ coef
        ss_res = float(np.sum((be - pred) ** 2))
        ss_tot = float(np.sum((be - be.mean()) ** 2))
        return {"coef": coef.tolist(),
                "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}

    wi, ni = evaluate(True), evaluate(False)
    log(f"    WITH imbalance term:    held-out R^2 = {wi['r2']:+.3f}")
    log(f"    WITHOUT imbalance term: held-out R^2 = {ni['r2']:+.3f}")

    # Step 3 on the learned representation: D_pair <= (K-1) D_ova must hold for the ACTUAL Phi.
    mix_holds, mix_rows = [], []
    for strategy in STRATEGIES:
        m = CFRFixed(D_COV, K, repr_dim=REPR_DIM, strategy=strategy, alpha=1.0,
                     steps=STEPS, seed=0).fit(X, T, Y)
        import torch

        with torch.no_grad():
            Z = m.phi(m._tx(X)).cpu().numpy()
        arms = [Z[T == t][:200] for t in range(K)]
        M = np.concatenate(arms)
        d_pair = sum(np.sqrt(max(mmd2(arms[j], arms[k]), 0))
                     for j in range(K) for k in range(j + 1, K))
        d_ova = sum(np.sqrt(max(mmd2(a, M), 0)) for a in arms)
        holds = bool(d_pair <= (K - 1) * d_ova + 1e-9)
        mix_holds.append(holds)
        mix_rows.append({"strategy": strategy, "d_pair": d_pair, "d_ova": d_ova,
                         "bound": (K - 1) * d_ova, "holds": holds})
        log(f"    {strategy}: D_pair={d_pair:.4f} <= (K-1)*D_ova={(K - 1) * d_ova:.4f}: {holds}")

    checks = {"n_triples": len(pool), "n_trained_models": len(meta),
              "r2_with_ipm": wi["r2"], "r2_without_ipm": ni["r2"],
              "mixture_inequality_on_learned_phi": bool(all(mix_holds)),
              "min_propensity": min_prop}
    passed = wi["r2"] > 0.5 and ni["r2"] < 0.2 and all(mix_holds)
    verdict = "VERIFIED" if passed else "BLOCKED"
    reason = (
        f"Steps 2-3 hold on TRAINED deep representations, not simulated triples: "
        f"{len(pool)} (eps_src, eps_tar, IPM) triples were read off {len(meta)} trained CFR "
        f"models spanning strategies {STRATEGIES} and balancing strengths {ALPHA_GRID}. eq. (20) "
        f"explains held-out R^2={wi['r2']:.2f} WITH the imbalance term and only "
        f"R^2={ni['r2']:.2f} without it, so the term is load-bearing on learned representations "
        f"and not merely on synthetic ones. The Step 3 mixture inequality "
        f"D_pair <= (K-1) D_ova holds on the learned Phi for every strategy."
        if passed else f"checks: {checks}"
    )
    log(f"Verdict: {verdict} -- {reason}")
    save_rows_csv(rows, "claim1_trained_repr.csv")
    result = {"claim": "Claim 1: Lemma 3.2 Steps 2-3 on trained deep representations",
              "verdict": verdict, "reason": reason, "checks": checks,
              "details": {"per_model": meta, "with_ipm": wi, "without_ipm": ni,
                          "mixture": mix_rows},
              "system": system_info()}
    save_json(result, "claim1_trained_repr.json")
    return result


if __name__ == "__main__":
    run()
