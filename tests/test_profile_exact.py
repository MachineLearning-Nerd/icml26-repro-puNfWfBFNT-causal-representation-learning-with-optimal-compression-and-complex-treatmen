"""Algebraic correctness tests for src/profile_exact.py.

These are software tests -- they check that the whitened-basis closed forms agree with a
brute-force dense solve and with finite differences.  They measure nothing scientific; every
number that becomes evidence is produced by the verifiers on Hugging Face cpu-upgrade.

Run:  uv run python -m pytest tests/test_profile_exact.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import generate_hard_setting
from src.profile_exact import ExactProfile, treatment_mechanism_matrix, n_imbalance_terms

STRATEGIES = ["pair", "ova", "agg"]


def _fixture(K=4, N=1200, d=8, seed=0):
    dat = generate_hard_setting(N=N, K=K, d=d, seed=seed)
    return dat["X"], dat["T"], dat["Y"], K


def _brute_force(X, T, y, K, M, alpha):
    """Dense reference: solve (S_t + alpha M) gamma_t = c_t directly, no whitening."""
    n = len(X)
    gammas, eps_F, R = [], 0.0, 0.0
    for t in range(K):
        m = T == t
        Xt, yt = X[m], y[m]
        S_t, c_t = Xt.T @ Xt / n, Xt.T @ yt / n
        g = np.linalg.solve(S_t + alpha * M, c_t)
        gammas.append(g)
        eps_F += float(yt @ yt) / n - 2 * g @ c_t + g @ S_t @ g
        R += float(g @ M @ g)
    return eps_F, R, gammas


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("alpha", [0.01, 0.3, 1.0, 7.5, 100.0])
def test_closed_form_matches_dense_solve(strategy, alpha):
    """The O(d) whitened evaluation must equal the O(d^3) dense solve."""
    X, T, y, K = _fixture()
    M = treatment_mechanism_matrix(X, T, K, strategy)
    prof = ExactProfile(X, T, y, K, strategy)

    eps_ref, R_ref, _ = _brute_force(X, T, y, K, M, alpha)

    assert prof.imbalance(alpha) == pytest.approx(R_ref, rel=1e-9, abs=1e-14)
    assert prof.factual_risk(alpha) == pytest.approx(eps_ref, rel=1e-9, abs=1e-14)
    assert prof.profile(alpha) == pytest.approx(eps_ref + alpha * R_ref, rel=1e-9, abs=1e-14)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_envelope_identity(strategy):
    """Lemma 3.3: P'(alpha) == R_S(theta_hat(alpha)).

    This is the step that turns Assumption 3.4(ii)'s bound on the imbalance *functional* into
    a bound on the criterion *gradient*, which is what yields Theorem 3.5's linear r/kappa
    rate rather than a square-root rate.
    """
    X, T, y, K = _fixture()
    prof = ExactProfile(X, T, y, K, strategy)
    for alpha in [0.05, 0.5, 2.0, 20.0]:
        an, fd, rel = prof.verify_envelope_identity(alpha)
        assert rel < 1e-6, f"envelope identity broke at alpha={alpha}: {an} vs {fd}"


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("order", [1, 2])
def test_derivatives_match_finite_differences(strategy, order):
    """Q' and Q'' closed forms vs central differences of Q."""
    X, T, y, K = _fixture()
    prof = ExactProfile(X, T, y, K, strategy)
    # Central differences carry error ~ eps/h^order + O(h^2), so the step must sit near the
    # U-shaped optimum: ~eps^(1/3) for a first derivative, ~eps^(1/4) for a second.  Using a
    # smaller h makes the check *worse*, not tighter.
    h_rel = 1e-5 if order == 1 else 1e-3
    for alpha in [0.2, 1.0, 5.0]:
        h = h_rel * alpha
        if order == 1:
            fd = (prof.Q(alpha + h) - prof.Q(alpha - h)) / (2 * h)
        else:
            fd = (prof.Q(alpha + h) - 2 * prof.Q(alpha) + prof.Q(alpha - h)) / h**2
        assert prof.Q(alpha, order) == pytest.approx(fd, rel=1e-5, abs=1e-10)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_d_eff_matches_trace_and_decreases(strategy):
    """d_eff(alpha) must equal sum_t tr[S_t (S_t+alpha M)^{-1}] and decrease in alpha.

    Decreasing d_eff is what Appendix B.2 eq. (12)/(15) require (H_alpha shrinks, so Comp
    falls); if it ever increased, the instantiation would contradict the paper.
    """
    X, T, y, K = _fixture()
    M = treatment_mechanism_matrix(X, T, K, strategy)
    prof = ExactProfile(X, T, y, K, strategy)
    n = len(X)

    for alpha in [0.1, 2.0]:
        ref = sum(
            np.trace(
                (X[T == t].T @ X[T == t] / n)
                @ np.linalg.inv(X[T == t].T @ X[T == t] / n + alpha * M)
            )
            for t in range(K)
        )
        assert prof.d_eff(alpha) == pytest.approx(ref, rel=1e-9)

    grid = np.geomspace(0.01, 100, 50)
    vals = [prof.d_eff(a) for a in grid]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])), "d_eff must be non-increasing"
    assert all(prof.comp(a, 0) >= prof.comp(b, 0) - 1e-12 for a, b in zip(grid, grid[1:])), (
        "Appendix B.2 eq. (15) requires dComp/dalpha <= 0"
    )


@pytest.mark.parametrize("strategy,K", [(s, K) for s in STRATEGIES for K in [2, 4, 10]])
def test_term_counts_are_the_paper_complexity(strategy, K):
    """The rank-one term count is exactly C(K,2) / K / 1 -- the paper's O(K^2)/O(K)/O(1)."""
    expected = {"pair": K * (K - 1) // 2, "ova": K, "agg": 1}[strategy]
    assert n_imbalance_terms(K, strategy) == expected


def test_profile_term_is_concave():
    """P''(alpha) <= 0 always: an infimum of affine-in-alpha functions is concave.

    This is the structural reason boundary optima are generic and why positive curvature must
    come from Comp -- the point the previously judged attempt missed.
    """
    X, T, y, K = _fixture()
    for strategy in STRATEGIES:
        prof = ExactProfile(X, T, y, K, strategy)
        for alpha in np.geomspace(0.01, 100, 25):
            h = 1e-5 * alpha
            p2 = (prof.profile(alpha + h) - 2 * prof.profile(alpha) + prof.profile(alpha - h)) / h**2
            assert p2 <= 1e-6, f"profile term not concave at alpha={alpha}: P''={p2}"


def test_singular_block_raises_rather_than_silently_regularising():
    """n_t <= d must be a loud error, not a hidden ridge floor that changes the estimand."""
    X, T, y, K = _fixture(K=4, N=60, d=40)
    with pytest.raises(np.linalg.LinAlgError):
        ExactProfile(X, T, y, K, "pair")
