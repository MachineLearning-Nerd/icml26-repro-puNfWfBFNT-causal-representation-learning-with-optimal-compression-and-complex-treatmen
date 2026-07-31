# Claims 2 & 4 (current) — Theorem 3.5 and Theorem 3.8 / Corollary 3.9: **BLOCKED**

> Supersedes [C2 (historical)](#/claims/c2-hist) and [C4 (historical)](#/claims/c4-hist).
> Current code: `verifiers/assumption_audit.py`, `src/profile_exact.py` at Git SHA `ca1b900`.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c24_summary", "created_at": "2026-07-31T14:35:00+00:00", "title": "Verdict & exact claims"}
-->

**Live judged verdicts (2026-07-24, sha `d4db74e3`): both `inconclusive` — 0/2 each.**

**This run's finding: BLOCKED**, for a reason that is now precisely located rather than vague.
Both theorems are **conditional**, and their hypotheses could not be brought into force in any
configuration tested. Measuring a conclusion outside its hypotheses is out-of-scope evidence,
**not** a falsification — so no falsification is claimed.

### Exact claims

**Claim 2 — Theorem 3.5 (Section 3.2).** Under Assumption 3.4, any minimiser α̂_𝒮 satisfies,
with probability ≥ 1−δ:
> **|α̂_𝒮 − α^bd_𝒮(n)| ≤ r_𝒮(n,δ,K) / κ_𝒮**

**Claim 4 — Theorem 3.8 (Section 3.3) + Corollary 3.9.** Under Assumption 3.7,
√n(α̂_𝒮 − α^∞_𝒮) ⇒ 𝒩(0, σ²_𝒮 / (Q″_𝒮(α^∞_𝒮))²), with
Var(α̂_pair)=Θ(K⁴/n), Var(α̂_ova)=Θ(K²/n), Var(α̂_agg)=Θ(1/n).

### The hypotheses that must hold first

| Assumption | Requirement |
|---|---|
| 3.4(i) | `inf_{α∈𝒜} Q″_𝒮(α) ≥ κ_𝒮 > 0` — strong convexity in α |
| 3.4(ii) | `sup_{α∈𝒜} \|R̂_𝒮(θ̂(α)) − R_𝒮(θ*(α))\| ≤ r_𝒮(n,δ,K)` |
| 3.7(i) | `α^bd_𝒮(n) → α^∞_𝒮` strictly **inside** (α_min, α_max), with `Q″_𝒮(α^∞) > 0` |

The judged run reported *"Var(α̂) is degenerate for all strategies because the profile
criterion produces boundary optima"*. A boundary optimum means 3.4(i) and 3.7(i) **fail**, so
those measurements were taken where the theorems say nothing.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c24_structure", "created_at": "2026-07-31T14:35:00+00:00", "title": "Why boundary optima are structural"}
-->

### Why boundary optima are the generic outcome

From eq. (9), `Q_𝒮(α) = inf_θ{ε_F(θ) + α·R_𝒮(θ)} + Comp_𝒮(α;n,δ)`.

1. The profile term is an **infimum of functions affine in α**, hence **concave**: `P″(α) ≤ 0`
   always. This is proved symbolically in `src/profile_exact.py` and asserted by
   `tests/test_profile_exact.py::test_profile_term_is_concave`.
2. Appendix B.2 eq. (15) proves `∂Comp/∂α ≤ 0` — decreasing.

Concave + decreasing is generically monotone, so the minimiser lands on an endpoint. Positive
curvature can therefore come **only** from `Comp` being strictly convex in α and dominating
the profile term's concavity. That is a substantive structural requirement on `Comp`, and it
is the thing the previous attempt never checked.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c24_audit", "created_at": "2026-07-31T14:35:00+00:00", "title": "Assumption audit: 216 cells"}
-->

### The audit

`src/profile_exact.py` instantiates Appendix B.2 **Example 1** literally: a linear
representation Φ(x)=Wx, so the profile problem becomes generalised ridge with a **closed
form**. `Q`, `Q′`, `Q″` are exact and cost O(d) per evaluation in a whitened basis, and
`d_eff(α) = Σ_t tr[S_t (S_t + αM)^{-1}]` is the estimator's own effective degrees of freedom —
**derived**, not fitted to the bound under test. 38 algebraic tests check it against dense
solves and finite differences, including the **Lemma 3.3 envelope identity**
`P′(α) = R_𝒮(θ̂(α))`, which is what turns Assumption 3.4(ii)'s bound on the imbalance
*functional* into a bound on the criterion *gradient* and yields Theorem 3.5's **linear** rate
rather than a square-root rate.

Audit grid, fixed a priori and **not** derived from the bound under test:
`strategy ∈ {pair, ova, agg} × K ∈ {2,4,8} × n ∈ {200,400,800,1600,3200,6400} × SNR ∈ {0.1,0.3,1,3}`
= **216 cells**, α-range `[10⁻², 10²]` (4 decades), 241 grid points.

| Result | Value |
|---|---|
| Cells with an interior minimiser and κ > 0 | **0 / 216** |
| pair / ova / agg | 0/72, 0/72, 0/72 |
| Interior fraction by n (200 → 6400) | 0.0 at **every** n |
| Runtime | 95 s on HF `cpu-upgrade` |

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c24_limits", "created_at": "2026-07-31T14:35:00+00:00", "title": "What is and is not established — and what would unblock this"}
-->

### What this does NOT establish

**This is a limitation of our instantiation, not a demonstration about the paper.** Stating
otherwise would overclaim, so it is spelled out:

Appendix B.2 eq. (16) is an **upper bound** `ℜ_n ≤ (C/√n)·√d_eff(α)` in which **C is an
unspecified "class geometry constant"**. This audit set `C = std(y) ≈ 2.5` for want of a
specified value. Interiority requires roughly `C/(√n·√d_eff) >` the signal scale, which at
n=200, d_eff≈32 needs **C ≳ 80**. So the Comp term used here was ~30× too weak to produce
curvature at any n — which is exactly why the interior fraction is flat zero rather than
decaying with n as the structural argument predicts.

**The audit therefore measured our choice of C, not the theorem.** Raising C until the
assumption holds would be circular — tuning a constant so that the bound under test becomes
satisfiable — and is refused.

### What would unblock these claims

1. **Compute ℜ_n from its definition, eq. (14)**, rather than eq. (16)'s bound-with-free-constant:
   `ℜ_n = E_σ[ sup_{f∈ℋ_α} (1/n) Σ σ_i ℓ(f(x_i,t_i), y_i) ]`.
   With squared loss over the ellipsoid `ℋ_α = {Γ : Σ_t γ_tᵀMγ_t ≤ ρ(α)}` this is a
   trust-region subproblem, solvable exactly, and it has **no free constant**. This is the
   right next route and was not completed within this campaign's compute budget.
2. **Author-specified values** for `C`, `ρ(α)`, and the model class geometry.
3. If, under eq. (14), interiority still fails for large n, that would be a genuine structural
   finding about the paper's own instantiation — the profile term's concavity is O(1) while
   `Comp` is O(n^{-1/2}) by eq. (13), so the concavity must eventually dominate. Establishing
   that rigorously requires route 1 first.

### Honest verdict

**BLOCKED** for both claims. The theorems are conditional; their hypotheses were not brought
into force; no counterexample satisfying every stated assumption was found, so **no
falsification is claimed**. The reusable asset built here — an exact closed-form profile
criterion with derived `d_eff` and a verified envelope identity — makes route 1 tractable for
a future attempt.
