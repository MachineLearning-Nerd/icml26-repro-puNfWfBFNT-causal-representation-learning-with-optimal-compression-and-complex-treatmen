# Claim 1 (current verification) — Lemma 3.2, derived not fitted

> **This page supersedes [C1 (historical rejected baseline)](#/claims/c1).**
> The superseded page fitted the constants `C_F=0.82, C_B=0.01, C_C=0.01` numerically.
> Current code: `verifiers/claim1_lemma32_proof.py` at Git SHA `5d6bdf5`.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c1cur_summary", "created_at": "2026-07-31T14:09:15+00:00", "title": "Verdict & exact claim"}
-->

**Paper:** Causal Representation Learning with Optimal Compression under Complex Treatments
(arXiv 2603.11907, OpenReview `puNfWfBFNT`).
Source retrieved from `https://ar5iv.labs.arxiv.org/html/2603.11907` on 2026-07-31,
SHA-256 `c8773e4f4c981bc4c3b84d2ae4ea3f51423f126574414f2c73c422107d3e63a8`.

**Live judged verdict (2026-07-24, sha `d4db74e3`): `toy` — 1/2.**
Judge's stated reason: *"the constants are numerically fitted rather than derived from the
proof"* and *"the negative control shows the bound remains feasible without the imbalance
term"*.

**This run's finding: Step 1 of the proof is VERIFIED with an exact, provably tight constant.**
Steps 2–4 are reported as **not verifiable from the paper** — see Scope below.

### Exact claim (Lemma 3.2, Section 3.1)

> Under Assumption 3.1, for any fixed strategy 𝒮 and any (Φ,h) in the model class, with
> probability at least 1−δ,
> **ε_ITE(Φ,h) ≤ C_F·ε_F(Φ,h) + C_B·ℛ_𝒮(Φ) + C_C·Complexity(h∘Φ; n,δ)**,
> where C_F, C_B, C_C > 0 depend only on regularity parameters (e.g. L_h, L_ℓ, M and the
> discrepancy function class) and do not depend on n.

The paper labels this **"(schematic form)"** and states plainly that it
*"does not claim that a particular coefficient ratio is known or identifiable in practice."*

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c1cur_scope", "created_at": "2026-07-31T14:09:15+00:00", "title": "Why the constants cannot be derived — scope"}
-->

### Why the previous attempt had to fit, and why fitting can never work

Appendix C.2 splits into four steps of very different status:

| Step | Content | Constant | Status |
|---|---|---|---|
| **1** (eqs. 18–19) | W₂ reverse triangle inequality + (a+b)²≤2a²+2b², summed over pairs | **exactly 2(K−1)** | **fully explicit** |
| 2 (eq. 20) | domain-adaptation step to factual risk + IPM | `c₁,c₂,c₃` | *"proof template"*, constants never given |
| 3 | summation over arms / strategies | `C_F′, C_B′, C_0` | inherits Step 2 |
| 4 | add uniform generalisation term | `C_C` | *"absorbing constants"* |

Step 2 states only that `c₁,c₂,c₃` *"depend only on regularity constants and boundedness"*.
They are never instantiated. **So C_F, C_B, C_C are not identifiable from the paper, and any
numerical value for them is fitted by construction.** That also explains the judge's second
objection mechanically: with C_F fitted large enough, the C_F·ε_F term alone dominates, so
deleting the imbalance term leaves the bound feasible and the control cannot fail.

This page therefore verifies **Step 1 only**, and says so.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c1cur_symbolic", "created_at": "2026-07-31T14:09:15+00:00", "title": "Evidence 1: symbolic proof certificate"}
-->

### Evidence 1 — symbolic certificate (independent checker: `sympy`)

Three machine-checked facts, all returning `True`:

| Check | Statement | Result |
|---|---|---|
| Quadratic step | `2a² + 2b² − (a+b)²` simplifies **identically** to `(a−b)²`, hence ≥ 0 | ✅ `True` |
| Summation identity | `Σ_{j<k} [2W_j² + 2W_k²] ≡ 2(K−1)·Σ_t W_t²`, expanded exhaustively for **K = 2…12** | ✅ `True` (all 11) |
| Source of the constant | each arm `t` appears in **exactly K−1** pairs, verified exhaustively for K = 2…12 | ✅ `True` (all 11) |

Together these derive eq. (19), `ε_ITE ≤ 2(K−1)·Σ_t E_X[W₂(P̂_t,X, P_t,X)²]`, from eq. (18)
with the constant **derived rather than assumed**.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c1cur_numeric", "created_at": "2026-07-31T14:09:15+00:00", "title": "Evidence 2: numerical verification, 2800 configurations"}
-->

### Evidence 2 — numerical verification on non-degenerate outcomes

`W₂` between univariate Gaussians is exact: `W₂(N(m₁,s₁²), N(m₂,s₂²)) = √((m₁−m₂)² + (s₁−s₂)²)`.
Using **non-degenerate** distributions matters: for point masses W₂ collapses to |mean
difference| and Step 1 would only ever be tested in the degenerate special case.

400 random configurations × 200 units per K, seed `20260731`:

| K | eq. (18) violations | eq. (19) violations | max ratio ε_ITE / RHS(19) |
|---|---|---|---|
| 2 | **0** | **0** | 0.4042 |
| 3 | **0** | **0** | 0.3811 |
| 4 | **0** | **0** | 0.3594 |
| 6 | **0** | **0** | 0.3538 |
| 8 | **0** | **0** | 0.3477 |
| 12 | **0** | **0** | 0.3371 |
| 20 | **0** | **0** | 0.3352 |

**2800 configurations, zero violations.** Max ratio ≈ 0.34 everywhere, so the bound holds
with slack — it is satisfied, but *not* tight on random draws. That observation is what
motivates the next section rather than being the end of the story.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c1cur_control", "created_at": "2026-07-31T14:09:15+00:00", "title": "Evidence 3: extremal tightness + the negative control that fails"}
-->

### Evidence 3 — extremal tightness, and a control that genuinely fails

The judge's second objection was that the old negative control still passed. Probing
tightness on *random* configurations cannot fail: the bound is ~3× loose there, so shrinking
the constant never violates it. Tightness is a **worst-case** property, so the worst case is
constructed directly.

Equality in `(a+b)² ≤ 2a²+2b²` needs `a=b`; equality in the W₂ reverse triangle inequality
needs the two arms' errors to point in **opposite directions**. So set every true
`P_t = N(0,1)` and every prediction `P̂_t = N(s_t·d, 1)` with `s_t = ±1`, signs split as evenly
as possible. Then `W_t = d` for all t, `τ_true = 0` for every pair, and `τ̂ = |s_j − s_k|·d`:

```
eps_ITE = 4 d² · floor(K/2) · ceil(K/2)
RHS(19) = 2 (K−1) · K · d²
ratio   = 2·floor(K/2)·ceil(K/2) / ((K−1)·K)
```

| K | extremal ratio (measured) | closed form | match |
|---|---|---|---|
| **2** | **1.000000** | **1.000000** | ✅ |
| 3 | 0.666667 | 0.666667 | ✅ |
| 4 | 0.666667 | 0.666667 | ✅ |
| 6 | 0.600000 | 0.600000 | ✅ |
| 8 | 0.571429 | 0.571429 | ✅ |
| 12 | 0.545455 | 0.545455 | ✅ |
| 20 | 0.526316 | 0.526316 | ✅ |

Measured values match the closed form to machine precision at every K.

**Negative control (must fail, and does):** at K=2 the extremal ratio is **exactly 1.000000**,
so replacing `2(K−1)` by `c·(K−1)` for **any** c < 2 is violated by an explicit,
reproducible configuration. Checked at factors 0.99, 0.9, 0.75, 0.5 — **all violated**.
The control passes for the true constant and fails for every smaller one.

**Two-sided conclusion:** the constant `2(K−1)` is **exactly optimal at K=2** and becomes up
to **2× loose** as K→∞ (ratio → 1/2). The paper claims only an upper bound, so this is
consistent with Lemma 3.2 while quantifying its slack — a strictly stronger statement than
"the inequality held".

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c1cur_repro", "created_at": "2026-07-31T14:09:15+00:00", "title": "Reproduction: command, environment, provenance"}
-->

### Exact reproduction

```bash
git clone https://github.com/MachineLearning-Nerd/icml26-repro-puNfWfBFNT-causal-representation-learning-with-optimal-compression-and-complex-treatmen
cd icml26-repro-puNfWfBFNT-*
git checkout 5d6bdf5
pip install uv && uv run python verify_all.py          # the fixed run command, unchanged across nodes
```

| Item | Value |
|---|---|
| Verifier | `verifiers/claim1_lemma32_proof.py` |
| Git SHA | `5d6bdf5` (branch `orx/claims-1-and-6-corrected-proof-certificate-de-ci`) |
| Seed | `RNG_SEED = 20260731` (deterministic; `numpy.random.default_rng`) |
| Grid | `K ∈ {2,3,4,6,8,12,20}`, 400 configs × 200 units each |
| Environment | Python 3.12, pinned via `uv.lock`; numpy, scipy, sympy |
| Compute | **Hugging Face `cpu-upgrade`**, job `DineshAI/6a6cac4ba00abefd4b289b2c` |
| Estimated cores | 8 vCPU requested; workload is single-threaded numpy/sympy |
| Runtime | **≈15 s** for this verifier (full suite job longer) |
| Exit behaviour | `verify_all.py` exits **nonzero** if any claim is not VERIFIED |

Raw outputs: `.openresearch/artifacts/claim1_lemma32_proof.json` and `.csv`, regenerated by
the command above. The complete results payload is also emitted to stdout between
`<<<BEGIN_ALL_RESULTS_JSON>>>` / `<<<END_ALL_RESULTS_JSON>>>` markers in the run log.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c1cur_limits", "created_at": "2026-07-31T14:09:15+00:00", "title": "Limitations & deviations"}
-->

### Limitations and deviations — stated plainly

1. **Only Step 1 is verified.** C_F, C_B and C_C are *not* verified and, per the argument
   above, are **not derivable from the paper**. Anyone reporting numerical values for them is
   fitting, not deriving. The claim as worded ("decomposing ITE error into factual prediction
   error, representation-level imbalance, and model complexity") is therefore verified as to
   its **first reduction and its exact constant**, and explicitly **not** as to the
   coefficients of the three terms.
2. **Gaussian outcome distributions.** W₂ is exact in closed form for Gaussians; the
   inequality is proved symbolically for arbitrary distributions, so the numerical part is a
   confirmation on a family where the estimand is exactly computable, not a general proof.
3. **Exhaustive symbolic expansion covers K = 2…12.** The identity is proved for symbolic K
   by the pair-incidence argument; exhaustive expansion is an independent cross-check over a
   finite range, not the proof itself.
4. **Assumption 3.1 is not re-derived.** Step 1 uses only metric properties of W₂ and does
   not invoke Assumption 3.1 (i)–(iv); those assumptions enter at Step 2, which is out of
   scope here.

### Conclusion

Step 1 of Lemma 3.2's proof — the reduction of multi-treatment ITE risk to per-arm
potential-outcome error with constant **2(K−1)** — is **symbolically proved**, **numerically
confirmed over 2800 non-degenerate configurations with zero violations**, and **shown to be
exactly tight at K=2**, with a negative control that fails for every smaller constant.
The remaining coefficients are not identifiable from the paper and are reported as such.
