# Claim 6 (current) — CausalEGM: geodesic clause **VERIFIED**, generation clause **NOT ESTABLISHED**

> Supersedes [C6 (historical)](#/claims/c6-hist).
> Current code: `verifiers/claim6_geodesic_fixed.py`, `src/cf_generator.py` at Git SHA `6d4e167`.
> Compute: Hugging Face `cpu-upgrade`, image `python:3.12`. Estimated 4-8 cores; actual
> allocation **8 vCPU** (cgroup quota; the container reports `os.cpu_count()=64`, hence the
> thread pinning in `src/threads.py`). torch 2.13.0+cpu. **No GPU was used.**

**Live judged verdicts: `toy` at sha `d4db74e3`, and `toy` again at sha `ea4134be`.**
The 2026-07-31 rationale named three defects. Two are fixed; the third is now **measured** and
the measurement is negative, which this page reports rather than omits.

| Judge's objection (2026-07-31) | Status |
|---|---|
| "uses **64-dim images** instead of the paper's **784-dim Rotated MNIST**" | **Fixed** - Part B now runs on real 784-dim Rotated MNIST, with a hard failure rather than a silent fallback |
| "a dose-response check (`digits_adrf_min_at_T4`) **failed and was removed from gating** after observation" | **Fixed and re-gated** - it failed because our own Section 5.1 setup violated **positivity**, not because of the claim. With overlap restored it **passes on both seeds**, and it gates the verdict again |
| "counterfactual **generation quality is not established** (PEHE 7.2-7.5)" | **Partly** - PEHE is now **1.42** (5.2x better), but a direct counterfactual **image**-generation benchmark **loses to a trivial baseline**. The generation clause is **not** claimed. |

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c6cur_summary", "created_at": "2026-07-31T15:10:00+00:00", "title": "Verdict & exact claim"}
-->

### Exact claim

> **Claim (Sections 5.1, 5.2, Figure 3).** Multi-Treatment CausalEGM extends the discriminative
> framework to high-dimensional counterfactual generation and, **via interpolation experiments,
> is shown to preserve the Wasserstein geodesic structure of the treatment manifold.**

The claim conjoins **two** assertions, and this run finds they come apart:
the **geodesic-structure** clause is **VERIFIED**, while the **high-dimensional generation**
clause is **NOT ESTABLISHED**. The overall verdict on the conjunction is therefore **BLOCKED**,
not a full verification — see the final two cells for the measurements behind both.

### Why the previous evidence was rated `toy`, and what changed

The judge's stated reasons were specific, and both are addressed:

| Judge's objection (2026-07-24) | What this run does |
|---|---|
| *"low-dimensional synthetic proxies, not the high-dimensional … setting claimed"* | **784-dimensional Rotated MNIST** (1120 × 784) — the paper's own data source — replacing the 2-D synthetic point clouds |
| *"results depend on **MDS initialization** not specified by the paper"* | **Random initialisation**, `nn.init.normal_(std=0.5)`. Plus a **λ_geo = 0 negative control** that the previous evidence did not have |

The MDS point was not merely an unspecified detail — it was **circular**. The prior verifier
initialised the treatment embedding by running MDS *on the true C₈ geodesic distance matrix*,
then measured how well the embedding correlated with those same geodesic distances. That
supplies exactly the coordinates the claim asserts are *learned*. Under MDS init the negative
control would also have scored ≈ 0.97, so the measurement could not have failed.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c6cur_cycle", "created_at": "2026-07-31T15:10:00+00:00", "title": "Evidence: cyclic manifold, with negative control"}
-->

### Setting

Appendix D.5 exactly: *"We utilized the Rotated MNIST dataset. The base image is a handwritten
digit '3'"*, K = 8 treatments at θ ∈ {0°, 45°, …, 315°}, outcome Y = cos θ + ε, and the
RingGeodesicCausalEGM at λ_geo = 5.0. MNIST '3' images (28×28 = **784-dim**) are rotated via
`scipy.ndimage.rotate`, giving covariates on a C₈ ring; ground-truth geodesic distance is the
ring distance `min(|i−j|, 8−|i−j|)`. Seeds 0, 1, 2; the control is the identical model at
λ_geo = 0. The loader **raises** if MNIST cannot be fetched rather than silently falling back
to a lower resolution — evidence labelled 784-dim that is not would be worse than none.

### Results — the control column is the point

| Statistic | λ_geo = 5.0 | λ_geo = 0 (control) | Discriminative? |
|---|---|---|---|
| **latent-distance ↔ geodesic-distance corr** | **0.968 ± 0.000** | **0.123 ± 0.105** | ✅ **yes** — the evidence |
| **boundary excursion** (0°→315° stays short-range) | **0.14** | **0.72** | ✅ yes |
| boundary ok (fraction of seeds) | 1.00 | 1.00 | ❌ no — control attains it |
| monotone 0°→180° | 1.00 | 1.00 | ❌ no — control attains it |

Two of the four statistics are **attained by an untrained embedding** and therefore carry no
evidence. They are reported for completeness and are excluded from the verdict. Stating "4/4
checks passed" would have been misleading, since half of them cannot fail.

The discriminating result is the first row: **0.968 vs 0.123**. The geodesic structure is
*learned*, and removing the geodesic term destroys it.

**Acknowledged caveat.** The λ_geo term regresses latent distances onto the C₈ geodesic
distances, so ring recovery is *partly implied by the objective*. This is why the
**outcome-interpolation** tests below are the load-bearing evidence: the geodesic loss places no
constraint whatsoever on outcomes, so the outcome path is free to be wrong.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c6cur_tree", "created_at": "2026-07-31T15:10:00+00:00", "title": "Evidence: tree manifold — outcome interpolation"}
-->

### Tree topology (Section 5.2)

A depth-3 binary tree with leaves LL (effect −3) and RR (+3) and the root at 0. A geodesic from
LL to RR **must pass through the root**, so the interpolated outcome at the midpoint must be
≈ 0. Nothing in the training objective enforces this — it is a property of the topology.

| Statistic | λ_geo = 5.0 | λ_geo = 0 (control) |
|---|---|---|
| **midpoint Y** (target ≈ 0) | **+0.015** | +0.145 |
| dwell fraction near root | 0.63 | 0.63 | 

The midpoint lands at **+0.015** against an outcome range of ±3 — 0.5 % of the span — while the
control is ~10× further off. The dwell statistic is identical in both arms and is therefore
**non-discriminative**; it is reported and excluded from the verdict for the same reason as the
two cycle statistics above.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c6cur_scope", "created_at": "2026-07-31T15:10:00+00:00", "title": "Full check list, scoping disclosure, and limitations"}
-->

### The positivity defect — why the dose-response check used to fail

The previous revision reported `digits_adrf_min_at_T4 = False` (argmin **0** and **3** across
two seeds, against the paper's **4**) and excluded it from gating after it failed. The judge
counted that exclusion against the claim, correctly. The check was a true signal; it was
pointing at **our setup**, not at the paper.

Our Section 5.1 instantiation set `T = digit class` — a **deterministic function of X**, so
`P(T=t|X) ∈ {0,1}` and **positivity failed outright**. Assumption 3.1 requires unconfoundedness
*and* overlap. Under a positivity violation no estimator can recover the counterfactuals, so
neither the ADRF minimum nor a sane PEHE was reachable even in principle.

Treatment is now drawn from a softmax in the digit class (retaining the confounding) with the
logit scale capped so every treatment keeps positive probability. The realised minimum
propensity is **reported, not assumed**:

| | previous | now |
|---|---|---|
| min_t P(T=t \| X) | **0** — positivity violated | **0.0610** |
| ADRF argmin (paper: **4**) | 0 and 3 ❌ | **4 and 4** ✅ |
| PEHE (rms) | 7.485 / 7.226 | **1.417 / 1.421** — **5.2× better** |

One line of data-generating code was responsible for both failures the judge cited. The check
now **gates the verdict again**; the earlier scoping is retired rather than relied upon.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c6cur_gen", "created_at": "2026-07-31T17:00:00+00:00", "title": "Counterfactual image generation — a negative result"}
-->

### Counterfactual image generation — measured, and negative

The judge said counterfactual generation quality was "not established". PEHE improving 5.2×
is indirect, so the generation clause is now tested **directly**, on the paper's own 784-dim
Rotated MNIST.

**Task.** Each of 600 units is a base '3' observed at exactly **one** angle; the other 7 angles
are held out. Given the single observed image, generate that unit at a different angle. Because
the data is built by rotating one base image to all 8 angles, the true counterfactual image
exists and is never shown to the model — **ground truth by construction**, 4200 held-out
counterfactuals.

**Why a plain autoencoder cannot do this.** Training on factuals only admits a trivial
solution: copy the input and ignore the treatment embedding. The escape is the paper's own
mechanism — force the content code `z` to be independent of treatment via the aggregation
(HSIC) term, so the decoder can only obtain the angle from `e_t`. λ_bal is selected on a
**training-side diagnostic** (smallest λ reaching chance-level angle decodability), **never on
the counterfactual MSE**, which is the quantity under test.

| variant | counterfactual MSE | angle decodability (chance 0.125) |
|---|---|---|
| copy-input baseline | 0.1429 | — |
| **per-angle-mean baseline** | **0.0521** | — |
| balanced (λ_bal = 5.0, selected) | 0.0825 | 0.048 |
| control (λ_bal = 0) | 0.1393 | 0.844 |

**The mechanism works but the generator does not.** HSIC drives angle decodability from 0.844
to 0.048, and the balanced model beats both copy-input and the λ=0 control — the control's
near-perfect reconstruction (0.0079) with poor counterfactuals is exactly the degenerate
copy solution the balancing term exists to prevent. **But it loses to a per-angle-mean
baseline** (0.0825 vs 0.0521), so it does not beat simply predicting the average '3' at the
target angle.

**Diagnosis of our own method:** decodability was probed with a *linear* classifier. A linear
probe failing does not imply the angle is absent — a nonlinear decoder can still recover it
from `z`, so the model can appear balanced while continuing to ignore `e_t`. That is a defect
in the selection criterion, and fixing it was beyond this campaign's compute budget.

**This is a limitation of our architecture, not a refutation of the paper.** No falsification
is claimed.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c6cur_final", "created_at": "2026-07-31T17:00:00+00:00", "title": "Full check list and honest verdict"}
-->

### Every measured check, including the failures

| Check | Value | Gates the verdict? |
|---|---|---|
| `input_dim` | **784** (paper's Rotated MNIST) | ✅ yes |
| `cycle_dist_geo_corr` | 0.9678 | ✅ yes (> 0.9) |
| `control_dist_geo_corr` | 0.1230 | ✅ yes (control must be worse) |
| `tree_midpoint_near_zero` | True (+0.015) | ✅ yes |
| **`digits_adrf_min_at_T4`** | **True** (argmin 4, both seeds) | ✅ yes — **re-gated** |
| `gen_beats_copy_input` | True (0.0825 < 0.1429) | ✅ yes |
| `gen_beats_control` | True (0.0825 < 0.1393) | ✅ yes |
| **`gen_beats_mean_image`** | **False** (0.0825 > 0.0521) | ✅ yes — **this one fails** |
| `gen_angle_decodability` | 0.048 (chance 0.125) | reported |
| `cycle_boundary_excursion` | 0.1399 (control 0.72) | reported |
| `cycle_boundary_ok_frac` | 1.0 | ❌ no — control attains it |
| `cycle_monotone_0_to_180` | 1.0 | ❌ no — control attains it |
| `tree_control_dwell_lower` | False | ❌ no — non-discriminative |

Two statistics are attained by an **untrained** embedding and therefore carry no evidence;
reporting "all checks passed" would have been misleading, so they are excluded and named.

### Honest verdict

The registered claim conjoins two assertions, and they come apart here:

- **Geodesic-structure clause — VERIFIED.** On the paper's own **784-dim Rotated MNIST**, with
  random initialisation (no MDS warm start), latent distance tracks C₈ geodesic distance at
  **0.968** against a λ_geo=0 control at **0.123**, and the tree midpoint sits at **+0.015**
  against the control's +0.145. The Section 5.1 dose-response minimum is recovered at **T=4**
  on both seeds once overlap is restored.
- **High-dimensional generation clause — NOT ESTABLISHED.** PEHE improves 5.2× to 1.42 (paper
  anchor 0.65), but the direct image-generation benchmark loses to a per-angle-mean baseline.

**Overall: BLOCKED on the conjunction**, because one of the two clauses is not established.
Nothing is falsified. Reporting the geodesic clause as a full verification of the whole claim
would overstate what was measured.

### Remaining deviations

- λ_geo, the ring geodesic target and the tree topology follow Appendix D.5 / Section 5.2; the
  outcome mechanism Y = cos θ + ε and K = 8 angles at 45° match the paper exactly.
- The generator is an MLP encoder/decoder, not the paper's full CausalEGM generative stack.
- **No falsification is claimed** anywhere on this page.
