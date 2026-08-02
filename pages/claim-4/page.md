# Claim 4 — asymptotic normality and strategy-dependent variance

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_pu_c4_claim_20260802", "created_at": "2026-08-02T12:55:00+00:00", "title": "Exact claim, scope, and live baseline"}
-->
**Official claim.** Under Assumption 3.7, Theorem 3.8 states
`√n(α̂_S−α∞_S) ⇒ Normal(0, σ²_S / Q″_S(α∞_S)²)`. Under mild dependence across treatment
arms, Corollary 3.9 gives
`Var(α̂_pair)=Θ(K⁴/n)`, `Var(α̂_ova)=Θ(K²/n)`, and `Var(α̂_agg)=Θ(1/n)`.

**Live judged baseline:** `inconclusive`, 0/2, at exact revision
`1396ce3ce8364ba4073e59348db14422c2855557`. The previous estimator was degenerate at the
boundary. This page uses the same explicit bounded construction as Claim 2, now inside every
condition of Assumption 3.7. It reports an audit finding, not an earned leaderboard point.

Paper anchors: [Section 3.3, Assumption 3.7, Theorem 3.8, Corollary 3.9](https://arxiv.org/html/2603.11907#S3.SS3),
[Appendix C.6](https://arxiv.org/html/2603.11907#A3.SS6), and
[Appendix C.7](https://arxiv.org/html/2603.11907#A3.SS7).

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_pu_c4_exact_20260802", "created_at": "2026-08-02T12:55:00+00:00", "title": "Exact normal-limit and variance identities"}
-->
The population optimum is the fixed interior point `α∞=α₀=1`, and `Q″(α∞)=κ=0.5`.
The empirical profile score at `α∞` is `mean(S_i)`. The `S_i` are iid, bounded, centered, and
have positive finite variance

`v_m = m + ρm(m−1)`, with `ρ=0.25`.

The classical iid CLT gives `√n mean(S_i) ⇒ Normal(0,v_m)`, uniformly on every fixed
neighborhood because the random score offset does not depend on α. With probability tending
to one, the compact-range projection is inactive, so

`Var(α̂) = [m + ρm(m−1)] / (κ² n)`.

Substituting the paper's exact component counts gives:

| Strategy | m | Exact normalized variance | Limit |
| --- | ---: | ---: | ---: |
| pair | `K(K−1)/2` | `n Var(α̂)/K⁴` | `ρ/(4κ²)=0.25` |
| one-vs-all | `K` | `n Var(α̂)/K²` | `ρ/κ²=1` |
| aggregate | `1` | `n Var(α̂)` | `1/κ²=4` |

Thus the claimed Θ(K⁴/n), Θ(K²/n), and Θ(1/n) rates follow from an explicit non-vanishing
positive covariance across a non-vanishing fraction of component pairs—the dependence premise
used in the paper's own Corollary 3.9 proof.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_pu_c4_results_20260802", "created_at": "2026-08-02T12:55:00+00:00", "title": "Fixed-seed diagnostics"}
-->
The analytic identities are the evidence; the Monte Carlo run is an independent fixed-seed
diagnostic. It uses 20,000 replicates, `K∈{4,8,12,20,32}`, and
`n∈{4096,65536,1048576,16777216}`. Exact binomial aggregation samples each replicate without
materializing an n-by-m array.

At `K=8`, `n=16777216`:

| Strategy | KS statistic | Wasserstein to N(0,1) | standardized variance | variance relative error | interior rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| pair | 0.00574 | 0.00685 | 1.01450 | 0.01450 | 1.000 |
| one-vs-all | 0.00411 | 0.00514 | 1.00815 | 0.00815 | 1.000 |
| aggregate | 0.00639 | 0.00838 | 1.01390 | 0.01390 | 1.000 |

At the largest n, the exact normalized pair variance increases from 0.21094 at K=4 to 0.23604
at K=32 toward its 0.25 limit; OVA decreases from 1.75 to 1.09375 toward 1; aggregate is
exactly 4 for every K. Every final-n Monte Carlo variance is within 1.5% of its analytic value.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_pu_c4_control_20260802", "created_at": "2026-08-02T12:55:00+00:00", "title": "Boundary control and reproduction"}
-->
**Condition-relaxing control.** Restricting the range to `[0,α₀]` puts the population optimum
on its upper boundary and violates Assumption 3.7(i). The estimator then has mass 0.50175 at
that boundary and KS statistic 0.5 against a normal law. The checker rejects the theorem audit
as out of scope. This directly reproduces the failure mode of the earlier judged attempt.

```bash
.venv/bin/python -m verifiers.claim24_assumption_satisfying
.venv/bin/python -m verifiers.check_claim24_assumption_satisfying
```

Published evidence: [producer](evidence/claim-2-4/claim24_assumption_satisfying.py),
[independent checker](evidence/claim-2-4/check_claim24_assumption_satisfying.py),
[raw JSON](evidence/claim-2-4/result.json), and
[asymptotic CSV](evidence/claim-2-4/asymptotic.csv). All files reproduce byte-for-byte on a
second clean run. Local CPU only; no Hugging Face Job and no GPU.

**Finding:** asymptotic normality and all three variance rates are verified on the explicit
assumption-satisfying profile construction. This does not claim that the paper's unreleased
neural implementation satisfies the assumptions.
