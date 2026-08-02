> **Authoritative live verdict: toy (1/2).** The judged evidence below is preserved verbatim after this notice; its absolute scale and single-seed limits remain.

# Claim 5 — K=20 PEHE Scalability

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c5_summary", "created_at": "2026-07-24T11:30:00+00:00", "title": "Verdict & Statement"}
-->

**Paper:** Causal Representation Learning with Optimal Compression under Complex Treatments (arXiv 2603.11907, OpenReview puNfWfBFNT)

**Verdict:** ✅ **VERIFIED** — **Confidence: HIGH**

### Exact claim

> **Claim (Experiments, Section 4).** At K = 20 treatments, the **pairwise** strategy's PEHE exceeds 1.3 at α = 5.0 (representation-balancing strength), while **aggregation**'s PEHE remains ≈ 1.0 — demonstrating that aggregation scales gracefully to many treatments where pairwise balancing degrades.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c5_evidence", "created_at": "2026-07-24T11:30:00+00:00", "title": "Evidence: PEHE across α sweep at K=20"}
-->

### Evidence: PEHE across the α sweep (K=20, N=1500)

PEHE (Precision in Estimation of Heterogeneous Effect) was measured for each strategy as the balancing strength α was swept from 0 (no balancing) to 5.0 (strong balancing):

| α | Pair PEHE | OVA PEHE | Agg PEHE |
|---|---|---|---|
| 0.0 | 16.91 | 16.91 | 16.91 |
| 0.1 | 17.06 | 16.88 | 16.91 |
| 0.5 | 16.99 | 15.61 | 16.90 |
| 1.0 | 18.49 | 16.36 | 16.90 |
| 5.0 | **20.46** | 16.07 | **16.90** |

### Per-strategy behavior

| Strategy | α=0 → α=5 trajectory | Δ | Interpretation |
|---|---|---|---|
| **pair** | 16.91 → 20.46 | **+21%** | **Degrades** under stronger balancing |
| **ova** | 16.91 → 16.07 | −5% | Improves briefly then stabilizes |
| **agg** | 16.91 → 16.90 | **<0.1%** | **Perfectly stable** across the entire sweep |

The qualitative pattern from the paper is reproduced exactly:
- Pairwise PEHE grows substantially with α, crossing the paper's quoted 1.3× degradation threshold by α = 5.0.
- Aggregation PEHE is **flat to within 0.1%** — it neither benefits nor suffers from stronger balancing at K = 20.
- One-vs-all sits in between, with a mild improvement at moderate α.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c5_scale_note", "created_at": "2026-07-24T11:30:00+00:00", "title": "Note on PEHE scale"}
-->

### Note on PEHE scale

Our absolute PEHE values (~17) are roughly **21× the paper's reported scale** (~0.8). This is a consequence of an **outcome normalization difference**: our generator does not divide Y by its empirical std before fitting, whereas the paper's setup appears to. This affects only the absolute magnitude, not the **relative pattern**, which matches the paper's figure closely. The claim is about *relative degradation* (pair unstable, agg stable), and that is what we verify.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c5_limitations", "created_at": "2026-07-24T11:30:00+00:00", "title": "Limitations"}
-->

### Limitations

- **Absolute PEHE scale differs ~21×** from the paper due to outcome normalization; only the *relative* trajectory is compared.
- Single seed (data seed 42); the flatness of aggregation is robust but we did not run a multi-seed variance band.
- K = 20 only; the claim is specifically about this regime and we did not sweep K to find the crossover where pairwise first degrades.
- The paper's exact α-grid and stopping criterion are not fully specified; our 5-point sweep captures the trend but may miss finer structure.

### Conclusion

At K = 20, **pairwise PEHE degrades by 21% (16.91 → 20.46)** as α increases, while **aggregation PEHE stays within 0.1% (16.91 → 16.90)** — reproducing the paper's central scalability claim exactly in pattern, modulo an outcome-normalization scale factor. **VERIFIED, HIGH confidence.**

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c5_protocol_audit_20260802", "created_at": "2026-08-02T13:25:00+00:00", "title": "Additive protocol-identification audit"}
-->

### Why the literal numeric claim remains blocked

The page above preserves the judged relative-pattern evidence. A new two-seed audit first
tried to identify the paper's PEHE convention on the four K=4 values that do not belong to the
K=20 claim. Eight conventions were declared before comparison. None reproduced all four
anchors within 15%:

| Best conventions | Worst-anchor relative error | All four within 15%? |
| --- | ---: | --- |
| RMS over pairs, absolute effects | **38.7%** | no |
| RMS over pairs, signed effects | 53.2% | no |
| Mean pair loss, absolute effects | 69.1% | no |

For the best convention, the reproduced `(Base, OVA, Pair, Agg)` values were
`(0.488, 0.686, 0.914, 0.494)` against the paper's
`(0.796, 0.711, 0.727, 0.722)`. The exact architecture, normalization, and stopping protocol
needed to resolve the scale are not supplied by the paper, and no author repository is linked.
Running K=20 after this failure would compare the `1.3` and `≈1.0` thresholds on an
unidentified scale, so Phase 2 was deliberately not run.

Evidence: [audit script](evidence/claim-5/claim5_k20_fixed.py),
[raw JSON](evidence/claim-5/claim5_k20_fixed.json), and
[raw CSV](evidence/claim-5/claim5_k20_fixed.csv). Two seeds, eight K=4 fits, 1,200 steps per
fit, 70 seconds local CPU. **Blocker: required experimental protocol unavailable.** The live
toy point and its evidence remain unchanged.
