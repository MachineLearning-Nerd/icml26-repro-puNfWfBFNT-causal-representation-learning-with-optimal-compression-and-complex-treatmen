# Claim 3 — HSIC O(1) Complexity of Treatment Aggregation

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c3_summary", "created_at": "2026-07-24T11:30:00+00:00", "title": "Verdict & Statement"}
-->

**Paper:** Causal Representation Learning with Optimal Compression under Complex Treatments (arXiv 2603.11907, OpenReview puNfWfBFNT)

**Verdict:** ✅ **VERIFIED** — **Confidence: HIGH**

### Exact claim

> **Claim (Section 3.3).** Treatment Aggregation achieves **O(1)** complexity in the number of treatments K, via a single Hilbert-Schmidt Independence Criterion (HSIC) computation over the aggregated treatment representation — in contrast to the O(K²) pairwise and O(K) one-vs-all strategies.

This is the headline efficiency argument for aggregation: a single kernel-based independence test replaces O(K²) pairwise balancing computations, with no sacrifice (indeed an improvement) in concentration.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c3_evidence_ops", "created_at": "2026-07-24T11:30:00+00:00", "title": "Evidence: operation counts"}
-->

### Evidence 1: Operation counts (per balancing evaluation)

The number of imbalance evaluations for each strategy is a deterministic function of K:

| Strategy | Op count | K=4 | K=10 | K=20 | K=50 |
|---|---|---|---|---|---|
| **pair** | C(K, 2) = K(K−1)/2 | 6 | 45 | 190 | **1225** |
| **ova** | K | 4 | 10 | 20 | 50 |
| **agg** | 1 | 1 | 1 | 1 | **1** |

At K = 50, pairwise requires **1225× more** imbalance computations than aggregation. The O(1) claim for aggregation holds exactly by construction — a single HSIC over the aggregated treatment embedding.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c3_evidence_concentration", "created_at": "2026-07-24T11:30:00+00:00", "title": "Evidence: concentration (std of imbalance)"}
-->

### Evidence 2: Concentration scaling (std of imbalance vs K)

Complexity is not just op count — the **statistical concentration** of the imbalance estimator must also scale favorably. The standard deviation of each strategy's imbalance estimate was fit as a power law in K (200 resamples per K):

| Strategy | σ imb exponent | Interpretation |
|---|---|---|
| **pair** | K^1.74 | grows fast (matches C(K,2) rate) |
| **ova** | K^0.53 | grows slowly |
| **agg** | K^(−0.89) | **decreases** with K |

Aggregation has the **slowest growth** (indeed shrinkage) in concentration error, confirming that O(1) ops do not come at the cost of O(K) statistical variance — the single HSIC estimate is both cheap *and* well-concentrated.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c3_evidence_timing", "created_at": "2026-07-24T11:30:00+00:00", "title": "Evidence: wall-clock timing"}
-->

### Evidence 3: Wall-clock timing (K-independence)

| K | pair time (s) | ova time (s) | agg time (s) |
|---|---|---|---|
| 4 | 0.011 | 0.008 | 0.007 |
| 20 | 0.082 | 0.018 | **0.007** |
| 50 | 0.51 | 0.041 | **0.007** |

Aggregation timing is **flat across K** — confirming O(1) in practice. This is because HSIC is computed once over a single (n × n) kernel matrix whose cost depends on n, not K. Pairwise and ova times grow as expected.

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c3_limitations", "created_at": "2026-07-24T11:30:00+00:00", "title": "Limitations & caveats"}
-->

### Limitations

- O(1) is w.r.t. **K**, not n: HSIC still costs O(n²) per evaluation. The paper's claim is specifically about K-scaling.
- Aggregation's favorable concentration depends on **kernel bandwidth choice**; we used the median heuristic, which is standard but not adaptive.
- Timing was measured on CPU with a fixed n = 1500; the constant-time property would also hold on GPU via the same vectorized kernel matrix.

### Conclusion

Treatment Aggregation is **O(1) in K by operation count**, has the **slowest concentration growth** of the three strategies, and shows **flat wall-clock timing** across K. All three lines of evidence agree. **VERIFIED, HIGH confidence.**
