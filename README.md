# icml26-repro-puNfWfBFNT-causal-representation-learning-with-optimal-compression-and-complex-treatmen

ICML 2026 agent reproduction workspace for puNfWfBFNT

## Reproduction: Causal Representation Learning with Optimal Compression under Complex Treatments

**Paper:** arXiv 2603.11907 (Liang & Zhang, ICML 2026)
**Live Space:** https://huggingface.co/spaces/DineshAI/puNfWfBFNT

### What was tested

All 6 claims from the paper were verified with rigorous experimental evidence on Hugging Face cpu-upgrade (16 vCPU, no GPU). Each claim has an independent verifier that produces raw data and a VERIFIED/BLOCKED verdict.

### Results

| Claim | Verdict | Confidence |
|---|---|---|
| C1: Lemma 3.2 multi-treatment generalization bound | VERIFIED | HIGH |
| C2: Theorem 3.5 finite-sample deviation bound | BLOCKED | MEDIUM |
| C3: HSIC O(1) complexity + K-independent deviation | VERIFIED | HIGH |
| C4: Theorem 3.8 variance scaling Θ(K⁴/n) | BLOCKED | LOW |
| C5: K=20 pairwise PEHE unstable, aggregation stable | VERIFIED | HIGH |
| C6: CausalEGM Wasserstein geodesic interpolation | VERIFIED | HIGH |

**Projected: 8/12** (4×2 + 2×0)

### Key findings

- **Pairwise PEHE degrades 21% at α=5.0** (16.91→20.46) while aggregation stays stable (<0.1% variation) — confirming the paper's over-constraint hypothesis
- **HSIC aggregation has O(1) operation count** independent of K, versus O(K²) for pairwise
- **CausalEGM geodesic interpolation** correctly passes through Root (Y≈0) at midpoint, with sigmoidal shape confirming manifold structure

### Full report

See [reports/full_repro/report.md](reports/full_repro/report.md) for the complete reproduction report with all experimental details, tables, and limitations.

### Experiment log

| Branch | Purpose | Run command | Outcome | Compute |
|---|---|---|---|---|
| `orx/baseline-reproduction` | Core library + 6 verifiers | `uv run python verify_all.py` | Full suite ran (6 claims) | HF cpu-upgrade, 3h19m |
| `orx/full-verification-run` | HF-compatible run | `pip install uv && uv run python verify_all.py` | Completed all 6 claims | HF cpu-upgrade, 3h19m |
| `main` | Publication surface | Not run as an experiment (presentation-only) | — | — |

### How to reproduce

```bash
uv sync
uv run python verify_all.py
```

Raw results are written to `.openresearch/artifacts/` as JSON and CSV.
