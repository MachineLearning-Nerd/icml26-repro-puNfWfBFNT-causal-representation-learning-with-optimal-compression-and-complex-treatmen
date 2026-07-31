#!/usr/bin/env python3
"""Main verification entrypoint for puNfWfBFNT reproduction.

Runs all 6 claim verifiers sequentially, collects results, and writes EVAL.md.
Each verifier writes raw artifacts to .openresearch/artifacts/.
"""
from __future__ import annotations
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verifiers.common import save_json, log, system_info, ARTIFACTS_DIR

# This node (orx/closed-form-profile) tests one decision: does the profile criterion of
# eq. (8)/(9) admit an INTERIOR minimiser with positive curvature -- i.e. do Assumptions
# 3.4(i) and 3.7(i) actually hold?  Theorems 3.5 and 3.8 are conditional on that, and the
# previously judged run measured them in cells where it fails.  Claim 3 re-runs unchanged as
# the cumulative regression check for the one already-verified claim.
CLAIMS = [
    ("claim5_k20_fixed", "Claim 5: K=20 pairwise instability vs aggregation stability"),
    ("claim3_hsic_o1", "HSIC O(1) complexity + K-independent deviation [REGRESSION]"),
]


def run_claim(module_name: str, claim_text: str) -> dict:
    """Run a single claim verifier, catching exceptions."""
    log(f"--- Running {module_name}: {claim_text} ---")
    try:
        mod = __import__(f"verifiers.{module_name}", fromlist=["run"])
        result = mod.run()
        log(f"--- {module_name}: {result.get('verdict', 'ERROR')} ---\n")
        return result
    except Exception as e:
        log(f"--- {module_name}: ERROR: {e} ---\n")
        traceback.print_exc()
        return {
            "claim": claim_text,
            "verdict": "BLOCKED",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def write_eval_md(results: list[dict], total_elapsed: float):
    """Write EVAL.md summary."""
    lines = [
        "# EVAL.md — puNfWfBFNT Reproduction Results",
        "",
        f"**Paper:** Causal Representation Learning with Optimal Compression under Complex Treatments (arXiv 2603.11907)",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Total runtime:** {total_elapsed:.1f}s",
        "",
        "## Summary",
        "",
        "| # | Claim | Verdict | Key Evidence |",
        "|---|-------|---------|--------------|",
    ]

    for i, r in enumerate(results, 1):
        verdict = r.get("verdict", "BLOCKED")
        claim = r.get("claim", "Unknown")
        # Extract key evidence
        evidence = ""
        if "eps_ite" in r:
            evidence = f"eps_ITE={r['eps_ite']:.4f}"
        elif "K_scaling" in r:
            evidence = f"K-scaling exponents measured"
        elif "pair_pehe_alpha5" in r:
            evidence = f"pair PEHE@α=5: {r['pair_pehe_alpha5']:.3f}"
        elif "tree_ok" in r:
            evidence = f"tree={r['tree_ok']}, cyclic={r['cyclic_ok']}"
        elif "error" in r:
            evidence = f"ERROR: {r['error'][:80]}"
        lines.append(f"| {i} | {claim} | **{verdict}** | {evidence} |")

    n_verified = sum(1 for r in results if r.get("verdict") == "VERIFIED")
    n_falsified = sum(1 for r in results if r.get("verdict") == "FALSIFIED")
    n_blocked = sum(1 for r in results if r.get("verdict") == "BLOCKED")

    lines.extend([
        "",
        f"**Verified:** {n_verified}/6  **Falsified:** {n_falsified}/6  **Blocked:** {n_blocked}/6",
        "",
        "## Detailed Results",
        "",
    ])

    for i, r in enumerate(results, 1):
        lines.append(f"### Claim {i}: {r.get('claim', 'Unknown')}")
        lines.append(f"- **Verdict:** {r.get('verdict', 'BLOCKED')}")
        if "elapsed_seconds" in r:
            lines.append(f"- **Runtime:** {r['elapsed_seconds']:.1f}s")
        lines.append(f"- **Claim text:** {r.get('claim_text', 'N/A')}")
        lines.append("")

    eval_path = os.path.join(ARTIFACTS_DIR, "EVAL.md")
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        f.write("\n".join(lines))
    return eval_path


def main():
    log("=" * 70)
    log("puNfWfBFNT: Full Claim Verification Suite")
    log("Paper: Causal Representation Learning with Optimal Compression (2603.11907)")
    log("=" * 70)

    t_start = time.perf_counter()
    results = []

    for module_name, claim_text in CLAIMS:
        result = run_claim(module_name, claim_text)
        results.append(result)

    total_elapsed = time.perf_counter() - t_start

    # Write EVAL.md
    eval_path = write_eval_md(results, total_elapsed)
    log(f"EVAL.md written to {eval_path}")

    # Write combined results
    combined = {
        "paper": "Causal Representation Learning with Optimal Compression under Complex Treatments",
        "arxiv": "2603.11907",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_seconds": total_elapsed,
        "system_info": system_info(),
        "results": results,
        "n_verified": sum(1 for r in results if r.get("verdict") == "VERIFIED"),
        "n_falsified": sum(1 for r in results if r.get("verdict") == "FALSIFIED"),
        "n_blocked": sum(1 for r in results if r.get("verdict") == "BLOCKED"),
    }
    save_json(combined, "all_results.json")

    log("=" * 70)
    log(f"DONE: {combined['n_verified']} VERIFIED, {combined['n_falsified']} FALSIFIED, {combined['n_blocked']} BLOCKED")
    log(f"Total runtime: {total_elapsed:.1f}s")
    log("=" * 70)

    # Exit nonzero if any claim is BLOCKED
    if combined["n_blocked"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
