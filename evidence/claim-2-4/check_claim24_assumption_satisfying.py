"""Independent fail-closed checker for the Claims 2/4 profile audit artifact."""
from __future__ import annotations

import copy
import json
import math
import os

try:
    from verifiers.claim24_assumption_satisfying import DELTA, KAPPA, RHO, term_count
except ModuleNotFoundError:
    from claim24_assumption_satisfying import DELTA, KAPPA, RHO, term_count

REPOSITORY_ARTIFACT = os.path.join(
    os.path.dirname(__file__),
    "..",
    ".openresearch",
    "artifacts",
    "claim24_assumption_satisfying",
    "result.json",
)
ARTIFACT = os.environ.get(
    "CLAIM24_RESULT",
    os.path.join(os.path.dirname(__file__), "result.json")
    if os.path.isfile(os.path.join(os.path.dirname(__file__), "result.json"))
    else REPOSITORY_ARTIFACT,
)


def validate(result: dict) -> list[str]:
    errors = []
    constants = result.get("constants", {})
    if constants.get("delta") != DELTA or constants.get("rho") != RHO:
        errors.append("constants changed")
    if result.get("assumptions", {}).get("infimum_population_curvature") != KAPPA:
        errors.append("curvature mismatch")

    finite_cells = result.get("claim_2_finite_sample", {}).get("cells", [])
    if not finite_cells:
        errors.append("finite-sample cells missing")
    for cell in finite_cells:
        m = term_count(cell["strategy"], cell["K"])
        expected_r = m * math.sqrt(2 * math.log(2 / DELTA) / cell["n"])
        if cell["m"] != m or not math.isclose(cell["r_bound"], expected_r, rel_tol=1e-12):
            errors.append(f"finite identity mismatch: {cell['strategy']} K={cell['K']} n={cell['n']}")
        if cell["violations_on_event"] != 0:
            errors.append(f"finite conclusion violation: {cell['strategy']} K={cell['K']} n={cell['n']}")
        if cell["empirical_event_coverage"] < 1 - DELTA:
            errors.append(f"finite coverage failure: {cell['strategy']} K={cell['K']} n={cell['n']}")

    asymptotic_cells = result.get("claim_4_asymptotic", {}).get("cells", [])
    if not asymptotic_cells:
        errors.append("asymptotic cells missing")
    for cell in asymptotic_cells:
        m = term_count(cell["strategy"], cell["K"])
        expected = (m + RHO * m * (m - 1)) / (KAPPA**2 * cell["n"])
        if cell["m"] != m or not math.isclose(
            cell["predicted_alpha_variance"], expected, rel_tol=1e-12
        ):
            errors.append(f"variance identity mismatch: {cell['strategy']} K={cell['K']} n={cell['n']}")

    controls = result.get("controls", {})
    boundary = controls.get("boundary_optimum", {})
    zero = controls.get("zero_curvature", {})
    if boundary.get("assumption_3_7_i_satisfied") is not False:
        errors.append("boundary control did not fail Assumption 3.7(i)")
    if boundary.get("mass_at_boundary", 0) <= 0.45 or boundary.get("ks_statistic", 0) <= 0.20:
        errors.append("boundary control lacks the predicted atom/non-normality")
    if zero.get("assumption_3_4_i_satisfied") is not False:
        errors.append("zero-curvature control did not fail Assumption 3.4(i)")
    if zero.get("finite_sample_denominator_defined") is not False:
        errors.append("zero-curvature denominator was accepted")

    return errors


def mutation_checks(result: dict) -> dict:
    mutations = {}

    bad_finite = copy.deepcopy(result)
    bad_finite["claim_2_finite_sample"]["cells"][0]["violations_on_event"] = 1
    mutations["finite_violation_rejected"] = bool(validate(bad_finite))

    bad_variance = copy.deepcopy(result)
    bad_variance["claim_4_asymptotic"]["cells"][0]["predicted_alpha_variance"] *= 0.5
    mutations["variance_mutation_rejected"] = bool(validate(bad_variance))

    bad_control = copy.deepcopy(result)
    bad_control["controls"]["boundary_optimum"]["assumption_3_7_i_satisfied"] = True
    mutations["boundary_mutation_rejected"] = bool(validate(bad_control))
    return mutations


def run() -> None:
    with open(ARTIFACT) as f:
        result = json.load(f)
    errors = validate(result)
    mutations = mutation_checks(result)
    if errors or not all(mutations.values()):
        raise SystemExit(f"FAIL: errors={errors}, mutations={mutations}")
    print(f"PASS: independent identities and controls; mutations={mutations}")


if __name__ == "__main__":
    run()
