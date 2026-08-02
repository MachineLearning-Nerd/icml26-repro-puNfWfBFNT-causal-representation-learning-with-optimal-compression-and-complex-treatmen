"""Independent arithmetic checker for Claim 1 trained-TV evidence."""
from __future__ import annotations

import copy
import csv
import json
import math
import os
from pathlib import Path


def evidence_dir():
    override = os.environ.get("CLAIM1_TV_EVIDENCE_DIR")
    if override:
        return Path(override)
    sibling = Path(__file__).resolve().parent
    if (sibling / "claim1_trained_tv.json").is_file():
        return sibling
    return Path(__file__).resolve().parents[1] / ".openresearch" / "artifacts"


def validate(result, rows):
    step2 = [row for row in rows if row["part"] == "step2_trained_tv"]
    step4 = [row for row in rows if row["part"] == "step4_finite_class"]
    design = result["design"]
    if len(step2) != 9 * 4 * 3 or len(step4) != 9 * 4:
        return False
    if result["derived_constants"] != {"c1": 1.0, "c2": 1.0, "c3": 0.0}:
        return False

    deletion_failures = 0
    arm_loss_disagreements = []
    for row in step2:
        source = float(row["source_risk"])
        target = float(row["target_risk"])
        tv = float(row["tv"])
        ratio = float(row["abs_gap_over_tv"])
        if tv <= 0 or target > source + tv + 1e-11:
            return False
        if not math.isclose(ratio, abs(target - source) / tv, rel_tol=0, abs_tol=1e-12):
            return False
        deletion_failures += int(target > source + 1e-8)
        arm_loss_disagreements.append(float(row["model_arm_loss_disagreement"]))
    if deletion_failures == 0:
        return False
    if max(arm_loss_disagreements) >= 1e-12:
        return False

    expected_radius = math.sqrt(
        math.log(2 * len(step4) / float(design["delta"])) / (2 * int(design["n_eval"]))
    )
    for row in step4:
        empirical = float(row["empirical_risk"])
        population = float(row["population_risk"])
        gap = float(row["absolute_gap"])
        radius = float(row["hoeffding_radius"])
        if not math.isclose(gap, abs(empirical - population), rel_tol=0, abs_tol=1e-12):
            return False
        if not math.isclose(radius, expected_radius, rel_tol=0, abs_tol=1e-12):
            return False
        if gap > radius + 1e-12:
            return False
    return True


def main():
    root = evidence_dir()
    with (root / "claim1_trained_tv.json").open() as handle:
        result = json.load(handle)
    with (root / "claim1_trained_tv.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not validate(result, rows):
        raise SystemExit("FAIL: evidence identities")

    step2_mutation = copy.deepcopy(rows)
    first_step2 = next(row for row in step2_mutation if row["part"] == "step2_trained_tv")
    first_step2["target_risk"] = str(
        float(first_step2["source_risk"]) + float(first_step2["tv"]) + 0.01
    )
    if validate(result, step2_mutation):
        raise SystemExit("FAIL: accepted domain-bound mutation")

    loss_class_mutation = copy.deepcopy(rows)
    first_loss_row = next(
        row for row in loss_class_mutation if row["part"] == "step2_trained_tv"
    )
    first_loss_row["model_arm_loss_disagreement"] = "0.1"
    if validate(result, loss_class_mutation):
        raise SystemExit("FAIL: accepted arm-loss mutation")

    step4_mutation = copy.deepcopy(rows)
    first_step4 = next(row for row in step4_mutation if row["part"] == "step4_finite_class")
    first_step4["absolute_gap"] = str(float(first_step4["hoeffding_radius"]) + 0.01)
    if validate(result, step4_mutation):
        raise SystemExit("FAIL: accepted generalization mutation")

    print("PASS: trained-TV identities; domain and generalization mutations rejected")


if __name__ == "__main__":
    main()
