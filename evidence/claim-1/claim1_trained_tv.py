"""Claim 1 Steps 2--4 on trained representations with a derived IPM constant.

The earlier trained-representation attempt fitted an MMD regression and failed its held-out
control. This audit instead follows Appendix C.2 literally. On a finite population, every
trained representation is checked to be injective. The bounded treatment-specific loss is
therefore a function of the representation, and total variation is an IPM whose function
class contains that loss. Consequently

    |E_j loss_k - E_k loss_k| <= TV(Phi(X)|T=j, Phi(X)|T=k)

with the derived constants c1=c2=1 and c3=0. No coefficient is fitted. A separate independent
evaluation sample checks the Step 4 finite-class Hoeffding remainder.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (
    differentiable_hsic,
    differentiable_ova_mmd_sum,
    differentiable_pairwise_mmd_sum,
)
from verifiers.assumption_audit import save_rows_csv
from verifiers.common import log, save_json

RNG_SEED = 20260802
K = 4
N_TRAIN = 1200
N_EVAL = 12000
STEPS = 500
DELTA = 0.05
STRATEGIES = ("pair", "ova", "agg")
ALPHAS = (0.0, 0.5, 2.0)
X_GRID = np.linspace(-2.5, 2.5, 101)
TREATMENT_OFFSETS = 0.25 * np.arange(K)


def features(x):
    return np.column_stack((x, x**2 / 6.25, np.sin(x), np.cos(x)))


def treatment_probabilities(x):
    centered = np.arange(K) - (K - 1) / 2
    logits = 1.1 * x[:, None] * centered[None, :] / 2.5
    logits += 0.25 * np.sin(1.7 * x[:, None] + np.arange(K)[None, :])
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def potential_outcomes(x):
    return np.sin(1.4 * x[:, None]) + TREATMENT_OFFSETS[None, :]


class OffsetCFR:
    """Deep representation with a shared outcome surface and known additive offsets.

    The nonzero offsets make treatment effects non-degenerate. Subtracting the declared offset
    before fitting leaves one shared response surface, so the arm-specific prediction losses
    are exactly the same function of Phi(X), as required for the c3=0 construction.
    """

    def __init__(self, input_dim, strategy, alpha, steps, seed):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.strategy, self.alpha, self.steps = strategy, alpha, steps
        self.phi = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
            nn.Linear(64, 12), nn.ELU(),
        )
        self.head = nn.Sequential(nn.Linear(12, 64), nn.ELU(), nn.Linear(64, 1))
        self.balance_embedding = nn.Embedding(K, 4)
        parameters = (
            list(self.phi.parameters())
            + list(self.head.parameters())
            + list(self.balance_embedding.parameters())
        )
        self.optimizer = torch.optim.Adam(parameters, lr=1e-3, weight_decay=1e-5)

    def _balance(self, representation, treatment):
        import torch

        if self.strategy == "pair":
            return differentiable_pairwise_mmd_sum(representation, treatment, K, 1.0)
        if self.strategy == "ova":
            return differentiable_ova_mmd_sum(representation, treatment, K, 1.0)
        if self.strategy == "agg":
            return differentiable_hsic(
                representation, self.balance_embedding(treatment), 1.0, 1.0
            )
        return torch.zeros(())

    def fit(self, x, treatment, outcome):
        import torch
        import torch.nn.functional as functional

        self.x_mean, self.x_std = x.mean(axis=0), x.std(axis=0) + 1e-8
        adjusted = outcome - TREATMENT_OFFSETS[treatment]
        self.y_mean, self.y_std = float(adjusted.mean()), float(adjusted.std()) + 1e-8
        x_tensor = torch.tensor((x - self.x_mean) / self.x_std, dtype=torch.float32)
        treatment_tensor = torch.tensor(treatment, dtype=torch.long)
        y_tensor = torch.tensor((adjusted - self.y_mean) / self.y_std, dtype=torch.float32)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.steps
        )
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            representation = self.phi(x_tensor)
            prediction = self.head(representation).squeeze(-1)
            prediction_loss = functional.mse_loss(prediction, y_tensor)
            loss = prediction_loss + self.alpha * self._balance(
                representation, treatment_tensor
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.phi.parameters()) + list(self.head.parameters()), 5.0
            )
            self.optimizer.step()
            scheduler.step()
        return self

    def _tx(self, x):
        import torch

        return torch.tensor((x - self.x_mean) / self.x_std, dtype=torch.float32)

    def predict_all_treatments(self, x):
        import torch

        with torch.no_grad():
            base = self.head(self.phi(self._tx(x))).squeeze(-1).numpy()
        base = base * self.y_std + self.y_mean
        return base[:, None] + TREATMENT_OFFSETS[None, :]


def draw_training_data(rng):
    index = rng.integers(0, len(X_GRID), N_TRAIN)
    x = X_GRID[index]
    probabilities = treatment_probabilities(x)
    treatment = np.array([rng.choice(K, p=row) for row in probabilities])
    all_outcomes = potential_outcomes(x)
    observed = all_outcomes[np.arange(N_TRAIN), treatment] + rng.normal(0, 0.05, N_TRAIN)
    return features(x), treatment, observed


def min_representation_separation(z):
    distance = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


def run():
    import torch

    rng = np.random.default_rng(RNG_SEED)
    train_x, train_t, train_y = draw_training_data(rng)
    population_x = features(X_GRID)
    population_y = potential_outcomes(X_GRID)
    assignment = treatment_probabilities(X_GRID)
    domain_weights = assignment / assignment.sum(axis=0, keepdims=True)

    rows = []
    models = []
    all_domain_bounds_hold = True
    no_ipm_control_failures = 0
    minimum_separation = math.inf
    maximum_ratio = 0.0
    maximum_arm_loss_disagreement = 0.0

    log("=== Claim 1: trained-representation TV audit ===")
    log(f"  finite population={len(X_GRID)}, training n={N_TRAIN}, K={K}")
    for strategy in STRATEGIES:
        for alpha in ALPHAS:
            seed = 100 + STRATEGIES.index(strategy) * 10 + ALPHAS.index(alpha)
            model = OffsetCFR(
                input_dim=population_x.shape[1], strategy=strategy, alpha=alpha,
                steps=STEPS, seed=seed,
            ).fit(train_x, train_t, train_y)
            with torch.no_grad():
                representation = model.phi(model._tx(population_x)).cpu().numpy()
            separation = min_representation_separation(representation)
            minimum_separation = min(minimum_separation, separation)
            predictions = model.predict_all_treatments(population_x)
            losses = np.minimum((predictions - population_y) ** 2, 1.0)
            arm_loss_disagreement = float(np.max(np.ptp(losses, axis=1)))
            maximum_arm_loss_disagreement = max(
                maximum_arm_loss_disagreement, arm_loss_disagreement
            )
            model_id = f"{strategy}-alpha-{alpha:g}"
            models.append((model_id, model, losses))

            # Injectivity makes each bounded loss_k a well-defined function of Phi(X). Total
            # variation is the IPM over all [0,1]-valued functions, hence the coefficient is 1.
            injective = separation > 1e-7
            for source in range(K):
                for target in range(K):
                    if source == target:
                        continue
                    source_risk = float(domain_weights[:, source] @ losses[:, source])
                    target_risk = float(domain_weights[:, target] @ losses[:, target])
                    tv = float(0.5 * np.abs(
                        domain_weights[:, source] - domain_weights[:, target]
                    ).sum())
                    gap = target_risk - source_risk
                    holds = injective and target_risk <= source_risk + tv + 1e-12
                    all_domain_bounds_hold = all_domain_bounds_hold and holds
                    if gap > 1e-8:
                        no_ipm_control_failures += 1
                    maximum_ratio = max(maximum_ratio, abs(gap) / tv if tv else 0.0)
                    rows.append({
                        "part": "step2_trained_tv", "model": model_id,
                        "source": source, "target": target,
                        "source_risk": source_risk, "target_risk": target_risk,
                        "tv": tv, "abs_gap_over_tv": abs(gap) / tv if tv else 0.0,
                        "model_arm_loss_disagreement": arm_loss_disagreement,
                        "bound_holds": holds,
                    })
            log(f"  {model_id:18s} min ||Phi(x)-Phi(x')||={separation:.3e}")

    # Step 4: the trained models are frozen before this independent sample is drawn. The
    # finite class has len(models)*K bounded losses, so a union-bound Hoeffding radius applies.
    eval_index = rng.integers(0, len(X_GRID), N_EVAL)
    class_size = len(models) * K
    radius = math.sqrt(math.log(2 * class_size / DELTA) / (2 * N_EVAL))
    all_generalization_bounds_hold = True
    zero_complexity_control_failures = 0
    max_generalization_gap = 0.0
    for model_id, _, losses in models:
        for treatment in range(K):
            empirical = float(losses[eval_index, treatment].mean())
            population = float(losses[:, treatment].mean())
            gap = abs(empirical - population)
            holds = gap <= radius
            all_generalization_bounds_hold = all_generalization_bounds_hold and holds
            zero_complexity_control_failures += int(gap > 1e-12)
            max_generalization_gap = max(max_generalization_gap, gap)
            rows.append({
                "part": "step4_finite_class", "model": model_id,
                "treatment": treatment, "empirical_risk": empirical,
                "population_risk": population, "absolute_gap": gap,
                "hoeffding_radius": radius, "bound_holds": holds,
            })

    checks = {
        "trained_models": len(models),
        "minimum_representation_separation": minimum_separation,
        "all_representations_injective": minimum_separation > 1e-7,
        "all_step2_tv_bounds_hold": all_domain_bounds_hold,
        "maximum_absolute_risk_gap_over_tv": maximum_ratio,
        "maximum_arm_loss_disagreement": maximum_arm_loss_disagreement,
        "no_ipm_control_failures": no_ipm_control_failures,
        "step4_class_size": class_size,
        "step4_hoeffding_radius": radius,
        "step4_maximum_gap": max_generalization_gap,
        "all_step4_bounds_hold": all_generalization_bounds_hold,
        "zero_complexity_control_failures": zero_complexity_control_failures,
        "minimum_treatment_probability": float(assignment.min()),
    }
    passed = (
        checks["all_representations_injective"]
        and checks["maximum_arm_loss_disagreement"] < 1e-12
        and checks["all_step2_tv_bounds_hold"]
        and checks["no_ipm_control_failures"] > 0
        and checks["all_step4_bounds_hold"]
        and checks["zero_complexity_control_failures"] > 0
    )
    verdict = "VERIFIED" if passed else "BLOCKED"
    log(f"  max |risk gap|/TV={maximum_ratio:.4f}; deletion failures={no_ipm_control_failures}")
    log(f"  Step 4 max gap={max_generalization_gap:.5f} <= radius={radius:.5f}")
    log(f"Verdict: {verdict}")

    save_rows_csv(rows, "claim1_trained_tv.csv")
    result = {
        "claim": "Claim 1 Steps 2-4 on trained representations with TV IPM",
        "verdict": verdict,
        "derived_constants": {"c1": 1.0, "c2": 1.0, "c3": 0.0},
        "checks": checks,
        "design": {
            "seed": RNG_SEED, "K": K, "population_states": len(X_GRID),
            "n_train": N_TRAIN, "n_eval": N_EVAL, "steps": STEPS,
            "strategies": list(STRATEGIES), "alphas": list(ALPHAS),
            "loss_bound": 1.0, "delta": DELTA,
        },
    }
    save_json(result, "claim1_trained_tv.json")
    return result


if __name__ == "__main__":
    run()
