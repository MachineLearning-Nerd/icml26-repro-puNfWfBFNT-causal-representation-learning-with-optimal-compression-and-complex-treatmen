"""Corrected CFR-style multi-treatment model for Section 4 of arXiv 2603.11907.

Why a new module
---------------
The previously judged run reported a Base Model PEHE of 16.91 where the paper reports 0.796,
and the judge correctly flagged the ~21x scale gap.  The cause was not the discrepancy code
(src/model.py's MMD/HSIC routines are correctly vectorised and are reused here unchanged) but
the optimisation:

  * CFRModel.fit() ran `epochs` FULL-BATCH Adam steps at lr=1e-3 -- so `epochs=150` meant 150
    gradient steps in total, not 150 passes over minibatches.  A randomly initialised network
    barely moves in 150 steps at that learning rate.
  * Neither X nor Y was standardised, so the outcome scale (which grows like 0.5*(t+1)*x^T beta,
    i.e. +/-22 at K=20) sat far outside the range a freshly initialised net can reach.

The result was a model that predicted essentially no treatment effect.  Under the Appendix D.1
generator a zero-effect predictor scores almost exactly the 16.9 that was reported -- so the
number measured the untrained network, not the paper's method.  src/pehe_conventions.py
therefore always reports the zero-effect reference alongside every PEHE.

Fidelity notes
--------------
* The balancing losses are the paper's raw aggregates and are deliberately NOT normalised by
  term count: R_pair sums C(K,2) MMD terms, R_ova sums K, R_agg is one HSIC.  At K=20 that
  makes R_pair ~190x larger than R_agg at equal alpha, which IS the paper's "over-constraint"
  mechanism (Section 4.2).  Normalising them would erase the effect under test.
* All three strategies share one architecture -- a representation trunk plus a head conditioned
  on a learned treatment embedding (the paper's "Vectorized Treatment Embeddings", Section 5.1)
  -- so differences between strategies come from the balancing term alone, not from capacity.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.model import (
    differentiable_hsic,
    differentiable_ova_mmd_sum,
    differentiable_pairwise_mmd_sum,
)


class CFRFixed:
    def __init__(
        self,
        input_dim: int,
        K: int,
        repr_dim: int = 100,
        hidden: int = 200,
        emb_dim: int = 8,
        strategy: str = "pair",
        alpha: float = 1.0,
        steps: int = 3000,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        sigma: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.K, self.strategy, self.alpha = K, strategy, alpha
        self.steps, self.lr, self.sigma, self.device = steps, lr, sigma, device
        self.emb_dim = min(emb_dim, K)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, repr_dim), nn.ELU(),
        ).to(device)
        self.emb = nn.Embedding(K, self.emb_dim).to(device)
        self.head = nn.Sequential(
            nn.Linear(repr_dim + self.emb_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        ).to(device)
        # Balancing embedding for HSIC, kept separate from the head embedding so the
        # aggregation constraint cannot be trivially satisfied by collapsing the head input.
        self.bal_emb = nn.Embedding(K, self.emb_dim).to(device)

        self.params = (
            list(self.phi.parameters())
            + list(self.emb.parameters())
            + list(self.head.parameters())
            + list(self.bal_emb.parameters())
        )
        self.opt = torch.optim.Adam(self.params, lr=lr, weight_decay=weight_decay)
        self._x_mu = self._x_sd = self._y_mu = self._y_sd = None

    # -- scaling ----------------------------------------------------------------------
    def _fit_scalers(self, X, Y):
        self._x_mu, self._x_sd = X.mean(0), X.std(0) + 1e-8
        self._y_mu, self._y_sd = float(Y.mean()), float(Y.std()) + 1e-8

    def _tx(self, X):
        return torch.tensor((X - self._x_mu) / self._x_sd, dtype=torch.float32, device=self.device)

    # -- model ------------------------------------------------------------------------
    def _forward(self, Xt, Tt):
        Z = self.phi(Xt)
        return self.head(torch.cat([Z, self.emb(Tt)], dim=1)).squeeze(-1), Z

    def _balance(self, Z, Tt):
        if self.strategy == "pair":
            return differentiable_pairwise_mmd_sum(Z, Tt, self.K, self.sigma)
        if self.strategy == "ova":
            return differentiable_ova_mmd_sum(Z, Tt, self.K, self.sigma)
        if self.strategy == "agg":
            return differentiable_hsic(Z, self.bal_emb(Tt), self.sigma, 1.0)
        return torch.zeros((), device=Z.device)

    def fit(self, X, T, Y, log_every=0):
        self._fit_scalers(X, Y)
        Xt = self._tx(X)
        Tt = torch.tensor(T, dtype=torch.long, device=self.device)
        Yt = torch.tensor((Y - self._y_mu) / self._y_sd, dtype=torch.float32, device=self.device)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.steps)
        for step in range(self.steps):
            self.opt.zero_grad()
            Yh, Z = self._forward(Xt, Tt)
            pred = nn.functional.mse_loss(Yh, Yt)
            loss = pred + self.alpha * self._balance(Z, Tt) if self.alpha > 0 else pred
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 5.0)
            self.opt.step()
            sched.step()
            if log_every and step % log_every == 0:
                print(f"    step {step:5d}  mse={pred.item():.4f}  total={loss.item():.4f}", flush=True)
        self.final_mse = float(pred.item())
        return self

    @torch.no_grad()
    def predict_all_treatments(self, X):
        """(n, K) matrix of predicted outcomes on the ORIGINAL outcome scale."""
        Xt = self._tx(X)
        Z = self.phi(Xt)
        out = np.zeros((len(X), self.K))
        for t in range(self.K):
            tt = torch.full((len(X),), t, dtype=torch.long, device=self.device)
            yh = self.head(torch.cat([Z, self.emb(tt)], dim=1)).squeeze(-1)
            out[:, t] = yh.cpu().numpy() * self._y_sd + self._y_mu
        return out
