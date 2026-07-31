"""Counterfactual IMAGE generation on Rotated MNIST, for the first clause of Claim 6.

The claim is that Multi-Treatment CausalEGM "extends the discriminative framework to
high-dimensional counterfactual generation".  The judged 2026-07-31 evidence did not establish
that clause, so this module measures it directly rather than by proxy.

Task.  Each unit i is a base '3' image observed at exactly ONE angle t_i.  Given that single
observed image, generate what unit i would have looked like at a different angle t'.  Because
the dataset is built by rotating one base image to all K angles, the true counterfactual image
x_i^(t') exists and is never shown to the model -- it is ground truth, not a proxy.

Why a plain autoencoder cannot do this, and what makes it work.  Training only on factuals,
(x_i^(t_i), t_i) -> x_i^(t_i), admits a trivial solution: copy the input and ignore the
treatment embedding.  That solution has zero training loss and generates nothing.  The escape
is the paper's own mechanism -- force the content code z to be INDEPENDENT of the treatment
via the aggregation (HSIC) balancing term.  If z carries no angle information, the decoder can
only obtain the angle from e_t, so rendering at e_{t'} necessarily re-renders the digit.  The
generative capability is therefore produced by the balancing penalty the paper proposes, which
is exactly the claim under test, and lambda_bal=0 is the natural negative control.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _rbf(A, sigma):
    d2 = torch.cdist(A, A) ** 2
    return torch.exp(-d2 / (2 * sigma ** 2 + 1e-12))


def hsic(Z, Toh):
    """Biased HSIC between representation Z and one-hot treatment Toh (agg strategy)."""
    n = Z.shape[0]
    with torch.no_grad():
        d = torch.cdist(Z, Z)
        sigma = torch.median(d[d > 0]).clamp(min=1e-3)
    Kz, Kt = _rbf(Z, sigma), Toh @ Toh.T
    H = torch.eye(n, device=Z.device) - 1.0 / n
    return torch.trace(Kz @ H @ Kt @ H) / (n - 1) ** 2


class CounterfactualGenerator:
    """Encoder x -> z (content), decoder (z, e_t) -> x_hat. Trained on factuals only."""

    def __init__(self, input_dim, K, d_e=8, d_z=32, hidden=256,
                 lambda_bal=50.0, steps=1500, lr=2e-3, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.K, self.lambda_bal, self.steps = K, lambda_bal, steps
        self.emb = nn.Embedding(K, d_e)
        nn.init.normal_(self.emb.weight, std=0.5)
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, d_z),
        )
        self.dec = nn.Sequential(
            nn.Linear(d_z + d_e, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, input_dim), nn.Sigmoid(),     # pixels are in [0, 1]
        )
        self.params = (list(self.emb.parameters()) + list(self.enc.parameters())
                       + list(self.dec.parameters()))
        self.opt = torch.optim.Adam(self.params, lr=lr)

    def fit(self, X_fact, T_fact, log=None):
        Xt = torch.tensor(X_fact, dtype=torch.float32)
        Tt = torch.tensor(T_fact, dtype=torch.long)
        Toh = torch.nn.functional.one_hot(Tt, self.K).float()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.steps)
        for it in range(self.steps):
            self.opt.zero_grad()
            z = self.enc(Xt)
            xh = self.dec(torch.cat([z, self.emb(Tt)], 1))
            rec = nn.functional.mse_loss(xh, Xt)
            bal = hsic(z, Toh)
            (rec + self.lambda_bal * bal).backward()
            torch.nn.utils.clip_grad_norm_(self.params, 5.0)
            self.opt.step()
            sched.step()
            if log and it % 500 == 0:
                log(f"      step {it}: rec={rec.item():.5f} hsic={bal.item():.6f}")
        with torch.no_grad():
            self.final_rec = float(rec.item())
            self.final_hsic = float(bal.item())
        return self

    @torch.no_grad()
    def generate(self, X_obs, t_target):
        """Counterfactual image for each row of X_obs, rendered at treatment t_target."""
        Xt = torch.tensor(X_obs, dtype=torch.float32)
        z = self.enc(Xt)
        e = self.emb.weight[t_target].expand(len(Xt), -1)
        return self.dec(torch.cat([z, e], 1)).numpy()

    @torch.no_grad()
    def codes(self, X_obs):
        return self.enc(torch.tensor(X_obs, dtype=torch.float32)).numpy()


def treatment_decodability(Z, T, seed=0):
    """How much angle information survives in z: multinomial logistic accuracy, CV.

    Chance level is 1/K.  A generator that merely copies its input leaves the angle fully
    decodable (accuracy ~1.0) and cannot generate counterfactuals; this quantifies that
    directly instead of inferring it from the reconstruction loss.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    clf = LogisticRegression(max_iter=2000)   # multinomial is the default in sklearn >= 1.5
    return float(cross_val_score(clf, Z, T, cv=3, scoring="accuracy").mean())
