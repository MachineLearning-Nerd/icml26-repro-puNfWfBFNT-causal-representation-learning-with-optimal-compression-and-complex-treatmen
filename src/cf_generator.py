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
                 lambda_bal=50.0, lambda_cyc=1.0, lambda_var=10.0, steps=1500, batch=256,
                 lr=2e-3, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.K, self.lambda_bal, self.steps = K, lambda_bal, steps
        self.lambda_cyc, self.lambda_var, self.batch = lambda_cyc, lambda_var, batch
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
        """Minibatch training with reconstruction, HSIC balancing, and latent cycle consistency.

        Minibatching matters for more than speed.  With only a few hundred factual images and
        full-batch updates, the decoder can MEMORISE each observed (z, e_t) pair -- driving
        reconstruction near zero while learning nothing about the compositional structure
        (identity x angle) that an unseen (z, e_t') combination requires.  Measured on the
        2026-07-31 revision: reconstruction 0.0079, yet generated counterfactuals preserved
        identity barely above chance (top-1 retrieval 0.011 vs chance 0.0025).

        Cycle consistency attacks that directly and uses NO counterfactual supervision: decode
        at a random OTHER treatment t', re-encode, and require the content code to come back
        unchanged.  A memorised decoder cannot satisfy this, because its off-diagonal outputs
        are unconstrained; a compositional one can.
        """
        Xt = torch.tensor(X_fact, dtype=torch.float32)
        Tt = torch.tensor(T_fact, dtype=torch.long)
        n = len(Xt)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.steps)
        g = torch.Generator().manual_seed(12345)
        for it in range(self.steps):
            idx = torch.randint(0, n, (min(self.batch, n),), generator=g)
            xb, tb = Xt[idx], Tt[idx]
            toh = torch.nn.functional.one_hot(tb, self.K).float()
            self.opt.zero_grad()
            z = self.enc(xb)
            rec = nn.functional.mse_loss(self.dec(torch.cat([z, self.emb(tb)], 1)), xb)
            bal = hsic(z, toh)
            # Variance floor (VICReg-style).  HSIC is trivially minimised by a CONSTANT z --
            # independence is free if the code carries no information -- and the collapsed
            # optimum makes the decoder emit the per-angle mean, which is exactly the baseline
            # we are trying to beat.  Penalising per-dimension std below 1 makes collapse
            # unavailable while leaving genuine independence attainable.
            var = torch.relu(1.0 - z.std(0)).mean()
            loss = rec + self.lambda_bal * bal + self.lambda_var * var
            if self.lambda_cyc > 0:
                t2 = (tb + torch.randint(1, self.K, tb.shape, generator=g)) % self.K
                x_cf = self.dec(torch.cat([z, self.emb(t2)], 1))
                cyc = nn.functional.mse_loss(self.enc(x_cf), z)
                loss = loss + self.lambda_cyc * cyc
            else:
                cyc = torch.zeros(())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 5.0)
            self.opt.step()
            sched.step()
            if log and it % 1000 == 0:
                log(f"      step {it}: rec={rec.item():.5f} hsic={bal.item():.6f} "
                    f"cyc={float(cyc):.5f}")
        with torch.no_grad():
            self.final_rec = float(rec.item())
            self.final_hsic = float(bal.item())
            self.final_cyc = float(cyc)
            self.final_zstd = float(z.std(0).mean())
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


class ConvCounterfactualGenerator(CounterfactualGenerator):
    """Same objective, but convolutional -- the right inductive bias for 28x28 images.

    The MLP version reached counterfactual MSE 0.0586 against a per-angle-mean baseline of
    0.0522: it renders the correct angle (classifier accuracy 0.982) but loses unit identity,
    because a dense 784->784 map has to learn rotation as an unstructured permutation of
    pixels.  Convolutions share weights across spatial positions, so local stroke structure
    survives the transform and identity has a chance to be preserved.
    """

    def __init__(self, input_dim, K, d_e=8, d_z=32, hidden=256, lambda_bal=50.0,
                 lambda_cyc=1.0, lambda_var=10.0, steps=1500, batch=256, lr=2e-3, seed=0):
        super().__init__(input_dim, K, d_e, d_z, hidden, lambda_bal, lambda_cyc,
                         lambda_var, steps, batch, lr, seed)
        assert input_dim == 784, "conv variant expects 28x28"
        torch.manual_seed(seed)
        self.enc = nn.Sequential(
            nn.Unflatten(1, (1, 28, 28)),
            nn.Conv2d(1, 32, 4, 2, 1), nn.ELU(),      # 14x14
            nn.Conv2d(32, 64, 4, 2, 1), nn.ELU(),     # 7x7
            nn.Flatten(), nn.Linear(64 * 7 * 7, d_z),
        )
        self.dec_fc = nn.Sequential(nn.Linear(d_z + d_e, 64 * 7 * 7), nn.ELU())
        self.dec_deconv = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ELU(),   # 14x14
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),  # 28x28
            nn.Flatten(),
        )
        self.dec = nn.Sequential(self.dec_fc, self.dec_deconv)
        self.params = (list(self.emb.parameters()) + list(self.enc.parameters())
                       + list(self.dec.parameters()))
        self.opt = torch.optim.Adam(self.params, lr=lr)
