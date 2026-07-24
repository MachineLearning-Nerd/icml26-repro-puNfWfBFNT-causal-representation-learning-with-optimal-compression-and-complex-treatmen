"""Multi-Treatment CausalEGM: generative model with treatment embeddings,
softmax intervention, and geodesic regularization (Section 5, arXiv 2603.11907).

Uses MDS initialization + two-phase training for robust geodesic embedding.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class TreatmentEmbedding(nn.Module):
    def __init__(self, K: int, d_e: int = 2, init: np.ndarray | None = None):
        super().__init__()
        self.embedding = nn.Embedding(K, d_e)
        if init is not None:
            self.embedding.weight.data = torch.tensor(init, dtype=torch.float32)
        else:
            nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, T):
        return self.embedding(T)


class OutcomeGenerator(nn.Module):
    def __init__(self, input_dim: int, d_e: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + d_e, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, X, e_T):
        return self.net(torch.cat([X, e_T], dim=-1)).squeeze(-1)


def mds_init(geo_dist: np.ndarray, d_e: int = 2, seed: int = 42) -> np.ndarray:
    """Initialize embeddings via classical multidimensional scaling from distance matrix."""
    from sklearn.manifold import MDS
    n = geo_dist.shape[0]
    mds = MDS(n_components=d_e, dissimilarity="precomputed", random_state=seed,
              normalized_stress="auto")
    emb = mds.fit_transform(geo_dist)
    # Scale to match geodesic distance scale
    emb_dist = np.zeros_like(geo_dist)
    for i in range(n):
        for j in range(n):
            emb_dist[i, j] = np.linalg.norm(emb[i] - emb[j])
    scale = np.median(geo_dist[geo_dist > 0]) / max(np.median(emb_dist[emb_dist > 0]), 1e-8)
    return emb * scale


class GeodesicCausalEGM:
    """Multi-Treatment CausalEGM with geodesic regularization.

    Two-phase training:
      Phase 1: Pre-train embeddings with geodesic loss only
      Phase 2: Train full model (embeddings + generator) jointly
    """

    def __init__(
        self,
        input_dim: int,
        K: int,
        geo_dist: np.ndarray,
        d_e: int = 2,
        lambda_geo: float = 10.0,
        lr: float = 1e-3,
        epochs: int = 800,
        pretrain_epochs: int = 1000,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.K = K
        self.d_e = d_e
        self.lambda_geo = lambda_geo
        self.lr = lr
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.device = device

        self.geo_dist_t = torch.tensor(geo_dist, dtype=torch.float32, device=device)

        torch.manual_seed(seed)
        init = mds_init(geo_dist, d_e, seed)
        self.treat_emb = TreatmentEmbedding(K, d_e, init=init).to(device)
        self.generator = OutcomeGenerator(input_dim, d_e).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.treat_emb.parameters()) + list(self.generator.parameters()), lr=lr
        )

    def _geodesic_loss(self) -> torch.Tensor:
        emb = self.treat_emb.embedding.weight
        dist_emb = torch.cdist(emb, emb)
        return ((dist_emb - self.geo_dist_t) ** 2).mean()

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        T_t = torch.tensor(T, dtype=torch.long, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)

        # Phase 1: Pre-train embeddings with geodesic loss
        emb_opt = torch.optim.Adam(self.treat_emb.parameters(), lr=self.lr)
        for epoch in range(self.pretrain_epochs):
            emb_opt.zero_grad()
            loss = self._geodesic_loss()
            loss.backward()
            emb_opt.step()

        # Phase 2: Train full model jointly
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            e_T = self.treat_emb(T_t)
            Y_hat = self.generator(X_t, e_T)
            pred_loss = nn.functional.mse_loss(Y_hat, Y_t)
            geo_loss = self._geodesic_loss()
            loss = pred_loss + self.lambda_geo * geo_loss
            loss.backward()
            self.optimizer.step()

        return self

    def interpolate(self, X: np.ndarray, t_A: int, t_B: int, n_steps: int = 101) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        alphas = np.linspace(0, 1, n_steps)
        results = np.zeros(n_steps)

        emb_A = self.treat_emb.embedding.weight[t_A]
        emb_B = self.treat_emb.embedding.weight[t_B]

        with torch.no_grad():
            for i, alpha in enumerate(alphas):
                e_interp = (1 - alpha) * emb_A + alpha * emb_B
                e_batch = e_interp.unsqueeze(0).expand(len(X_t), -1)
                Y_hat = self.generator(X_t, e_batch)
                results[i] = Y_hat.mean().item()

        return results

    def get_embeddings(self) -> np.ndarray:
        with torch.no_grad():
            return self.treat_emb.embedding.weight.cpu().numpy()

    def predict_outcome(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        T_t = torch.tensor(T, dtype=torch.long, device=self.device)
        with torch.no_grad():
            e_T = self.treat_emb(T_t)
            Y_hat = self.generator(X_t, e_T)
        return Y_hat.cpu().numpy()
