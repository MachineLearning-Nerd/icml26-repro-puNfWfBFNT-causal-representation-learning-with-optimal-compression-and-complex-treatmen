"""Multi-Treatment CausalEGM: generative model with treatment embeddings,
softmax intervention, and geodesic regularization (Section 5, arXiv 2603.11907).

Architecture (simplified for CPU):
  - Treatment embedding: e: T -> R^{d_e}
  - Outcome generator: G(Z_c, e_T) -> Y_hat
  - Geodesic loss: forces ||e_i - e_j|| ≈ d_M(t_i, t_j)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class TreatmentEmbedding(nn.Module):
    """Learnable treatment embedding e: T -> R^{d_e}."""

    def __init__(self, K: int, d_e: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(K, d_e)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, T):
        return self.embedding(T)


class OutcomeGenerator(nn.Module):
    """Maps (X_features, treatment_embedding) -> outcome Y."""

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


class GeodesicCausalEGM:
    """Multi-Treatment CausalEGM with geodesic regularization.

    Parameters:
      input_dim: covariate dimension
      K: number of treatments
      d_e: embedding dimension
      lambda_geo: geodesic loss weight
      geo_dist: (K, K) ground-truth geodesic distance matrix
    """

    def __init__(
        self,
        input_dim: int,
        K: int,
        geo_dist: np.ndarray,
        d_e: int = 2,
        lambda_geo: float = 5.0,
        lr: float = 1e-3,
        epochs: int = 500,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.K = K
        self.d_e = d_e
        self.lambda_geo = lambda_geo
        self.lr = lr
        self.epochs = epochs
        self.device = device

        self.geo_dist_t = torch.tensor(geo_dist, dtype=torch.float32, device=device)

        torch.manual_seed(seed)
        self.treat_emb = TreatmentEmbedding(K, d_e).to(device)
        self.generator = OutcomeGenerator(input_dim, d_e).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.treat_emb.parameters()) + list(self.generator.parameters()), lr=lr
        )

    def _geodesic_loss(self) -> torch.Tensor:
        """L_geo = mean_{i,j} (||e_i - e_j|| - d_M(i,j))^2."""
        emb = self.treat_emb.embedding.weight  # (K, d_e)
        dist_emb = torch.cdist(emb, emb)  # (K, K)
        return ((dist_emb - self.geo_dist_t) ** 2).mean()

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """Train the model."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        T_t = torch.tensor(T, dtype=torch.long, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)

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

    def interpolate(
        self,
        X: np.ndarray,
        t_A: int,
        t_B: int,
        n_steps: int = 101,
    ) -> np.ndarray:
        """Interpolate outcomes from treatment t_A to t_B.

        Y_alpha = G(X, (1-alpha)*e_A + alpha*e_B)
        Returns array of shape (n_steps,) for mean outcome at each alpha.
        """
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
        """Return learned treatment embeddings (K, d_e)."""
        with torch.no_grad():
            return self.treat_emb.embedding.weight.cpu().numpy()

    def predict_outcome(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Predict outcome for given X and T."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        T_t = torch.tensor(T, dtype=torch.long, device=self.device)
        with torch.no_grad():
            e_T = self.treat_emb(T_t)
            Y_hat = self.generator(X_t, e_T)
        return Y_hat.cpu().numpy()
