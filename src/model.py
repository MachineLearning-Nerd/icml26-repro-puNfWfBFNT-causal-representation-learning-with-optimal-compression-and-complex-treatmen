"""CFR (Causal Fairness Regularization) model for ITE estimation.

Implements the representation-learning pipeline of Shalit et al. (2017)
extended to multi-treatment settings per arXiv 2603.11907:
  Phi: X -> Z (representation)
  h: (Z, T) -> Y_hat (outcome predictor)
  Loss = eps_F(theta) + alpha * R_hat_S(theta)

Uses differentiable MMD/HSIC implementations so gradients flow through
the balancing term to the representation network.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def differentiable_rbf_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: float) -> torch.Tensor:
    """RBF kernel matrix (differentiable)."""
    sq_dist = (
        (X ** 2).sum(dim=1, keepdim=True)
        + (Y ** 2).sum(dim=1, keepdim=True).t()
        - 2.0 * X @ Y.t()
    )
    sq_dist = torch.clamp(sq_dist, min=0.0)
    return torch.exp(-sq_dist / (2.0 * sigma ** 2))


def differentiable_mmd2(X: torch.Tensor, Y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Biased MMD^2 V-statistic (differentiable)."""
    m, n = X.shape[0], Y.shape[0]
    Kxx = differentiable_rbf_kernel(X, X, sigma)
    Kyy = differentiable_rbf_kernel(Y, Y, sigma)
    Kxy = differentiable_rbf_kernel(X, Y, sigma)
    term1 = Kxx.sum() / (m ** 2)
    term2 = Kyy.sum() / (n ** 2)
    term3 = 2.0 * Kxy.sum() / (m * n)
    return torch.clamp(term1 + term2 - term3, min=0.0)


def differentiable_mmd_ipm(X: torch.Tensor, Y: torch.Tensor, sigma: float) -> torch.Tensor:
    """MMD as IPM: sqrt(MMD^2)."""
    return torch.sqrt(differentiable_mmd2(X, Y, sigma) + 1e-10)


def differentiable_hsic(X: torch.Tensor, E: torch.Tensor, sigma_x: float, sigma_e: float) -> torch.Tensor:
    """HSIC V-statistic (differentiable)."""
    n = X.shape[0]
    Kx = differentiable_rbf_kernel(X, X, sigma_x)
    Le = differentiable_rbf_kernel(E, E, sigma_e)
    Kx_c = Kx - Kx.mean(dim=0, keepdim=True) - Kx.mean(dim=1, keepdim=True) + Kx.mean()
    Le_c = Le - Le.mean(dim=0, keepdim=True) - Le.mean(dim=1, keepdim=True) + Le.mean()
    return torch.clamp((Kx_c * Le_c).sum() / ((n - 1) ** 2), min=0.0)


def differentiable_pairwise_mmd_sum(Z: torch.Tensor, T: torch.Tensor, K: int, sigma: float) -> torch.Tensor:
    """Vectorized sum of pairwise MMD IPMs over all C(K,2) pairs.

    Computes the full N×N kernel matrix once, then extracts all pairwise MMDs
    via group indicator matrices. O(N²d + K²) instead of O(K²·n²d).
    """
    N = Z.shape[0]
    K_full = differentiable_rbf_kernel(Z, Z, sigma)  # (N, N)
    T_oh = torch.nn.functional.one_hot(T, K).float()  # (N, K)
    n_k = T_oh.sum(dim=0).clamp(min=1.0)  # (K,)

    # Cross-group kernel means: A[j,k] = (1/(n_j*n_k)) * sum K(x_i, x_l) for T_i=j, T_l=k
    # = (T_oh^T @ K_full @ T_oh) / (n_j * n_k)
    cross = T_oh.t() @ K_full @ T_oh  # (K, K)
    norm = n_k.unsqueeze(1) * n_k.unsqueeze(0)  # (K, K)
    A = cross / norm.clamp(min=1.0)  # A[j,k] = mean kernel between groups j,k

    # MMD^2(j,k) = A[j,j] + A[k,k] - 2*A[j,k]
    diag = torch.diag(A)
    mmd2_mat = diag.unsqueeze(1) + diag.unsqueeze(0) - 2.0 * A  # (K, K)
    mmd2_mat = torch.clamp(mmd2_mat, min=0.0)

    # Sum sqrt(MMD^2) for upper triangle (j < k)
    mmd_mat = torch.sqrt(mmd2_mat + 1e-10)
    mask = torch.triu(torch.ones(K, K, device=Z.device), diagonal=1).bool()
    return (mmd_mat * mask).sum()


def differentiable_ova_mmd_sum(Z: torch.Tensor, T: torch.Tensor, K: int, sigma: float) -> torch.Tensor:
    """Vectorized sum of one-vs-all MMD IPMs."""
    N = Z.shape[0]
    K_full = differentiable_rbf_kernel(Z, Z, sigma)
    T_oh = torch.nn.functional.one_hot(T, K).float()
    n_k = T_oh.sum(dim=0).clamp(min=1.0)
    n_neg = (N - n_k).clamp(min=1.0)
    T_neg = 1.0 - T_oh  # (N, K), 1 if T != k

    total = torch.tensor(0.0, device=Z.device)
    cross = T_oh.t() @ K_full @ T_oh
    cross_neg = T_oh.t() @ K_full @ T_neg
    cross_neg_neg = T_neg.t() @ K_full @ T_neg

    diag_kk = torch.diag(cross) / (n_k ** 2)
    diag_neg_neg = torch.diag(cross_neg_neg) / (n_neg ** 2)
    cross_k_neg = torch.diag(cross_neg) / (n_k * n_neg)

    mmd2_vec = torch.clamp(diag_kk + diag_neg_neg - 2.0 * cross_k_neg, min=0.0)
    return torch.sqrt(mmd2_vec + 1e-10).sum()


class RepresentationNet(nn.Module):
    def __init__(self, input_dim: int, repr_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim),
        )

    def forward(self, x):
        return self.net(x)


class OutcomePredictor(nn.Module):
    def __init__(self, repr_dim: int, K: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim + K, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, t_onehot):
        return self.net(torch.cat([z, t_onehot], dim=-1)).squeeze(-1)


class CFRModel:
    """CFR model with differentiable multi-treatment balancing."""

    def __init__(
        self,
        input_dim: int,
        K: int,
        repr_dim: int = 8,
        strategy: str = "pair",
        alpha: float = 1.0,
        lr: float = 1e-3,
        epochs: int = 200,
        sigma: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.K = K
        self.repr_dim = repr_dim
        self.strategy = strategy
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.sigma = sigma
        self.device = device

        torch.manual_seed(seed)
        self.phi = RepresentationNet(input_dim, repr_dim).to(device)
        self.h = OutcomePredictor(repr_dim, K).to(device)
        self._embeddings = nn.Embedding(K, min(K, 8)).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.phi.parameters()) + list(self.h.parameters())
            + list(self._embeddings.parameters()), lr=lr
        )

    def _compute_balance_loss(self, Z: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Compute strategy-specific differentiable balancing loss (vectorized)."""
        if self.strategy == "pair":
            return differentiable_pairwise_mmd_sum(Z, T, self.K, self.sigma)
        elif self.strategy == "ova":
            return differentiable_ova_mmd_sum(Z, T, self.K, self.sigma)
        elif self.strategy == "agg":
            E = self._embeddings(T)
            return differentiable_hsic(Z, E, self.sigma, 1.0)
        else:
            return torch.tensor(0.0, device=Z.device)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        T_t = torch.tensor(T, dtype=torch.long, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
        T_oh = torch.nn.functional.one_hot(T_t, self.K).float()

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            Z = self.phi(X_t)
            Y_hat = self.h(Z, T_oh)
            pred_loss = nn.functional.mse_loss(Y_hat, Y_t)
            if self.alpha > 0:
                bal_loss = self._compute_balance_loss(Z, T_t)
                loss = pred_loss + self.alpha * bal_loss
            else:
                loss = pred_loss
            loss.backward()
            self.optimizer.step()

        return self

    def predict_outcome(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        T_t = torch.tensor(T, dtype=torch.long, device=self.device)
        T_oh = torch.nn.functional.one_hot(T_t, self.K).float()
        with torch.no_grad():
            Z = self.phi(X_t)
            Y_hat = self.h(Z, T_oh)
        return Y_hat.cpu().numpy()

    def predict_all_treatments(self, X: np.ndarray) -> np.ndarray:
        N = len(X)
        preds = np.zeros((N, self.K))
        for t in range(self.K):
            preds[:, t] = self.predict_outcome(X, np.full(N, t))
        return preds

    def compute_pehe(self, X: np.ndarray, Y_all_mean: np.ndarray) -> float:
        """Compute PEHE = mean over samples of (1/n_pairs) * sum of squared ITE errors."""
        Y_hat = self.predict_all_treatments(X)
        K = self.K
        pairs = [(j, k) for j in range(K) for k in range(j + 1, K)]
        sq_errors = []
        for j, k in pairs:
            tau_true = np.abs(Y_all_mean[:, j] - Y_all_mean[:, k])
            tau_hat = np.abs(Y_hat[:, j] - Y_hat[:, k])
            sq_errors.append((tau_hat - tau_true) ** 2)
        # Normalize by number of pairs (standard multi-treatment PEHE)
        return float(np.mean(np.sum(sq_errors, axis=0)) / len(pairs))

    def get_representation(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            Z = self.phi(X_t)
        return Z.cpu().numpy()
