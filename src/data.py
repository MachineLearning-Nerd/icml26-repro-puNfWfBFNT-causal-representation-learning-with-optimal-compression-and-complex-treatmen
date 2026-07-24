"""Synthetic data generation per Appendix D.1 of arXiv 2603.11907.

Hard Setting (Section 4):
  N=1500, d=20, X ~ N(0, I_d), K treatments via high-temperature softmax,
  Y(t) = sin(2*X1) + X3^2 + 0.5*(t+1)*(X_{1:5}^T beta) + eps.

Digits Setting (Section 5.1):
  UCI Digits, K=10 treatments (digit classes), Y(t) = f(X) + (t-4)^2 + eps.

Hierarchical Tree (Section 5.2 / D.4):
  7-node binary tree, outcomes by semantic distance from root.
"""
from __future__ import annotations
import numpy as np


def generate_hard_setting(
    N: int = 1500,
    K: int = 4,
    d: int = 20,
    kappa: float = 5.0,
    seed: int = 42,
) -> dict:
    """Generate the 'Hard Setting' semi-synthetic dataset (Appendix D.1).

    Returns dict with keys: X, T, Y, Y_all (all potential outcomes), beta, W, propensity.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d))

    # Random projection vectors for treatment assignment
    W = rng.uniform(-1.0, 1.0, (K, d))

    # High-temperature softmax propensity
    logits = kappa * (X @ W.T)  # (N, K)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    propensity = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # (N, K)

    # Sample treatments (vectorized)
    cum_prop = np.cumsum(propensity, axis=1)
    u = rng.uniform(size=N)
    T = (cum_prop < u[:, None]).sum(axis=1).astype(int)

    # Outcome generation: Y(t) = sin(2*X1) + X3^2 + 0.5*(t+1)*(X_{1:5}^T beta) + eps
    beta = rng.standard_normal(5)
    eps = rng.standard_normal(N) * np.sqrt(0.1)

    base = np.sin(2.0 * X[:, 0]) + X[:, 2] ** 2
    treatment_effect = 0.5 * (np.arange(K)[None, :] + 1) * (X[:, :5] @ beta)[:, None]  # (N, K)

    # All potential outcomes (ground truth without noise for PEHE)
    Y_all_mean = base[:, None] + treatment_effect  # (N, K)
    Y_all = Y_all_mean + rng.standard_normal((N, K)) * np.sqrt(0.1)

    # Factual outcome
    Y = Y_all[np.arange(N), T]

    return {
        "X": X.astype(np.float64),
        "T": T.astype(int),
        "Y": Y.astype(np.float64),
        "Y_all": Y_all.astype(np.float64),
        "Y_all_mean": Y_all_mean.astype(np.float64),
        "beta": beta,
        "W": W,
        "propensity": propensity,
        "K": K,
        "d": d,
        "kappa": kappa,
    }


def true_ite_hard_setting(data: dict) -> np.ndarray:
    """Compute true pairwise ITEs tau_{j,k}(x) = |Y_mean(x,j) - Y_mean(x,k)|.

    Returns array of shape (N, n_pairs) where n_pairs = C(K,2).
    In the scalar-outcome regime, W2 between point masses = |difference|.
    """
    Y_mean = data["Y_all_mean"]  # (N, K)
    K = data["K"]
    pairs = [(j, k) for j in range(K) for k in range(j + 1, K)]
    taus = np.zeros((Y_mean.shape[0], len(pairs)))
    for idx, (j, k) in enumerate(pairs):
        taus[:, idx] = np.abs(Y_mean[:, j] - Y_mean[:, k])
    return taus


def generate_tree_data(
    n_per_leaf: int = 200,
    seed: int = 42,
) -> dict:
    """Generate hierarchical tree data (Appendix D.4).

    7-node binary tree: Root(0) -> L(1), R(2) -> LL(3), LR(4), RL(5), RR(6).
    Outcomes: Root=0, L=-2, LL=-3, LR=-1, R=+2, RL=+1, RR=+3.
    """
    rng = np.random.default_rng(seed)
    K = 7
    # Tree structure: parent[i] = parent node index
    tree_adj = {
        0: [1, 2],   # Root -> L, R
        1: [3, 4],   # L -> LL, LR
        2: [5, 6],   # R -> RL, RR
    }
    # Shortest path distances on tree
    import networkx as nx_ref  # only for reference; we compute manually
    # Compute geodesic (shortest path) distances manually
    node_effects = {0: 0.0, 1: -2.0, 2: 2.0, 3: -3.0, 4: -1.0, 5: 1.0, 6: 3.0}

    # BFS shortest path distances
    from collections import deque
    geo_dist = np.zeros((K, K))
    for src in range(K):
        visited = {src: 0}
        queue = deque([(src, 0)])
        while queue:
            node, dist = queue.popleft()
            for parent, children in tree_adj.items():
                if node in children and parent not in visited:
                    visited[parent] = dist + 1
                    queue.append((parent, dist + 1))
                if node != 0 and node in children:
                    # find siblings
                    for c in children:
                        if c != node and c not in visited:
                            visited[c] = dist + 2  # via parent
                            queue.append((c, dist + 2))
            # also check if node is a parent
            if node in tree_adj:
                for c in tree_adj[node]:
                    if c not in visited:
                        visited[c] = dist + 1
                        queue.append((c, dist + 1))
        for dst in range(K):
            geo_dist[src, dst] = visited.get(dst, 99)

    # Generate samples
    X_list, T_list, Y_list = [], [], []
    for t in range(K):
        n = n_per_leaf
        X_t = rng.standard_normal((n, 10))
        Y_t = node_effects[t] + rng.standard_normal(n) * 0.3
        X_list.append(X_t)
        T_list.append(np.full(n, t))
        Y_list.append(Y_t)

    return {
        "X": np.vstack(X_list),
        "T": np.concatenate(T_list),
        "Y": np.concatenate(Y_list),
        "K": K,
        "node_effects": node_effects,
        "geo_dist": geo_dist,
        "tree_adj": tree_adj,
    }


def generate_cyclic_data(
    K: int = 8,
    n_per_t: int = 200,
    seed: int = 42,
) -> dict:
    """Generate cyclic topology data (Appendix D.5).

    K=8 treatments at angles 0, 45, ..., 315 degrees.
    Y = cos(theta) + eps.
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)  # 0, 45, ..., 315 deg
    # Geodesic distances on cycle C_K
    geo_dist = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            diff = abs(i - j)
            geo_dist[i, j] = min(diff, K - diff)

    X_list, T_list, Y_list = [], [], []
    for t in range(K):
        X_t = rng.standard_normal((n_per_t, 10))
        Y_t = np.cos(angles[t]) + rng.standard_normal(n_per_t) * 0.1
        X_list.append(X_t)
        T_list.append(np.full(n_per_t, t))
        Y_list.append(Y_t)

    return {
        "X": np.vstack(X_list),
        "T": np.concatenate(T_list),
        "Y": np.concatenate(Y_list),
        "K": K,
        "angles": angles,
        "geo_dist": geo_dist,
    }
