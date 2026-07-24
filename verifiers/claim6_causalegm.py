"""Claim 6: Multi-Treatment CausalEGM preserves Wasserstein geodesic structure.

Claim (Section 5.1, 5.2, Figure 3):
  Multi-Treatment CausalEGM extends the discriminative framework to high-dimensional
  counterfactual generation and, via interpolation experiments, preserves the
  Wasserstein geodesic structure of the treatment manifold.

Verification:
  A. Hierarchical tree (D.4): interpolation LL->RR passes through Root effect (Y~0)
  B. Cyclic topology (D.5): 0deg and 315deg treated as neighbors
  C. Negative control: linear (Euclidean) interpolation does not respect topology
"""
from __future__ import annotations
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.causalegm import GeodesicCausalEGM
from verifiers.common import save_json, save_csv, log, system_info


def compute_tree_geo_dist(K: int = 7) -> np.ndarray:
    """Compute shortest-path distances on the 7-node binary tree.

    Tree: Root(0) -> L(1), R(2); L(1)->LL(3),LR(4); R(2)->RL(5),RR(6)
    """
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    adj = {i: [] for i in range(K)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # Floyd-Warshall
    INF = 999
    dist = np.full((K, K), INF)
    np.fill_diagonal(dist, 0)
    for a, b in edges:
        dist[a, b] = 1
        dist[b, a] = 1
    for k in range(K):
        for i in range(K):
            for j in range(K):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    return dist.astype(float)


def generate_tree_data_fixed(K: int = 7, n_per_leaf: int = 200, seed: int = 42) -> dict:
    """Generate hierarchical tree data with correct structure."""
    rng = np.random.default_rng(seed)
    node_effects = {0: 0.0, 1: -2.0, 2: 2.0, 3: -3.0, 4: -1.0, 5: 1.0, 6: 3.0}
    geo_dist = compute_tree_geo_dist(K)

    X_list, T_list, Y_list = [], [], []
    for t in range(K):
        X_t = rng.standard_normal((n_per_leaf, 10))
        Y_t = node_effects[t] + rng.standard_normal(n_per_leaf) * 0.3
        X_list.append(X_t)
        T_list.append(np.full(n_per_leaf, t))
        Y_list.append(Y_t)

    return {
        "X": np.vstack(X_list),
        "T": np.concatenate(T_list),
        "Y": np.concatenate(Y_list),
        "K": K,
        "node_effects": node_effects,
        "geo_dist": geo_dist,
    }


def generate_cyclic_data_fixed(K: int = 8, n_per_t: int = 200, seed: int = 42) -> dict:
    """Generate cyclic topology data."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
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


def verify_tree_interpolation() -> dict:
    """Test hierarchical tree geodesic interpolation."""
    log("  Training GeodesicCausalEGM on hierarchical tree...")
    data = generate_tree_data_fixed(K=7, n_per_leaf=200, seed=42)

    model = GeodesicCausalEGM(
        input_dim=10, K=7, geo_dist=data["geo_dist"],
        d_e=2, lambda_geo=5.0, epochs=500, seed=42,
    )
    model.fit(data["X"], data["T"], data["Y"])

    # Interpolate from LL(3) to RR(6)
    interp = model.interpolate(data["X"], t_A=3, t_B=6, n_steps=101)
    alphas = np.linspace(0, 1, 101)

    # Linear baseline: Y = -3 + 6*alpha (straight line from -3 to +3)
    linear = -3.0 + 6.0 * alphas

    # Check midpoint: geodesic Y(alpha=0.5) should be ~0 (Root effect)
    geo_midpoint = interp[50]
    lin_midpoint = linear[50]

    # Check sigmoidal shape: the geodesic should have higher curvature
    # Measure: variance of second derivative
    geo_2nd_deriv = np.diff(interp, 2)
    lin_2nd_deriv = np.diff(linear, 2)
    geo_curvature = float(np.var(geo_2nd_deriv))
    lin_curvature = float(np.var(lin_2nd_deriv))

    # Check embedding: midpoint should be close to Root embedding
    emb = model.get_embeddings()
    e_mid = 0.5 * emb[3] + 0.5 * emb[6]
    dist_to_root = float(np.linalg.norm(e_mid - emb[0]))
    dist_to_all = [float(np.linalg.norm(e_mid - emb[i])) for i in range(7)]
    closest_node = int(np.argmin(dist_to_all))

    # Check that Root is placed centrally
    root_distances = [float(np.linalg.norm(emb[0] - emb[i])) for i in range(7)]
    root_mean_dist = float(np.mean(root_distances))

    # Geodesic loss value
    emb_dist = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            emb_dist[i, j] = np.linalg.norm(emb[i] - emb[j])
    geo_mse = float(np.mean((emb_dist - data["geo_dist"]) ** 2))

    return {
        "interp_values": interp.tolist(),
        "linear_baseline": linear.tolist(),
        "geo_midpoint_y": float(geo_midpoint),
        "lin_midpoint_y": float(lin_midpoint),
        "geo_curvature": geo_curvature,
        "lin_curvature": lin_curvature,
        "geo_has_more_curvature": geo_curvature > lin_curvature * 2,
        "embedding_distances_to_midpoint": dist_to_all,
        "closest_node_to_midpoint": closest_node,
        "closest_is_root": closest_node == 0,
        "dist_midpoint_to_root": dist_to_root,
        "geodesic_loss_mse": geo_mse,
        "embeddings": emb.tolist(),
        "root_mean_distance": root_mean_dist,
    }


def verify_cyclic_interpolation() -> dict:
    """Test cyclic topology geodesic interpolation."""
    log("  Training GeodesicCausalEGM on cyclic topology...")
    data = generate_cyclic_data_fixed(K=8, n_per_t=200, seed=42)

    model = GeodesicCausalEGM(
        input_dim=10, K=8, geo_dist=data["geo_dist"],
        d_e=2, lambda_geo=5.0, epochs=500, seed=42,
    )
    model.fit(data["X"], data["T"], data["Y"])

    emb = model.get_embeddings()

    # Key test: 0deg (node 0) and 315deg (node 7) should be neighbors in embedding
    dist_0_7 = float(np.linalg.norm(emb[0] - emb[7]))
    dist_0_4 = float(np.linalg.norm(emb[0] - emb[4]))  # 0 and 180 (opposite)
    neighbors = dist_0_7 < dist_0_4  # 0-315 should be closer than 0-180

    # Interpolation 0 -> 7 (should be smooth, short-range)
    interp_0_7 = model.interpolate(data["X"], t_A=0, t_B=7, n_steps=51)
    # Interpolation 0 -> 4 (should go through full range)
    interp_0_4 = model.interpolate(data["X"], t_A=0, t_B=4, n_steps=51)

    # Check smoothness of boundary interpolation
    range_0_7 = float(np.max(interp_0_7) - np.min(interp_0_7))
    range_0_4 = float(np.max(interp_0_4) - np.min(interp_0_4))

    # Check embedding forms a ring
    # Compute pairwise embedding distances
    emb_dist = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            emb_dist[i, j] = np.linalg.norm(emb[i] - emb[j])

    # Correlation between embedding distances and geodesic distances
    from scipy.stats import spearmanr
    geo_flat = data["geo_dist"].flatten()
    emb_flat = emb_dist.flatten()
    corr, p_val = spearmanr(geo_flat, emb_flat)

    return {
        "dist_0_7": dist_0_7,
        "dist_0_4": dist_0_4,
        "are_neighbors": neighbors,
        "interp_0_7_range": range_0_7,
        "interp_0_4_range": range_0_4,
        "boundary_smooth": range_0_7 < range_0_4 * 0.5,
        "embedding_geodesic_correlation": float(corr),
        "correlation_p_value": float(p_val),
        "embeddings": emb.tolist(),
        "interp_0_7": interp_0_7.tolist(),
        "interp_0_4": interp_0_4.tolist(),
    }


def run() -> dict:
    log("=== Claim 6: CausalEGM Wasserstein Geodesic Interpolation ===")
    t_start = time.perf_counter()

    log("Part A: Hierarchical tree topology")
    tree_result = verify_tree_interpolation()
    log(f"  Geodesic midpoint Y: {tree_result['geo_midpoint_y']:.4f} (target ~0)")
    log(f"  Closest node to midpoint: {tree_result['closest_node_to_midpoint']} (target: 0=Root)")
    log(f"  Geodesic curvature: {tree_result['geo_curvature']:.6f} vs linear: {tree_result['lin_curvature']:.6f}")
    log(f"  Geodesic loss MSE: {tree_result['geodesic_loss_mse']:.4f}")

    log("Part B: Cyclic topology")
    cyclic_result = verify_cyclic_interpolation()
    log(f"  dist(0,7)={cyclic_result['dist_0_7']:.4f} vs dist(0,4)={cyclic_result['dist_0_4']:.4f}")
    log(f"  Are 0-315 neighbors: {cyclic_result['are_neighbors']}")
    log(f"  Embedding-geodesic correlation: {cyclic_result['embedding_geodesic_correlation']:.4f}")

    # Determine verdict
    tree_ok = (
        abs(tree_result["geo_midpoint_y"]) < 1.0 and  # midpoint ~0
        tree_result["closest_is_root"] and  # midpoint maps to Root
        tree_result["geo_curvature"] > tree_result["lin_curvature"]  # sigmoidal
    )
    cyclic_ok = (
        cyclic_result["are_neighbors"] and  # 0-315 are neighbors
        cyclic_result["embedding_geodesic_correlation"] > 0.7  # topology recovered
    )

    verified = tree_ok and cyclic_ok
    verdict = "VERIFIED" if verified else "BLOCKED"

    elapsed = time.perf_counter() - t_start
    result = {
        "claim": "CausalEGM preserves Wasserstein geodesic structure",
        "claim_text": "Interpolation follows geodesic paths, preserving treatment manifold topology",
        "verdict": verdict,
        "tree_result": tree_result,
        "cyclic_result": cyclic_result,
        "tree_ok": tree_ok,
        "cyclic_ok": cyclic_ok,
        "elapsed_seconds": elapsed,
        "system_info": system_info(),
    }

    save_json(result, "claim6_causalegm/result.json")
    # Save interpolation curves
    save_csv(
        [(i / 100, tree_result["interp_values"][i], tree_result["linear_baseline"][i])
         for i in range(101)],
        ["alpha", "geodesic_Y", "linear_Y"],
        "claim6_causalegm/tree_interpolation.csv",
    )

    log(f"Verdict: {verdict} (elapsed {elapsed:.1f}s)")
    return result
