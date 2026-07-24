"""Shared utilities for claim verifiers."""
from __future__ import annotations
import json
import os
import sys
import time
import numpy as np
from datetime import datetime, timezone

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", ".openresearch", "artifacts")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(data: dict, filename: str):
    path = os.path.join(ARTIFACTS_DIR, filename)
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def save_csv(rows: list, header: list, filename: str):
    path = os.path.join(ARTIFACTS_DIR, filename)
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    return path


def save_numpy(arr: np.ndarray, filename: str):
    path = os.path.join(ARTIFACTS_DIR, filename)
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)
    return path


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def system_info() -> dict:
    import platform
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "N/A"
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "torch_version": torch_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = a * x^b via log-linear regression.
    Returns (exponent b, coefficient a, R^2)."""
    mask = (x > 0) & (y > 0)
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])
    if len(log_x) < 2:
        return 0.0, 0.0, 0.0
    coeffs = np.polyfit(log_x, log_y, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])
    y_pred = a * x[mask] ** b
    ss_res = np.sum((y[mask] - y_pred) ** 2)
    ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(b), float(a), float(r2)
