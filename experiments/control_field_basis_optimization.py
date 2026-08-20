from __future__ import annotations

import json
import numpy as np

SEED = 42
POSITIONS = 256
SAMPLES = 240

rng = np.random.default_rng(SEED)
x = np.linspace(0.0, 1.0, POSITIONS)


def generate_target(kind: str | None = None) -> tuple[str, np.ndarray]:
    if kind is None:
        kind = str(rng.choice(["smooth", "mixed", "spiky"], p=[0.4, 0.4, 0.2]))
    y = np.zeros(POSITIONS)
    y += rng.normal(0, 0.15) + rng.normal(0, 0.5) * (x - 0.5)
    for _ in range(int(rng.integers(1, 4))):
        f = int(rng.integers(1, 4))
        y += rng.uniform(0.1, 0.6) * np.sin(2 * np.pi * f * x + rng.uniform(0, 2 * np.pi))
    if kind in {"mixed", "spiky"}:
        for _ in range(int(rng.integers(1, 3))):
            c, w, a = rng.uniform(0.15, 0.85), rng.uniform(0.04, 0.15), rng.uniform(-0.8, 0.8)
            y += a * np.exp(-0.5 * ((x - c) / w) ** 2)
    if kind == "spiky":
        for _ in range(int(rng.integers(1, 4))):
            c, w, a = rng.uniform(0.05, 0.95), rng.uniform(0.005, 0.025), rng.uniform(-1.2, 1.2)
            y += a * np.exp(-0.5 * ((x - c) / w) ** 2)
    return kind, np.clip(y, -1.5, 1.5)


def fourier_basis(n: int) -> np.ndarray:
    cols = [np.ones(POSITIONS)]
    f = 1
    while len(cols) < n:
        cols.append(np.sin(2 * np.pi * f * x))
        if len(cols) < n:
            cols.append(np.cos(2 * np.pi * f * x))
        f += 1
    return np.column_stack(cols[:n])


def linear_basis(n: int) -> np.ndarray:
    knots = np.linspace(0, 1, n)
    basis = np.zeros((POSITIONS, n))
    for row, xx in enumerate(x):
        if xx >= 1:
            basis[row, -1] = 1
            continue
        idx = min(int(xx * (n - 1)), n - 2)
        weight = (xx - knots[idx]) / (knots[idx + 1] - knots[idx])
        basis[row, idx] = 1 - weight
        basis[row, idx + 1] = weight
    return basis


def rbf_basis(n: int) -> np.ndarray:
    centers = np.linspace(0, 1, n)
    sigma = 0.65 / max(n - 1, 1)
    basis = np.exp(-0.5 * ((x[:, None] - centers[None, :]) / sigma) ** 2)
    return basis / (basis.sum(axis=1, keepdims=True) + 1e-12)


def fit(basis: np.ndarray, target: np.ndarray) -> np.ndarray:
    ridge = 1e-5
    coef = np.linalg.solve(basis.T @ basis + ridge * np.eye(basis.shape[1]), basis.T @ target)
    return basis @ coef


def add_local_overrides(pred: np.ndarray, target: np.ndarray, count: int, width: int = 4) -> np.ndarray:
    out = pred.copy()
    for _ in range(count):
        residual = target - out
        idx = int(np.argmax(np.abs(residual)))
        lo, hi = max(0, idx - width), min(POSITIONS, idx + width + 1)
        d = np.abs(np.arange(lo, hi) - idx)
        shape = np.maximum(0, 1 - d / (width + 1))
        amplitude = float((residual[lo:hi] @ shape) / (shape @ shape))
        out[lo:hi] += amplitude * shape
    return out


def main() -> None:
    targets = [generate_target() for _ in range(SAMPLES)]
    families = {"fourier": fourier_basis, "linear": linear_basis, "rbf": rbf_basis}
    rows = []
    for family, builder in families.items():
        for basis_count in (4, 8, 12, 16):
            basis = builder(basis_count)
            for local_count in (0, 1, 2, 4):
                errors = {"smooth": [], "mixed": [], "spiky": []}
                for kind, target in targets:
                    pred = fit(basis, target)
                    if local_count:
                        pred = add_local_overrides(pred, target, local_count)
                    errors[kind].append(float(np.mean((pred - target) ** 2)))
                rows.append({
                    "family": family,
                    "basis": basis_count,
                    "local_overrides": local_count,
                    "control_tokens_equivalent": 1 + local_count,
                    "mse_all": float(np.mean([v for group in errors.values() for v in group])),
                    "mse_smooth": float(np.mean(errors["smooth"])),
                    "mse_mixed": float(np.mean(errors["mixed"])),
                    "mse_spiky": float(np.mean(errors["spiky"])),
                })
    print(json.dumps({"seed": SEED, "samples": SAMPLES, "positions": POSITIONS, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
