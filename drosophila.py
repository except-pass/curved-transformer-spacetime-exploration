"""Drosophila Q/K experiment.
Source paper: https://arxiv.org/html/2511.03060
Core implementation follows scaled dot-product attention plus a residual update.
The flat-control metric is OUR explicitly specified geometry, not the paper's G.
Run: python drosophila.py
"""
from pathlib import Path
import json
import numpy as np

# Our untrained, attention-only "Drosophila":
# 2 tokens, 3 coordinates, 1 causal attention head, 4 residual layers.
SEED, N_TOKENS, WIDTH, N_LAYERS = 42, 2, 3, 4
rng = np.random.default_rng(SEED)
initial_states = rng.normal(size=(N_TOKENS, WIDTH))
weights = [
    {name: rng.normal(size=(WIDTH, WIDTH)) / np.sqrt(WIDTH)
     for name in ("Q", "K", "V", "O")}
    for _ in range(N_LAYERS)
]

def attention_layer(x, w):
    """Rows are token positions. All positions read the same incoming states."""
    q, k, v = x @ w["Q"], x @ w["K"], x @ w["V"]
    g = q @ k.T                         # Paper's token-to-token "effective metric"
    scores = g / np.sqrt(q.shape[1])
    future_tokens = np.triu(np.ones(g.shape, dtype=bool), k=1)
    scores = np.where(future_tokens, -np.inf, scores)
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    attention = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    update = (attention @ v) @ w["O"]
    return x + update, {
        "Q": q, "K": k, "V": v, "G": g,
        "Q_transpose_K": q.T @ k,
        "attention": attention, "update": update,
    }

def forward(x, layers):
    states, ledger = [x.copy()], []
    for w in layers:
        x, details = attention_layer(x, w)
        states.append(x.copy())
        ledger.append(details)
    return np.stack(states), ledger

def path_statistics(points):
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("A zero-length step makes a turning angle undefined.")
    cosines = np.sum(steps[:-1] * steps[1:], axis=1) / (lengths[:-1] * lengths[1:])
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    chord = np.linalg.norm(points[-1] - points[0])
    return angles, lengths.sum() / chord if chord > 1e-12 else float("inf")

states, ledger = forward(initial_states, weights)
angles, ratio = path_statistics(states[:, 1, :])

np.set_printoptions(precision=6, suppress=True)
print(f"Seed: {SEED}; no training; {N_LAYERS * 4 * WIDTH * WIDTH} weight entries; zero biases")
print("\nTOKEN_B's actual hidden states:")
for layer, point in enumerate(states[:, 1, :]):
    print(f"  {layer}: {point}")
print("\nTurning angles (degrees):", angles)
print(f"Path length / endpoint distance: {ratio:.6f}")

print("\nQ/K geometry audit, before each layer:")
for layer, entry in enumerate(ledger, start=1):
    g, alternative = entry["G"], entry["Q_transpose_K"]
    print(f"\nLayer {layer}, G = Q @ K.T:")
    print(g)
    print("  Symmetric?", np.allclose(g, g.T, atol=1e-10, rtol=0))
    print("  Rank:", np.linalg.matrix_rank(g), "of", g.shape[0])
    print("  Eigenvalues of its symmetric part:",
          np.linalg.eigvalsh((g + g.T) / 2))
    print("  Alternative Q.T @ K shape/rank:",
          alternative.shape, np.linalg.matrix_rank(alternative))

output_dir = Path(globals().get("__file__", "drosophila.py")).resolve().parent / "results"
output_dir.mkdir(exist_ok=True)
print("\nCurvature from the paper's literal recipe: NOT WELL-DEFINED in this setup.")
print("This is not a numerical estimate of zero curvature.")

# CONTROL: tie Q = K, using the same projection in every layer.
# This gives a genuine, constant feature-space metric M = W @ W.T.
# IMPORTANT: M is our explicitly chosen control geometry, NOT the paper's G.

shared_qk = weights[0]["Q"].copy()
flat_metric = shared_qk @ shared_qk.T
assert np.linalg.eigvalsh(flat_metric).min() > 0

flat_layers = [
    {"Q": shared_qk.copy(), "K": shared_qk.copy(),
     "V": w["V"].copy(), "O": w["O"].copy()}
    for w in weights
]
flat_states, flat_ledger = forward(initial_states, flat_layers)
flat_angles, flat_ratio = path_statistics(flat_states[:, 1, :])

# The changing token-to-token table is exactly a Gram matrix of the
# current token vectors under this one fixed, flat feature-space ruler.
for x, entry in zip(flat_states[:-1], flat_ledger):
    np.testing.assert_allclose(entry["G"], x @ flat_metric @ x.T, atol=1e-12)

# Measure trajectory bending in the control's own metric, not just Euclidean coordinates.
metric_coordinates = flat_states[:, 1, :] @ shared_qk
metric_angles, metric_ratio = path_statistics(metric_coordinates)

print("CONTROL: same genuine feature-space ruler at every point and every layer")
print("M =\n", flat_metric)
print("M's eigenvalues:", np.linalg.eigvalsh(flat_metric))
print("\nTOKEN_B turning angles, ordinary coordinates:", flat_angles)
print("TOKEN_B turning angles, using M's own ruler:", metric_angles)
print(f"Path / endpoint distance using M: {metric_ratio:.6f}")
print("G = X @ M @ X.T verified at all 4 layers.")
print("The token-to-token G table changes even though M never changes.")

# Check the paper's token-matrix inversion on a slightly longer sequence.
longer_input = np.random.default_rng(314159).normal(size=(4, WIDTH))
_, longer_entry = attention_layer(longer_input, weights[0])
print("\n4-token / 3-coordinate rank check:")
print("G shape:", longer_entry["G"].shape)
print("G rank:", np.linalg.matrix_rank(longer_entry["G"]))
print("G singular values:", np.linalg.svd(longer_entry["G"], compute_uv=False))
print("The ordinary inverse required by the recipe cannot exist exactly.")

# Keep the current X,Q,K identical and reverse only the output projection.
reversed_output = {name: matrix.copy() for name, matrix in weights[0].items()}
reversed_output["O"] *= -1
_, reversed_entry = attention_layer(initial_states, reversed_output)
np.testing.assert_array_equal(ledger[0]["G"], reversed_entry["G"])
np.testing.assert_allclose(ledger[0]["update"], -reversed_entry["update"])
print("\nSame-current-QK intervention: reversing O reverses the entire update.")
print("This shows Q/K alone do not fix motion; it does not refute a theory using V/O too.")

import sympy as sp
from itertools import product

def calculate_curvature(metric, coordinates):
    """Levi-Civita connection and full Riemann tensor for a supplied metric field.

    This does NOT invent a manifold or turn a token-similarity table into a metric.
    The caller must supply coordinates and a symmetric, nondegenerate metric.
    """
    metric = sp.Matrix(metric)
    n = len(coordinates)
    if metric.shape != (n, n):
        raise ValueError(f"{n} coordinates require a {n}×{n} metric; got {metric.shape}.")
    if metric != metric.T:
        raise ValueError("Not symmetric: the standard metric/Levi-Civita recipe does not apply.")
    if metric.det() == 0:
        raise ValueError("Degenerate metric: the required inverse does not exist.")

    inverse = metric.inv()
    gamma = {}
    for a, b, c in product(range(n), repeat=3):
        gamma[a, b, c] = sp.simplify(sum(
            inverse[a, m] * (
                sp.diff(metric[m, c], coordinates[b])
                + sp.diff(metric[m, b], coordinates[c])
                - sp.diff(metric[b, c], coordinates[m])
            ) / 2
            for m in range(n)
        ))

    riemann = {}
    for a, b, c, d in product(range(n), repeat=4):
        expression = (
            sp.diff(gamma[a, d, b], coordinates[c])
            - sp.diff(gamma[a, c, b], coordinates[d])
            + sum(gamma[a, c, m] * gamma[m, d, b]
                  - gamma[a, d, m] * gamma[m, c, b]
                  for m in range(n))
        )
        riemann[a, b, c, d] = sp.trigsimp(sp.simplify(expression))

    ricci = sp.Matrix(n, n, lambda b, d: sum(
        riemann[a, b, a, d] for a in range(n)
    ))
    scalar = sp.trigsimp(sp.simplify(sum(
        inverse[b, d] * ricci[b, d] for b, d in product(range(n), repeat=2)
    )))
    return gamma, riemann, scalar

x, y, z = sp.symbols("x y z", real=True)
flat_gamma, flat_riemann, flat_scalar = calculate_curvature(flat_metric, (x, y, z))
assert all(value == 0 for value in flat_riemann.values())
print("Flat Q=K control: all 81 Riemann-tensor components are EXACTLY zero.")
print("Flat Q=K control: scalar curvature =", flat_scalar)

# Positive control: actual unit sphere, away from its coordinate poles.
theta, phi = sp.symbols("theta phi", real=True)
sphere_metric = sp.diag(1, sp.sin(theta)**2)
_, sphere_riemann, sphere_scalar = calculate_curvature(sphere_metric, (theta, phi))
assert sp.simplify(sphere_scalar - 2) == 0
print("Unit sphere control: scalar curvature =", sphere_scalar)
print("Unit sphere control: Gaussian curvature =", sphere_scalar / 2)

print("\nAttempting the paper's token-to-token G as a hidden-state metric:")
try:
    calculate_curvature(ledger[0]["G"], (x, y, z))
except ValueError as error:
    print("  Rejected:", error)

print("Even pretending its two token labels are two spatial coordinates:")
try:
    calculate_curvature(ledger[0]["G"], (x, y))
except ValueError as error:
    print("  Rejected:", error)

# A modest robustness check, not a replication of the paper's LLM statistics.
# Seeds 0..999 are all used; none are selected based on the outcome.
sweep_rows = []
for seed in range(1000):
    local_rng = np.random.default_rng(seed)
    start = local_rng.normal(size=(N_TOKENS, WIDTH))
    layers = [
        {name: local_rng.normal(size=(WIDTH, WIDTH)) / np.sqrt(WIDTH)
         for name in ("Q", "K", "V", "O")}
        for _ in range(N_LAYERS)
    ]
    trajectory, records = forward(start, layers)
    turn, elongation = path_statistics(trajectory[:, 1, :])
    sweep_rows.append({
        "seed": seed,
        "max_turn_degrees": float(turn.max()),
        "path_chord_ratio": float(elongation),
        "asymmetric_G_count": sum(
            not np.allclose(r["G"], r["G"].T, atol=1e-10, rtol=0)
            for r in records
        ),
    })

bent_count = sum(row["max_turn_degrees"] > 1e-6 for row in sweep_rows)
asymmetric_count = sum(row["asymmetric_G_count"] for row in sweep_rows)
print(f"\nRandom-initialization sweep: {bent_count}/1000 TOKEN_B paths bend.")
print(f"Asymmetric G matrices: {asymmetric_count}/4000 layer evaluations.")
print("Median path / endpoint distance:",
      round(float(np.median([r["path_chord_ratio"] for r in sweep_rows])), 6))

import matplotlib.pyplot as plt

def plot_trajectory(points, title, filename):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(points[:, 0], points[:, 1], points[:, 2], marker="o", linewidth=2)
    for layer, point in enumerate(points):
        ax.text(*point, f"  L{layer}", fontsize=11)
    # Equal coordinate scale on all axes; no dimensionality reduction.
    midpoint = (points.max(axis=0) + points.min(axis=0)) / 2
    radius = np.ptp(points, axis=0).max() * 0.65
    ax.set_xlim(midpoint[0] - radius, midpoint[0] + radius)
    ax.set_ylim(midpoint[1] - radius, midpoint[1] + radius)
    ax.set_zlim(midpoint[2] - radius, midpoint[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("Hidden coordinate 1")
    ax.set_ylabel("Hidden coordinate 2")
    ax.set_zlabel("Hidden coordinate 3")
    ax.set_title(title, pad=16)
    fig.text(0.5, 0.025,
             "Actual 3-D hidden states. L0 = input. Lines only connect sampled states.",
             ha="center", fontsize=10)
    fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.show()

plot_trajectory(
    states[:, 1, :],
    "Random Transformer: TOKEN_B turns without any training",
    "random_trajectory.png",
)
plot_trajectory(
    flat_states[:, 1, :],
    "Control: bent TOKEN_B path, exactly flat specified metric",
    "flat_control_trajectory.png",
)


# Basic implementation checks: causal masking, normalized weights, residuals.
for trajectory, records in [(states, ledger), (flat_states, flat_ledger)]:
    for index, entry in enumerate(records):
        np.testing.assert_allclose(entry["attention"].sum(axis=1), 1.0)
        assert entry["attention"][0, 1] == 0.0
        np.testing.assert_allclose(
            trajectory[index + 1], trajectory[index] + entry["update"]
        )
changed_future = initial_states.copy()
changed_future[1] += np.array([0.1, -0.2, 0.3])
changed_states, _ = forward(changed_future, weights)
np.testing.assert_allclose(changed_states[:, 0], states[:, 0], atol=1e-12)
print("Implementation checks passed: causal mask, softmax, residuals, no future-token leakage.")

import csv
import platform

# Save every input, weight, intermediate attention table, and measured result.
def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value

results = {
    "configuration": {
        "seed": SEED, "tokens": N_TOKENS, "hidden_dimensions": WIDTH,
        "layers": N_LAYERS, "heads": 1, "causal": True,
        "training": False, "biases": "zero", "MLP": False, "normalization": False,
    },
    "initial_states": initial_states,
    "weights": weights,
    "random_model": {
        "states": states, "ledger": ledger,
        "token_B_turning_angles_degrees": angles,
        "token_B_path_chord_ratio": ratio,
        "paper_intrinsic_curvature": None,
        "curvature_status": "Not well-defined by the supplied metric/coordinate recipe",
    },
    "flat_control": {
        "metric_definition": "M = shared_Q @ shared_Q.T, constant on R^3",
        "not_the_papers_G": True,
        "shared_QK": shared_qk, "metric": flat_metric,
        "states": flat_states, "ledger": flat_ledger,
        "euclidean_turning_angles_degrees": flat_angles,
        "metric_turning_angles_degrees": metric_angles,
        "metric_path_chord_ratio": metric_ratio,
        "all_81_riemann_components_exactly_zero": True,
        "scalar_curvature": str(flat_scalar),
    },
    "sphere_positive_control": {
        "metric": "diag(1, sin(theta)^2)",
        "scalar_curvature": str(sphere_scalar),
        "gaussian_curvature": str(sphere_scalar / 2),
    },
    "four_token_rank_check": {
        "input": longer_input, "G": longer_entry["G"],
        "rank": int(np.linalg.matrix_rank(longer_entry["G"])),
    },
    "same_QK_opposite_update_check": True,
    "random_seed_sweep": sweep_rows,
}
(output_dir / "results.json").write_text(
    json.dumps(json_ready(results), indent=2, allow_nan=False), encoding="utf-8"
)

with (output_dir / "trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(["model", "layer", "token", "coordinate_1", "coordinate_2", "coordinate_3"])
    for model_name, trajectory in [("random", states), ("flat_control", flat_states)]:
        for layer in range(trajectory.shape[0]):
            for token in range(trajectory.shape[1]):
                writer.writerow([model_name, layer, token, *trajectory[layer, token]])

