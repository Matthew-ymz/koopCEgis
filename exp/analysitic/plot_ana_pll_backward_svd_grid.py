from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
import tempfile
import zlib

mpl_config_dir = Path(tempfile.gettempdir()) / "koopcegis-matplotlib"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

FEATURE_NAMES = ["x", "y", r"$x^2$"]
SPECTRUM_LOWER_FACTOR = 2.5
SPECTRUM_UPPER_FACTOR = 1.45
SPECTRUM_BAR_WIDTH = 0.52
WEAK_PROCESS_NOISE = 0.0002
MAX_PROCESS_NOISE = 0.2


def make_step_system_matrix(lam: float, mu: float) -> np.ndarray:
    return np.array(
        [
            [lam, 0.0, 0.0],
            [0.0, mu, lam**2 - mu],
            [0.0, 0.0, lam**2],
        ],
        dtype=float,
    )


def make_manual_sigma_matrix(a: float, b: float) -> np.ndarray:
    sigma = np.array(
        [
            [a, 0.0, b],
            [0.0, a, 0.0],
            [b, 0.0, a],
        ],
        dtype=float,
    )
    return 0.5 * (sigma + sigma.T)


@dataclass(frozen=True)
class MatrixCombo:
    label: str
    lam: float
    mu: float
    a: float = 0.0
    b: float = 0.0
    process_noise: float = 0.0
    mode: str = "backward"


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    title: str
    combos: tuple[MatrixCombo, ...]
    plot_evd: bool = False
    share_phase_limits: bool = False


REFERENCE_LAM_MU_PAIRS: tuple[tuple[float, float], ...] = tuple(
    sorted(
        (
            (20.0, 200.0),
            (20.0, 0.3),
            (20.0, 2.0),
            (5.0, 50.0),
            (0.5, 5.0),
            (0.5, 0.1),
            (0.1, 0.0001),
            (0.1, 0.05),
            (0.1, 0.9),
            (1.0, 0.3),
            (1.0, 2.6),
            (3.0, 1.0),
            (0.3, 1.0),
        ),
        key=lambda pair: (pair[0], pair[1]),
    )
)

DEFAULT_A_VALUES: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_B_RATIOS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.6, 0.85, 0.98)
DEFAULT_MU_SWEEPS: tuple[tuple[float, float, float, tuple[float, ...]], ...] = (
    (0.10, 1.0, 0.2, (0.0001, 0.001, 0.01, 0.05, 0.1, 0.3, 0.9)),
    (0.50, 1.0, 0.2, (0.01, 0.1, 0.3, 1.0, 2.0, 5.0, 20.0)),
    (1.00, 10.0, 1.0, (0.01, 0.3, 1.0, 2.6, 10.0, 50.0, 200.0)),
    (20.00, 100.0, 20.0, (0.3, 2.0, 20.0, 50.0, 200.0, 500.0, 1000.0)),
)
DEFAULT_LAM_SWEEPS: tuple[tuple[float, float, float, tuple[float, ...]], ...] = (
    (0.30, 1.0, 0.2, (0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 20.0)),
    (1.00, 1.0, 0.2, (0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 20.0)),
    (5.00, 10.0, 1.0, (0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 20.0)),
    (50.00, 10.0, 1.0, (0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0)),
    (200.00, 100.0, 20.0, (0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0)),
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.8,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 6.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "axes.unicode_minus": False,
        }
    )


def format_param(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 100 or abs(value) < 0.01:
        return f"{value:.1e}"
    return f"{value:.3g}"


def slug_number(value: float) -> str:
    text = f"{value:.6g}"
    text = text.replace("-", "m").replace(".", "p").replace("+", "")
    text = re.sub(r"e([mp]?)", r"e\1", text)
    return text


def fixed_slug(**kwargs: float) -> str:
    return "_".join(f"{key}{slug_number(value)}" for key, value in kwargs.items())


def noise_scale_for_pair(lam: float, mu: float) -> float:
    """Choose a coarse noise scale matched to the largest system parameter."""
    scale = max(abs(lam), abs(mu), 1.0)
    if scale >= 100.0:
        return 100.0
    if scale >= 10.0:
        return 10.0
    return 1.0


def process_noise_from_sigma_a(a: float) -> float:
    """Use a bounded process-noise scale for noisy phase portraits."""
    if a <= 0:
        return 0.0
    return float(np.clip(0.02 * np.sqrt(a), WEAK_PROCESS_NOISE, MAX_PROCESS_NOISE))


def process_noise_scan(n_values: int) -> tuple[float, ...]:
    """Match the senior noise-grid style with a geometric process-noise scan."""
    return tuple(float(value) for value in np.geomspace(WEAK_PROCESS_NOISE, MAX_PROCESS_NOISE, n_values))


def validate_sigma_params(a: float, b: float) -> None:
    if a <= 0:
        raise ValueError("a must be positive for the backward matrix")
    if b < 0:
        raise ValueError("b must be nonnegative")
    if not b < a:
        raise ValueError("b must be smaller than a so Sigma is positive definite")


def fix_vector_signs(vectors: np.ndarray) -> np.ndarray:
    vectors = np.array(np.real_if_close(vectors), dtype=float, copy=True)
    for col in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, col])))
        if vectors[pivot, col] < 0:
            vectors[:, col] *= -1.0
    return vectors


def sorted_eigendecomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    if np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
        values, vectors = np.linalg.eigh(matrix)
    else:
        values, vectors = np.linalg.eig(matrix)
    values = np.real_if_close(values)
    vectors = np.real_if_close(vectors)
    order = np.argsort(-np.abs(values.astype(float)))
    values = values[order].astype(float)
    vectors = vectors[:, order].astype(float)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=0, keepdims=True), 1e-12)
    return values, fix_vector_signs(vectors)


def analyze_combo(combo: MatrixCombo) -> dict[str, object]:
    A = make_step_system_matrix(combo.lam, combo.mu)
    trajectories = simulate_clipped_trajectories(
        lam=combo.lam,
        mu=combo.mu,
        process_noise=combo.process_noise,
        seed=stable_combo_seed(combo),
    )
    if combo.mode == "A":
        matrix = A
        matrix_label = "A"
        U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
        vectors = fix_vector_signs(Vt.T)
        vector_label = "Right singular vectors V"
        sigma = None
    elif combo.mode == "backward":
        validate_sigma_params(combo.a, combo.b)
        sigma = make_manual_sigma_matrix(combo.a, combo.b)
        sigma_inv = np.linalg.pinv(sigma, rcond=1e-12)
        matrix = A.T @ sigma_inv @ A
        matrix = 0.5 * (matrix + matrix.T)
        matrix_label = r"$A^T\Sigma^{-1}A$"
        U, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
        vectors = fix_vector_signs(U)
        vector_label = "Left singular vectors U"
    else:
        raise ValueError(f"Unsupported combo mode: {combo.mode}")

    eigenvalues, eigenvectors = sorted_eigendecomposition(matrix)
    return {
        "combo": combo,
        "trajectories": trajectories,
        "A": A,
        "Sigma": sigma,
        "matrix": matrix,
        "matrix_label": matrix_label,
        "singular_values": singular_values,
        "sv_vectors": vectors,
        "sv_vector_label": vector_label,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
    }


def step1_next(state: np.ndarray, lam: float, mu: float) -> np.ndarray:
    x, y = state
    return np.array([lam * x, mu * y + (lam**2 - mu) * x**2], dtype=float)


def stable_combo_seed(combo: MatrixCombo) -> int:
    seed_text = (
        f"{combo.label}|{combo.lam:.12g}|{combo.mu:.12g}|"
        f"{combo.a:.12g}|{combo.b:.12g}|{combo.process_noise:.12g}|{combo.mode}"
    )
    return zlib.crc32(seed_text.encode("utf-8")) & 0xFFFFFFFF


def simulate_clipped_trajectories(
    lam: float,
    mu: float,
    process_noise: float = 0.0,
    seed: int | None = None,
    steps: int = 90,
    phase_clip: float = 30.0,
) -> list[np.ndarray]:
    if process_noise < 0:
        raise ValueError("process_noise must be nonnegative")

    initial_states = [(x, y) for y in (-0.6, 0.0, 0.6) for x in (-1.1, 0.0, 1.1)]
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(len(initial_states))
    trajectories = []
    for initial, child_sequence in zip(initial_states, child_sequences):
        rng = np.random.default_rng(child_sequence)
        state = np.array(initial, dtype=float)
        points = [state.copy()]
        for _ in range(steps):
            state = step1_next(state, lam=lam, mu=mu)
            if process_noise > 0:
                state = state + rng.normal(loc=0.0, scale=process_noise, size=2)
            if not np.all(np.isfinite(state)):
                break
            if np.max(np.abs(state)) > phase_clip:
                points.append(np.clip(state, -phase_clip, phase_clip))
                break
            points.append(state.copy())
        trajectories.append(np.asarray(points, dtype=float))
    return trajectories


def auto_phase_limits(trajectories: list[np.ndarray], phase_clip: float = 30.0) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.vstack([trajectory for trajectory in trajectories if len(trajectory)])
    points = points[np.all(np.isfinite(points), axis=1)]
    points = np.clip(points, -phase_clip, phase_clip)
    if len(points) == 0:
        return (-1.0, 1.0), (-1.0, 1.0)
    x_min, x_max = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
    y_min, y_max = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
    x_pad = max(0.08, 0.08 * (x_max - x_min))
    y_pad = max(0.08, 0.08 * (y_max - y_min))
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def draw_phase_panel(
    ax: plt.Axes,
    trajectories: list[np.ndarray],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    colors = mpl.colormaps["viridis"](np.linspace(0.12, 0.88, len(trajectories)))
    for trajectory, color in zip(trajectories, colors):
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.75, alpha=0.82)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], s=8, facecolor="white", edgecolor=color, linewidth=0.5, zorder=3)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], s=8, marker="x", color=color, linewidth=0.6, zorder=3)
    ax.axhline(0.0, color="0.84", linewidth=0.45, zorder=0)
    ax.axvline(0.0, color="0.84", linewidth=0.45, zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # Keep every phase panel the same outer size. Very anisotropic trajectories
    # are allowed to look like strips inside the panel.
    ax.set_aspect("auto", adjustable="box")
    ax.set_box_aspect(1.0)
    ax.set_xlabel(r"$x_k$")


def spectrum_limits(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    positive = np.abs(values[np.abs(values) > 0])
    if positive.size == 0:
        return (1e-12, 1.0)
    upper = float(positive.max()) * SPECTRUM_UPPER_FACTOR
    lower = float(positive.min()) / SPECTRUM_LOWER_FACTOR
    if np.any(values < 0):
        return (-upper, upper)
    return (lower, upper)


def draw_spectrum_panel(ax: plt.Axes, values: np.ndarray, color: str, prefix: str) -> None:
    values = np.asarray(values, dtype=float)
    x = np.arange(1, len(values) + 1)
    ax.bar(x, values, color=color, edgecolor="0.25", linewidth=0.35, width=SPECTRUM_BAR_WIDTH)
    ax.plot(x, values, color="0.25", linewidth=0.65, marker="o", markersize=2.1)
    if np.all(values > 0):
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_ylim(spectrum_limits(values))
    ax.set_xticks(x, [rf"${prefix}_{i}$" for i in x])
    ax.grid(axis="y", color="0.90", linewidth=0.4)


def draw_vector_panel(ax: plt.Axes, vectors: np.ndarray, prefix: str) -> mpl.image.AxesImage:
    vectors = np.asarray(vectors, dtype=float)
    image = ax.imshow(vectors, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(vectors.shape[1]), [rf"${prefix}_{i + 1}$" for i in range(vectors.shape[1])])
    ax.set_yticks(np.arange(len(FEATURE_NAMES)), FEATURE_NAMES)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(vectors.shape[0]):
        for col in range(vectors.shape[1]):
            value = vectors[row, col]
            text_color = "white" if abs(value) > 0.62 else "0.18"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=5.0, color=text_color)
    return image


def combo_title(combo: MatrixCombo) -> str:
    parts = [rf"$\lambda={format_param(combo.lam)}$", rf"$\mu={format_param(combo.mu)}$"]
    if combo.mode == "backward":
        parts.extend(
            [
                rf"$a={format_param(combo.a)}$",
                rf"$b={format_param(combo.b)}$",
                rf"$p={format_param(combo.process_noise)}$",
            ]
        )
    else:
        parts.append("baseline A")
    return "\n".join([f"{combo.label}: " + ", ".join(parts[:2]), ", ".join(parts[2:])])


def plot_experiment(spec: ExperimentSpec, output_dir: Path) -> tuple[Path, pd.DataFrame]:
    results = [analyze_combo(combo) for combo in spec.combos]
    if spec.share_phase_limits:
        phase_trajectories = [trajectory for result in results for trajectory in result["trajectories"]]
        phase_limits = [auto_phase_limits(phase_trajectories)] * len(results)
    else:
        phase_limits = [auto_phase_limits(result["trajectories"]) for result in results]
    n_cols = len(results)
    n_rows = 5 if spec.plot_evd else 3
    fig_height = 8.4 if spec.plot_evd else 5.45
    height_ratios = (1.12, 0.62, 0.76, 0.62, 0.76) if spec.plot_evd else (1.12, 0.62, 0.76)
    fig = plt.figure(figsize=(max(10.5, 1.95 * n_cols), fig_height), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=height_ratios, hspace=0.08, wspace=0.18)

    vector_axes = []
    vector_image = None
    for col, result in enumerate(results):
        combo = result["combo"]
        ax_phase = fig.add_subplot(gs[0, col])
        ax_svd = fig.add_subplot(gs[1, col])
        ax_sv_vec = fig.add_subplot(gs[2, col])

        xlim, ylim = phase_limits[col]
        draw_phase_panel(ax_phase, result["trajectories"], xlim=xlim, ylim=ylim)
        sv_prefix = "v" if combo.mode == "A" else "u"
        draw_spectrum_panel(ax_svd, result["singular_values"], color="#2f6f8f", prefix="s")
        vector_image = draw_vector_panel(ax_sv_vec, result["sv_vectors"], prefix=sv_prefix)
        vector_axes.append(ax_sv_vec)

        if spec.plot_evd:
            ax_eig = fig.add_subplot(gs[3, col])
            ax_eig_vec = fig.add_subplot(gs[4, col])
            draw_spectrum_panel(ax_eig, result["eigenvalues"], color="#8aa9b6", prefix="e")
            draw_vector_panel(ax_eig_vec, result["eigenvectors"], prefix="q")
            vector_axes.append(ax_eig_vec)
        else:
            ax_eig = None
            ax_eig_vec = None

        ax_phase.set_title(combo_title(combo), loc="left", pad=4)
        if col == 0:
            ax_phase.set_ylabel(r"$y_k$")
            ax_svd.set_ylabel("SVD")
            ax_sv_vec.set_ylabel("Right SV" if combo.mode == "A" else "Left SV")
            if spec.plot_evd:
                ax_eig.set_ylabel("Eigen")
                ax_eig_vec.set_ylabel("Eig vec")
        else:
            ax_phase.tick_params(labelleft=False)
            ax_sv_vec.tick_params(labelleft=False)
            if spec.plot_evd:
                ax_eig_vec.tick_params(labelleft=False)

    if vector_image is not None:
        cbar = fig.colorbar(vector_image, ax=vector_axes, location="right", fraction=0.018, pad=0.01)
        cbar.set_label("Loading", rotation=90, labelpad=5)
        cbar.outline.set_linewidth(0.45)
        cbar.ax.tick_params(width=0.45, length=2.0, labelsize=5.8)

    fig.suptitle(spec.title, x=0.012, y=1.004, ha="left", va="bottom", fontsize=8.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{spec.key}.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=220)
    plt.close(fig)

    rows = []
    for result in results:
        combo = result["combo"]
        row = {
            "figure": spec.key,
            "combo": combo.label,
            "mode": combo.mode,
            "lambda": combo.lam,
            "mu": combo.mu,
            "a": combo.a,
            "b": combo.b,
            "process_noise": combo.process_noise,
        }
        for idx, value in enumerate(result["singular_values"], start=1):
            row[f"sv{idx}"] = float(value)
        for idx, value in enumerate(result["eigenvalues"], start=1):
            row[f"eig{idx}"] = float(value)
        rows.append(row)
    return output_path, pd.DataFrame(rows)


def make_experiment_specs(
    reference_lam_mu_pairs=REFERENCE_LAM_MU_PAIRS,
    a_values=DEFAULT_A_VALUES,
    b_ratios=DEFAULT_B_RATIOS,
    mu_sweeps=DEFAULT_MU_SWEEPS,
    lam_sweeps=DEFAULT_LAM_SWEEPS,
) -> list[ExperimentSpec]:
    reference_lam_mu_pairs = tuple(reference_lam_mu_pairs)
    a_values = tuple(a_values)
    b_ratios = tuple(b_ratios)
    mu_sweeps = tuple(mu_sweeps)
    lam_sweeps = tuple(lam_sweeps)

    baseline_combos = tuple(
        MatrixCombo(f"P{i:02d}", lam, mu, mode="A")
        for i, (lam, mu) in enumerate(reference_lam_mu_pairs, start=1)
    )

    specs = []
    for part_index, start in enumerate(range(0, len(baseline_combos), 7), start=1):
        part = baseline_combos[start : start + 7]
        specs.append(
            ExperimentSpec(
                key=f"case1_baseline_A_reference_lam_mu_part{part_index}",
                title=f"Case 1.{part_index}: no-noise baseline, direct SVD/EVD of A",
                combos=part,
                plot_evd=True,
            )
        )

    a_process_noises = process_noise_scan(len(a_values))
    for lam, mu in reference_lam_mu_pairs:
        specs.append(
            ExperimentSpec(
                key=f"case2_b0_scan_a_{fixed_slug(lam=lam, mu=mu)}",
                title=rf"Case 2: fixed $\lambda,\mu$, b=0, scan a ({fixed_slug(lam=lam, mu=mu)})",
                combos=tuple(
                    MatrixCombo(f"A{i}", lam, mu, a=value, b=0.0, process_noise=process_noise)
                    for i, (value, process_noise) in enumerate(zip(a_values, a_process_noises), start=1)
                ),
                share_phase_limits=True,
            )
        )

    for lam, mu in reference_lam_mu_pairs:
        a = noise_scale_for_pair(lam, mu)
        process_noise = process_noise_from_sigma_a(a)
        specs.append(
            ExperimentSpec(
                key=f"case3_scan_b_{fixed_slug(lam=lam, mu=mu, a=a)}",
                title=rf"Case 3: fixed $\lambda,\mu,a$, scan b ({fixed_slug(lam=lam, mu=mu, a=a)})",
                combos=tuple(
                    MatrixCombo(f"B{i}", lam, mu, a=a, b=a * ratio, process_noise=process_noise)
                    for i, ratio in enumerate(b_ratios, start=1)
                ),
                share_phase_limits=True,
            )
        )

    for lam, a, b, mu_values in mu_sweeps:
        process_noise = process_noise_from_sigma_a(a)
        specs.append(
            ExperimentSpec(
                key=f"case4_scan_mu_{fixed_slug(lam=lam, a=a, b=b)}",
                title=rf"Case 4: fixed $\lambda,a,b$, scan $\mu$ ({fixed_slug(lam=lam, a=a, b=b)})",
                combos=tuple(
                    MatrixCombo(f"M{i}", lam, mu, a=a, b=b, process_noise=process_noise)
                    for i, mu in enumerate(mu_values, start=1)
                ),
            )
        )

    for mu, a, b, lam_values in lam_sweeps:
        process_noise = process_noise_from_sigma_a(a)
        specs.append(
            ExperimentSpec(
                key=f"case5_scan_lam_{fixed_slug(mu=mu, a=a, b=b)}",
                title=rf"Case 5: fixed $\mu,a,b$, scan $\lambda$ ({fixed_slug(mu=mu, a=a, b=b)})",
                combos=tuple(
                    MatrixCombo(f"L{i}", lam, mu, a=a, b=b, process_noise=process_noise)
                    for i, lam in enumerate(lam_values, start=1)
                ),
            )
        )
    return specs


def run_experiments(output_dir: str | Path, **spec_kwargs) -> tuple[list[Path], pd.DataFrame]:
    configure_style()
    output_dir = Path(output_dir)
    paths = []
    summary_frames = []
    for spec in make_experiment_specs(**spec_kwargs):
        path, summary = plot_experiment(spec, output_dir=output_dir)
        paths.append(path)
        summary_frames.append(summary)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_path = output_dir / "case_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    paths.append(summary_path)
    return paths, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ana_pll step1 matrix-analysis grids.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/analysitic/figs/ana_pll"),
        help="Directory for generated PNG grids and the summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths, _ = run_experiments(output_dir=args.output_dir)
    print("Generated ana_pll outputs:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
