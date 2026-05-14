from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
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


FEATURE_NAMES = ["x", "y", r"$y^2$"]
DEFAULT_SIGMA_COUPLING = "y_y2"
SPECTRUM_LOWER_FACTOR = 2.5
SPECTRUM_UPPER_FACTOR = 1.45
SPECTRUM_BAR_WIDTH = 0.52
WEAK_PROCESS_NOISE = 0.0002
MAX_PROCESS_NOISE = 0.2


@dataclass(frozen=True)
class Step2Combo:
    label: str
    rho: float
    kappa: float
    sigma_a: float = 0.0
    sigma_b: float = 0.0
    process_noise: float = 0.0
    mode: str = "backward"
    sigma_coupling: str = DEFAULT_SIGMA_COUPLING


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    title: str
    combos: tuple[Step2Combo, ...]
    plot_evd: bool = False
    share_phase_limits: bool = False


REFERENCE_RHO_KAPPA_PAIRS: tuple[tuple[float, float], ...] = tuple(
    sorted(
        (
            (0.05, 0.0),
            (0.05, 1.0),
            (0.05, 100.0),
            (0.10, 10.0),
            (0.30, 1.0),
            (0.30, 30.0),
            (0.50, 0.1),
            (0.50, 5.0),
            (0.80, 10.0),
            (0.80, 100.0),
            (1.00, 10.0),
            (1.20, 10.0),
            (2.00, 50.0),
        ),
        key=lambda pair: (pair[0], pair[1]),
    )
)

DEFAULT_SIGMA_A_VALUES: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_SIGMA_B_RATIOS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.6, 0.85, 0.98)
DEFAULT_KAPPA_SWEEPS: tuple[tuple[float, float, float, tuple[float, ...]], ...] = (
    (0.05, 1.0, 0.2, (0.0, 0.1, 1.0, 5.0, 10.0, 30.0, 100.0)),
    (0.50, 1.0, 0.2, (0.0, 0.1, 1.0, 5.0, 10.0, 30.0, 100.0)),
    (0.80, 10.0, 1.0, (0.0, 0.1, 1.0, 5.0, 10.0, 30.0, 100.0)),
    (1.20, 10.0, 1.0, (0.0, 0.1, 1.0, 5.0, 10.0, 30.0, 100.0)),
)
DEFAULT_RHO_SWEEPS: tuple[tuple[float, float, float, tuple[float, ...]], ...] = (
    (0.1, 1.0, 0.2, (0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 2.0)),
    (1.0, 1.0, 0.2, (0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 2.0)),
    (10.0, 10.0, 1.0, (0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 2.0)),
    (50.0, 10.0, 1.0, (0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 2.0)),
    (100.0, 100.0, 20.0, (0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 2.0)),
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


def sigma_scale_for_pair(rho: float, kappa: float) -> float:
    scale = max(abs(rho), abs(kappa), 1.0)
    if scale >= 100.0:
        return 100.0
    if scale >= 10.0:
        return 10.0
    return 1.0


def process_noise_from_sigma_a(sigma_a: float) -> float:
    if sigma_a <= 0:
        return 0.0
    return float(np.clip(0.02 * np.sqrt(sigma_a), WEAK_PROCESS_NOISE, MAX_PROCESS_NOISE))


def process_noise_scan(n_values: int) -> tuple[float, ...]:
    return tuple(float(value) for value in np.geomspace(WEAK_PROCESS_NOISE, MAX_PROCESS_NOISE, n_values))


def make_step2_system_matrix(rho: float, kappa: float) -> np.ndarray:
    return np.array(
        [
            [rho, 0.0, kappa],
            [0.0, rho, 0.0],
            [0.0, 0.0, rho**2],
        ],
        dtype=float,
    )


def make_step2_sigma_matrix(sigma_a: float, sigma_b: float, coupling: str = DEFAULT_SIGMA_COUPLING) -> np.ndarray:
    sigma = np.diag([sigma_a, sigma_a, sigma_a]).astype(float)
    coupling_indices = {
        "y_y2": (1, 2),
        "x_y2": (0, 2),
        "x_y": (0, 1),
    }
    if coupling not in coupling_indices:
        supported = ", ".join(sorted(coupling_indices))
        raise ValueError(f"Unsupported sigma coupling {coupling!r}. Supported: {supported}")
    i, j = coupling_indices[coupling]
    sigma[i, j] = sigma_b
    sigma[j, i] = sigma_b
    return 0.5 * (sigma + sigma.T)


def validate_sigma_params(sigma_a: float, sigma_b: float) -> None:
    if sigma_a <= 0:
        raise ValueError("sigma_a must be positive for the backward matrix")
    if sigma_b < 0:
        raise ValueError("sigma_b must be nonnegative")
    if not sigma_b < sigma_a:
        raise ValueError("sigma_b must be smaller than sigma_a so Sigma is positive definite")


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


def step2_next(state: np.ndarray, rho: float, kappa: float) -> np.ndarray:
    x, y = state
    return np.array([rho * x + kappa * y**2, rho * y], dtype=float)


def stable_combo_seed(combo: Step2Combo) -> int:
    seed_text = (
        f"{combo.label}|{combo.rho:.12g}|{combo.kappa:.12g}|"
        f"{combo.sigma_a:.12g}|{combo.sigma_b:.12g}|"
        f"{combo.process_noise:.12g}|{combo.mode}|{combo.sigma_coupling}"
    )
    return zlib.crc32(seed_text.encode("utf-8")) & 0xFFFFFFFF


def simulate_clipped_trajectories(
    rho: float,
    kappa: float,
    process_noise: float = 0.0,
    seed: int | None = None,
    steps: int = 90,
    phase_clip: float = 30.0,
) -> list[np.ndarray]:
    if process_noise < 0:
        raise ValueError("process_noise must be nonnegative")

    initial_states = [(x, y) for y in (-0.5, 0.0, 0.5) for x in (-0.3, 0.0, 0.3)]
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(len(initial_states))
    trajectories = []
    for initial, child_sequence in zip(initial_states, child_sequences):
        rng = np.random.default_rng(child_sequence)
        state = np.array(initial, dtype=float)
        points = [state.copy()]
        for _ in range(steps):
            state = step2_next(state, rho=rho, kappa=kappa)
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


def analyze_combo(combo: Step2Combo) -> dict[str, object]:
    A = make_step2_system_matrix(combo.rho, combo.kappa)
    trajectories = simulate_clipped_trajectories(
        rho=combo.rho,
        kappa=combo.kappa,
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
        validate_sigma_params(combo.sigma_a, combo.sigma_b)
        sigma = make_step2_sigma_matrix(combo.sigma_a, combo.sigma_b, coupling=combo.sigma_coupling)
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
    ax.set_aspect("auto", adjustable="box")
    ax.set_box_aspect(1.0)
    ax.set_xlabel(r"$x_k$")


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


def combo_title(combo: Step2Combo) -> str:
    parts = [rf"$\rho={format_param(combo.rho)}$", rf"$\kappa={format_param(combo.kappa)}$"]
    if combo.mode == "backward":
        parts.extend(
            [
                rf"$\sigma_a={format_param(combo.sigma_a)}$",
                rf"$\sigma_b={format_param(combo.sigma_b)}$",
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
            "rho": combo.rho,
            "kappa": combo.kappa,
            "sigma_a": combo.sigma_a,
            "sigma_b": combo.sigma_b,
            "process_noise": combo.process_noise,
            "sigma_coupling": combo.sigma_coupling,
        }
        for idx, value in enumerate(result["singular_values"], start=1):
            row[f"sv{idx}"] = float(value)
        for idx, value in enumerate(result["eigenvalues"], start=1):
            row[f"eig{idx}"] = float(value)
        rows.append(row)
    return output_path, pd.DataFrame(rows)


def make_experiment_specs(
    sigma_coupling: str = DEFAULT_SIGMA_COUPLING,
    reference_rho_kappa_pairs=REFERENCE_RHO_KAPPA_PAIRS,
    sigma_a_values=DEFAULT_SIGMA_A_VALUES,
    sigma_b_ratios=DEFAULT_SIGMA_B_RATIOS,
    kappa_sweeps=DEFAULT_KAPPA_SWEEPS,
    rho_sweeps=DEFAULT_RHO_SWEEPS,
) -> list[ExperimentSpec]:
    reference_rho_kappa_pairs = tuple(reference_rho_kappa_pairs)
    sigma_a_values = tuple(sigma_a_values)
    sigma_b_ratios = tuple(sigma_b_ratios)
    kappa_sweeps = tuple(kappa_sweeps)
    rho_sweeps = tuple(rho_sweeps)

    baseline_combos = tuple(
        Step2Combo(f"P{i:02d}", rho, kappa, mode="A", sigma_coupling=sigma_coupling)
        for i, (rho, kappa) in enumerate(reference_rho_kappa_pairs, start=1)
    )

    specs = []
    for part_index, start in enumerate(range(0, len(baseline_combos), 7), start=1):
        part = baseline_combos[start : start + 7]
        specs.append(
            ExperimentSpec(
                key=f"case1_baseline_A_reference_rho_kappa_part{part_index}",
                title=f"Case 1.{part_index}: no-noise baseline, direct SVD/EVD of A_step2",
                combos=part,
                plot_evd=True,
            )
        )

    sigma_a_process_noises = process_noise_scan(len(sigma_a_values))
    for rho, kappa in reference_rho_kappa_pairs:
        specs.append(
            ExperimentSpec(
                key=f"case2_sigmab0_scan_sigmaa_{fixed_slug(rho=rho, kappa=kappa)}",
                title=rf"Case 2: fixed $\rho,\kappa$, $\sigma_b=0$, scan $\sigma_a$ ({fixed_slug(rho=rho, kappa=kappa)})",
                combos=tuple(
                    Step2Combo(
                        f"A{i}",
                        rho,
                        kappa,
                        sigma_a=value,
                        sigma_b=0.0,
                        process_noise=process_noise,
                        sigma_coupling=sigma_coupling,
                    )
                    for i, (value, process_noise) in enumerate(zip(sigma_a_values, sigma_a_process_noises), start=1)
                ),
                share_phase_limits=True,
            )
        )

    for rho, kappa in reference_rho_kappa_pairs:
        sigma_a = sigma_scale_for_pair(rho, kappa)
        process_noise = process_noise_from_sigma_a(sigma_a)
        specs.append(
            ExperimentSpec(
                key=f"case3_scan_sigmab_{fixed_slug(rho=rho, kappa=kappa, siga=sigma_a)}",
                title=rf"Case 3: fixed $\rho,\kappa,\sigma_a$, scan $\sigma_b$ ({fixed_slug(rho=rho, kappa=kappa, siga=sigma_a)})",
                combos=tuple(
                    Step2Combo(
                        f"B{i}",
                        rho,
                        kappa,
                        sigma_a=sigma_a,
                        sigma_b=sigma_a * ratio,
                        process_noise=process_noise,
                        sigma_coupling=sigma_coupling,
                    )
                    for i, ratio in enumerate(sigma_b_ratios, start=1)
                ),
                share_phase_limits=True,
            )
        )

    for rho, sigma_a, sigma_b, kappa_values in kappa_sweeps:
        process_noise = process_noise_from_sigma_a(sigma_a)
        specs.append(
            ExperimentSpec(
                key=f"case4_scan_kappa_{fixed_slug(rho=rho, siga=sigma_a, sigb=sigma_b)}",
                title=rf"Case 4: fixed $\rho,\sigma_a,\sigma_b$, scan $\kappa$ ({fixed_slug(rho=rho, siga=sigma_a, sigb=sigma_b)})",
                combos=tuple(
                    Step2Combo(
                        f"K{i}",
                        rho,
                        kappa,
                        sigma_a=sigma_a,
                        sigma_b=sigma_b,
                        process_noise=process_noise,
                        sigma_coupling=sigma_coupling,
                    )
                    for i, kappa in enumerate(kappa_values, start=1)
                ),
            )
        )

    for kappa, sigma_a, sigma_b, rho_values in rho_sweeps:
        process_noise = process_noise_from_sigma_a(sigma_a)
        specs.append(
            ExperimentSpec(
                key=f"case5_scan_rho_{fixed_slug(kappa=kappa, siga=sigma_a, sigb=sigma_b)}",
                title=rf"Case 5: fixed $\kappa,\sigma_a,\sigma_b$, scan $\rho$ ({fixed_slug(kappa=kappa, siga=sigma_a, sigb=sigma_b)})",
                combos=tuple(
                    Step2Combo(
                        f"R{i}",
                        rho,
                        kappa,
                        sigma_a=sigma_a,
                        sigma_b=sigma_b,
                        process_noise=process_noise,
                        sigma_coupling=sigma_coupling,
                    )
                    for i, rho in enumerate(rho_values, start=1)
                ),
            )
        )
    return specs


def run_experiments(
    output_dir: str | Path,
    sigma_coupling: str = DEFAULT_SIGMA_COUPLING,
    **spec_kwargs,
) -> tuple[list[Path], pd.DataFrame]:
    configure_style()
    output_dir = Path(output_dir)
    paths = []
    summary_frames = []
    for spec in make_experiment_specs(sigma_coupling=sigma_coupling, **spec_kwargs):
        path, summary = plot_experiment(spec, output_dir=output_dir)
        paths.append(path)
        summary_frames.append(summary)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_path = output_dir / "case_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    paths.append(summary_path)
    return paths, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ana_pll2 step2 matrix-analysis grids.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/analysitic/figs/ana_pll2"),
        help="Directory for generated PNG grids and the summary CSV.",
    )
    parser.add_argument(
        "--sigma-coupling",
        default=DEFAULT_SIGMA_COUPLING,
        choices=("y_y2", "x_y2", "x_y"),
        help="Feature pair coupled by sigma_b in the step2 Sigma matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths, _ = run_experiments(output_dir=args.output_dir, sigma_coupling=args.sigma_coupling)
    print("Generated ana_pll2 outputs:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
