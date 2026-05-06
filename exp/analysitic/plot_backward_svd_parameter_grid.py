import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

Path("/tmp/koopcegis-matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/koopcegis-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools import (
    FEATURE_NAMES,
    compute_gis_metrics,
    make_initial_state_grid,
    make_manual_sigma_matrix,
    make_step_system_matrix,
    save_publication_figure,
    simulate_many_trajectories,
)


@dataclass(frozen=True)
class ParameterCombo:
    label: str
    lam: float
    mu: float
    a: float
    b: float
    process_noise: float


@dataclass(frozen=True)
class ExperimentGroup:
    key: str
    description: str
    combos: tuple[ParameterCombo, ...]


def make_experiment_groups(noise_process_max: float = 0.05) -> tuple[ExperimentGroup, ExperimentGroup]:
    weak_noise_a = 0.02
    weak_noise_b = 0.0
    weak_process_noise = 0.0002
    if noise_process_max <= weak_process_noise:
        raise ValueError(f"noise_process_max must be greater than {weak_process_noise}")
    deterministic_scan = ExperimentGroup(
        key="deterministic",
        description="fixed weak noise; deterministic parameters vary",
        combos=(
            ParameterCombo("D1", 0.10, 0.90, weak_noise_a, weak_noise_b, weak_process_noise),
            ParameterCombo("D2", 0.25, 0.88, weak_noise_a, weak_noise_b, weak_process_noise),
            ParameterCombo("D3", 0.45, 0.84, weak_noise_a, weak_noise_b, weak_process_noise),
            ParameterCombo("D4", 0.65, 0.82, weak_noise_a, weak_noise_b, weak_process_noise),
            ParameterCombo("D5", 0.82, 0.80, weak_noise_a, weak_noise_b, weak_process_noise),
            ParameterCombo("D6", 0.95, 0.78, weak_noise_a, weak_noise_b, weak_process_noise),
        ),
    )

    fixed_lam = 0.10
    fixed_mu = 0.90
    noise_levels = (0.02, 0.05, 0.10, 0.20, 0.50, 1.00)
    process_noises = tuple(np.geomspace(weak_process_noise, noise_process_max, len(noise_levels)))
    noise_scan = ExperimentGroup(
        key="noise",
        description="fixed fast-slow dynamics; noise strength varies",
        combos=tuple(
            ParameterCombo(
                f"N{idx}",
                fixed_lam,
                fixed_mu,
                level,
                0.20 * level,
                process_noise,
            )
            for idx, (level, process_noise) in enumerate(zip(noise_levels, process_noises), start=1)
        ),
    )
    return deterministic_scan, noise_scan


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
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def fix_singular_vector_signs(U: np.ndarray) -> np.ndarray:
    U = np.array(U, dtype=float, copy=True)
    for col in range(U.shape[1]):
        pivot = int(np.argmax(np.abs(U[:, col])))
        if U[pivot, col] < 0:
            U[:, col] *= -1.0
    return U


def analyze_combo(combo: ParameterCombo, initial_states, steps: int, seed: int, eps: float):
    trajectories = simulate_many_trajectories(
        initial_states=initial_states,
        steps=steps,
        lam=combo.lam,
        mu=combo.mu,
        noise_scale=combo.process_noise,
        seed=seed,
    )
    A = make_step_system_matrix(lam=combo.lam, mu=combo.mu)
    Sigma = make_manual_sigma_matrix(a=combo.a, b=combo.b)
    metrics = compute_gis_metrics(A, Sigma, eps=eps)
    backward = metrics["A_t_Sigma_inv_A"]
    U, singular_values, Vt = np.linalg.svd(backward, full_matrices=False)
    U = fix_singular_vector_signs(U)
    return {
        "combo": combo,
        "trajectories": trajectories,
        "A": A,
        "Sigma": Sigma,
        "backward": backward,
        "U": U,
        "S": singular_values,
        "Vt": Vt,
        "N": metrics["N"],
    }


def summary_rows(results):
    rows = []
    feature_keys = ["x", "y", "x2"]
    for result in results:
        combo = result["combo"]
        U = result["U"]
        S = result["S"]
        row = {
            "combo": combo.label,
            "lambda": combo.lam,
            "mu": combo.mu,
            "sigma_a": combo.a,
            "sigma_b": combo.b,
            "process_noise": combo.process_noise,
            "log_pdet_backward": result["N"],
        }
        for idx, value in enumerate(S, start=1):
            row[f"sv{idx}"] = float(value)
        for col in range(U.shape[1]):
            for row_idx, feature in enumerate(feature_keys):
                row[f"u{col + 1}_{feature}"] = float(U[row_idx, col])
        rows.append(row)
    return rows


def draw_phase_panel(ax, trajectories, colors, xlim, ylim):
    for idx, trajectory in enumerate(trajectories):
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=colors[idx],
            linewidth=0.75,
            alpha=0.78,
            solid_capstyle="round",
        )
        ax.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            s=10,
            facecolor="white",
            edgecolor=colors[idx],
            linewidth=0.55,
            zorder=3,
        )
        ax.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            s=10,
            marker="x",
            color=colors[idx],
            linewidth=0.65,
            zorder=3,
        )

    ax.axhline(0.0, color="0.84", linewidth=0.45, zorder=0)
    ax.axvline(0.0, color="0.84", linewidth=0.45, zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelleft=False)


def draw_spectrum_panel(ax, singular_values, y_limits):
    x = np.arange(1, len(singular_values) + 1)
    colors = ["#2f6f8f", "#86a9b8", "#cad6df"]
    ax.bar(
        x,
        singular_values,
        color=colors[: len(singular_values)],
        edgecolor="0.25",
        linewidth=0.35,
        width=0.66,
    )
    ax.plot(x, singular_values, color="0.25", linewidth=0.65, marker="o", markersize=2.2)
    ax.set_yscale("log")
    ax.set_ylim(y_limits)
    ax.set_xticks(x, [rf"$s_{i}$" for i in x])
    ax.grid(axis="y", color="0.90", linewidth=0.4)


def draw_vector_panel(ax, U):
    image = ax.imshow(U, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(U.shape[1]), [rf"$u_{i + 1}$" for i in range(U.shape[1])])
    ax.set_yticks(np.arange(len(FEATURE_NAMES)), FEATURE_NAMES)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(U.shape[0]):
        for col in range(U.shape[1]):
            value = U[row, col]
            text_color = "white" if abs(value) > 0.62 else "0.18"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=5.2, color=text_color)
    return image


def plot_parameter_grid(results, output_base: Path, formats):
    n_cols = len(results)
    all_trajectories = np.concatenate([result["trajectories"] for result in results], axis=0)
    x_min, x_max = all_trajectories[:, :, 0].min(), all_trajectories[:, :, 0].max()
    y_min, y_max = all_trajectories[:, :, 1].min(), all_trajectories[:, :, 1].max()
    x_pad = max(0.05, 0.05 * (x_max - x_min))
    y_pad = max(0.05, 0.05 * (y_max - y_min))
    xlim = (float(x_min - x_pad), float(x_max + x_pad))
    ylim = (float(y_min - y_pad), float(y_max + y_pad))

    all_singular_values = np.concatenate([result["S"] for result in results])
    positive = all_singular_values[all_singular_values > 0]
    y_limits = (
        max(float(positive.min()) * 0.55, 1e-8),
        float(positive.max()) * 1.65,
    )

    cmap = mpl.colormaps["viridis"]
    n_trajectories = results[0]["trajectories"].shape[0]
    colors = [cmap(value) for value in np.linspace(0.12, 0.88, n_trajectories)]

    fig = plt.figure(figsize=(11.2, 5.9), constrained_layout=True)
    gs = fig.add_gridspec(
        3,
        n_cols,
        height_ratios=(1.15, 0.72, 0.86),
        hspace=0.08,
        wspace=0.08,
    )

    phase_axes = []
    spectrum_axes = []
    vector_axes = []
    vector_image = None
    for col, result in enumerate(results):
        combo = result["combo"]
        ax_phase = fig.add_subplot(gs[0, col])
        ax_spectrum = fig.add_subplot(gs[1, col])
        ax_vector = fig.add_subplot(gs[2, col])
        phase_axes.append(ax_phase)
        spectrum_axes.append(ax_spectrum)
        vector_axes.append(ax_vector)

        draw_phase_panel(ax_phase, result["trajectories"], colors, xlim, ylim)
        draw_spectrum_panel(ax_spectrum, result["S"], y_limits)
        vector_image = draw_vector_panel(ax_vector, result["U"])

        title = (
            rf"{combo.label}: $\lambda={combo.lam:.2f}$, $\mu={combo.mu:.2f}$"
            "\n"
            rf"$a={combo.a:.2f}$, $b={combo.b:.2f}$"
        )
        ax_phase.set_title(title, loc="left", pad=4)

        if col == 0:
            ax_phase.set_ylabel(r"$y_k$")
            ax_phase.tick_params(labelleft=True)
            ax_spectrum.set_ylabel("Singular value")
            ax_vector.set_ylabel("Feature")
        else:
            ax_spectrum.tick_params(labelleft=False)
            ax_vector.tick_params(labelleft=False)
        ax_phase.set_xlabel(r"$x_k$")

    handles = [
        Line2D([0], [0], marker="o", color="0.35", markerfacecolor="white", linestyle="None", markersize=3.8, label="initial"),
        Line2D([0], [0], marker="x", color="0.35", linestyle="None", markersize=4.0, label="final"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.992, 1.012),
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    cbar = fig.colorbar(vector_image, ax=vector_axes, location="right", fraction=0.018, pad=0.01)
    cbar.set_label("Loading", rotation=90, labelpad=5)
    cbar.outline.set_linewidth(0.45)
    cbar.ax.tick_params(width=0.45, length=2.0, labelsize=5.8)

    return save_publication_figure(fig, output_base, formats=formats)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot phase trajectories and the SVD of A^T Sigma^{-1} A over representative parameter combos."
    )
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument(
        "--noise-process-max",
        type=float,
        default=0.05,
        help="Maximum process noise used by the fixed-dynamics noise scan.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("exp/analysitic/figs/backward_svd_parameter_grid"),
        help="Base path. Group keys are appended before the file extension.",
    )
    parser.add_argument("--formats", nargs="+", default=["svg", "pdf", "png", "tiff"])
    return parser.parse_args()


def main():
    args = parse_args()
    configure_style()
    initial_states = make_initial_state_grid(
        x_values=(-1.10, 0.0, 1.10),
        y_values=(-0.60, 0.0, 0.60),
    )
    print("Saved figure files:")
    for group_index, group in enumerate(make_experiment_groups(noise_process_max=args.noise_process_max)):
        results = [
            analyze_combo(
                combo,
                initial_states=initial_states,
                steps=args.steps,
                seed=args.seed + 100 * group_index + idx,
                eps=args.eps,
            )
            for idx, combo in enumerate(group.combos)
        ]
        output_base = args.output_base.with_name(f"{args.output_base.name}_{group.key}")
        saved = plot_parameter_grid(results, output_base=output_base, formats=args.formats)

        summary_path = output_base.with_name(output_base.name + "_summary.csv")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        rows = summary_rows(results)
        for row in rows:
            row["group"] = group.key
            row["group_description"] = group.description
        pd.DataFrame(rows).to_csv(summary_path, index=False)

        print(f"  {group.key}: {group.description}")
        for ext, path in saved.items():
            print(f"    {ext}: {path}")
        print(f"    summary: {summary_path}")


if __name__ == "__main__":
    main()
