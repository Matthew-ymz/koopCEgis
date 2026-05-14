from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
import tempfile

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

from data.data_func import (
    compute_cluster_order_parameters,
    compute_order_parameter,
    generate_kuramoto_cluster_data_sin_cos,
)


DEFAULT_N = 10
DEFAULT_N_CLUSTERS = 2
DEFAULT_T = 100.0
DEFAULT_DT = 0.01
DEFAULT_NOISE = 0.0
DEFAULT_RANDOM_SEED1 = 41
DEFAULT_RANDOM_SEED2 = 2600
DEFAULT_BURN_IN_STEPS = 1000
DEFAULT_SAMPLE_STRIDE = 10
DEFAULT_LAG_STEPS = 1
DEFAULT_RIDGE = 1e-10
DEFAULT_REGULARIZATION = 1e-10
DEFAULT_N_FOURIER_FREQUENCIES = 1
DEFAULT_TOP_SPECTRUM = 12
DEFAULT_TOP_VECTORS = 6

DEFAULT_CASE1_FIXED_K_INTRA = (2.0, 4.0, 6.0, 10.0, 16.0, 20.0)
DEFAULT_CASE1_K_INTER_VALUES = (0.0, 0.02, 0.05, 0.1, 0.2)
DEFAULT_CASE2_FIXED_K_INTER = (0.0, 0.02, 0.05, 0.1, 0.2)
DEFAULT_CASE2_K_INTRA_VALUES = (2.0, 4.0, 6.0, 10.0, 16.0, 20.0)

SPECTRUM_LOWER_FACTOR = 2.5
SPECTRUM_UPPER_FACTOR = 1.45
SPECTRUM_BAR_WIDTH = 0.55


@dataclass(frozen=True)
class KuramotoConfig:
    N: int = DEFAULT_N
    n_clusters: int = DEFAULT_N_CLUSTERS
    T: float = DEFAULT_T
    dt: float = DEFAULT_DT
    noise: float = DEFAULT_NOISE
    random_seed1: int = DEFAULT_RANDOM_SEED1
    random_seed2: int = DEFAULT_RANDOM_SEED2
    burn_in_steps: int = DEFAULT_BURN_IN_STEPS
    sample_stride: int = DEFAULT_SAMPLE_STRIDE
    lag_steps: int = DEFAULT_LAG_STEPS
    ridge: float = DEFAULT_RIDGE
    regularization: float = DEFAULT_REGULARIZATION
    n_fourier_frequencies: int = DEFAULT_N_FOURIER_FREQUENCIES
    top_spectrum: int = DEFAULT_TOP_SPECTRUM
    top_vectors: int = DEFAULT_TOP_VECTORS


@dataclass(frozen=True)
class KuramotoCombo:
    label: str
    K_intra: float
    K_inter: float


@dataclass(frozen=True)
class KuramotoSpec:
    key: str
    title: str
    combos: tuple[KuramotoCombo, ...]


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
            "legend.fontsize": 5.8,
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


def as_float_tuple(values) -> tuple[float, ...]:
    if isinstance(values, (int, float, np.integer, np.floating)):
        return (float(values),)
    return tuple(float(value) for value in values)


def raw_state_names(N: int) -> list[str]:
    return [f"cos_theta_{idx}" for idx in range(N)] + [f"sin_theta_{idx}" for idx in range(N)]


def make_observables(
    x_data: np.ndarray,
    state_names: list[str],
    n_fourier_frequencies: int = DEFAULT_N_FOURIER_FREQUENCIES,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Match the identity+Fourier idea from kuramoto_exp_gisce without adding a runtime dependency."""
    x_data = np.asarray(x_data, dtype=float)
    blocks = [x_data]
    feature_names = list(state_names)
    feature_sources = list(range(len(state_names)))
    for frequency in range(1, n_fourier_frequencies + 1):
        blocks.append(np.sin(frequency * x_data))
        feature_names.extend([f"sin({frequency} {name})" for name in state_names])
        feature_sources.extend(range(len(state_names)))
        blocks.append(np.cos(frequency * x_data))
        feature_names.extend([f"cos({frequency} {name})" for name in state_names])
        feature_sources.extend(range(len(state_names)))
    return np.hstack(blocks), feature_names, np.asarray(feature_sources, dtype=int)


def feature_block_names(state_dim: int, n_fourier_frequencies: int) -> list[tuple[str, int, int]]:
    blocks = [("identity", 0, state_dim)]
    offset = state_dim
    for frequency in range(1, n_fourier_frequencies + 1):
        blocks.append((f"sin {frequency}", offset, offset + state_dim))
        offset += state_dim
        blocks.append((f"cos {frequency}", offset, offset + state_dim))
        offset += state_dim
    return blocks


def feature_cluster_masks(feature_sources: np.ndarray, N: int, n_clusters: int) -> list[np.ndarray]:
    cluster_size = N // n_clusters
    oscillator_sources = feature_sources % N
    masks = []
    for cluster_id in range(n_clusters):
        start = cluster_id * cluster_size
        end = N if cluster_id == n_clusters - 1 else (cluster_id + 1) * cluster_size
        masks.append((oscillator_sources >= start) & (oscillator_sources < end))
    return masks


def prepare_time_pairs(data: np.ndarray, tau: int = 1) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must have shape (T, d)")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if len(data) <= tau:
        raise ValueError("data is too short for the requested tau")
    return data[:-tau], data[tau:]


def fit_linear_operator(
    X_now: np.ndarray,
    X_next: np.ndarray,
    ridge: float = DEFAULT_RIDGE,
    regularization: float = DEFAULT_REGULARIZATION,
) -> np.ndarray:
    """Fit X_next ~= A X_now with the same column-vector convention used in the GIS notebooks."""
    X_now = np.asarray(X_now, dtype=float)
    X_next = np.asarray(X_next, dtype=float)
    if X_now.ndim != 2 or X_next.ndim != 2 or X_now.shape != X_next.shape:
        raise ValueError("X_now and X_next must have the same 2D shape")
    n_samples, dim = X_now.shape
    C00 = (X_now.T @ X_now) / n_samples
    C01 = (X_now.T @ X_next) / n_samples
    C00_reg = C00 + (ridge + regularization) * np.eye(dim, dtype=float)
    empirical_regression = np.linalg.pinv(C00_reg, rcond=regularization) @ C01
    return empirical_regression.T


def fix_vector_signs(vectors: np.ndarray) -> np.ndarray:
    vectors = np.array(np.real_if_close(vectors), dtype=float, copy=True)
    for col in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, col])))
        if vectors[pivot, col] < 0:
            vectors[:, col] *= -1.0
    return vectors


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


def make_experiment_specs(
    case1_fixed_k_intra=DEFAULT_CASE1_FIXED_K_INTRA,
    case1_k_inter_values=DEFAULT_CASE1_K_INTER_VALUES,
    case2_fixed_k_inter=DEFAULT_CASE2_FIXED_K_INTER,
    case2_k_intra_values=DEFAULT_CASE2_K_INTRA_VALUES,
) -> list[KuramotoSpec]:
    specs: list[KuramotoSpec] = []
    case1_k_inter_values = as_float_tuple(case1_k_inter_values)
    case2_k_intra_values = as_float_tuple(case2_k_intra_values)

    for fixed_k_intra in as_float_tuple(case1_fixed_k_intra):
        case1_combos = tuple(
            KuramotoCombo(f"E{i}", fixed_k_intra, value)
            for i, value in enumerate(case1_k_inter_values, start=1)
            if value < fixed_k_intra
        )
        specs.append(
            KuramotoSpec(
                key=f"case1_scan_inter_fixed_intra_ki{slug_number(fixed_k_intra)}",
                title=rf"Kuramoto Case 1: fixed $K_{{intra}}={format_param(fixed_k_intra)}$, scan $K_{{inter}}$",
                combos=case1_combos,
            )
        )

    for fixed_k_inter in as_float_tuple(case2_fixed_k_inter):
        case2_combos = tuple(
            KuramotoCombo(f"I{i}", value, fixed_k_inter)
            for i, value in enumerate(case2_k_intra_values, start=1)
            if value > fixed_k_inter
        )
        specs.append(
            KuramotoSpec(
                key=f"case2_scan_intra_fixed_inter_ke{slug_number(fixed_k_inter)}",
                title=rf"Kuramoto Case 2: fixed $K_{{inter}}={format_param(fixed_k_inter)}$, scan $K_{{intra}}$",
                combos=case2_combos,
            )
        )

    return [spec for spec in specs if spec.combos]


def analyze_combo(combo: KuramotoCombo, config: KuramotoConfig) -> dict[str, object]:
    x_data, theta_hist, t_raw, _ = generate_kuramoto_cluster_data_sin_cos(
        N=config.N,
        n_clusters=config.n_clusters,
        K_intra=combo.K_intra,
        K_inter=combo.K_inter,
        dt=config.dt,
        T=config.T,
        noise=config.noise,
        random_seed1=config.random_seed1,
        random_seed2=config.random_seed2,
    )
    fit_slice = slice(config.burn_in_steps, None, config.sample_stride)
    x_fit = np.asarray(x_data[fit_slice], dtype=float)
    theta_fit = np.asarray(theta_hist[fit_slice], dtype=float)
    t_fit = np.asarray(t_raw[fit_slice], dtype=float)

    r_total = compute_order_parameter(theta_fit)
    r_groups = compute_cluster_order_parameters(theta_fit, config.n_clusters)

    state_names = raw_state_names(config.N)
    obs_data, feature_names, feature_sources = make_observables(
        x_fit,
        state_names,
        n_fourier_frequencies=config.n_fourier_frequencies,
    )
    X_now, X_next = prepare_time_pairs(obs_data, tau=config.lag_steps)
    A = fit_linear_operator(
        X_now,
        X_next,
        ridge=config.ridge,
        regularization=config.regularization,
    )
    ata = A.T @ A
    ata = 0.5 * (ata + ata.T)
    U, singular_values, _ = np.linalg.svd(ata, full_matrices=False)
    U = fix_vector_signs(U)
    cluster_masks = feature_cluster_masks(feature_sources, config.N, config.n_clusters)

    return {
        "combo": combo,
        "t": t_fit,
        "r_total": r_total,
        "r_groups": r_groups,
        "A": A,
        "ATA": ata,
        "singular_values": singular_values,
        "vectors": U,
        "feature_names": feature_names,
        "feature_sources": feature_sources,
        "feature_blocks": feature_block_names(len(state_names), config.n_fourier_frequencies),
        "cluster_masks": cluster_masks,
    }


def combo_title(combo: KuramotoCombo) -> str:
    return "\n".join(
        [
            f"{combo.label}: "
            + rf"$K_{{intra}}={format_param(combo.K_intra)}$, "
            + rf"$K_{{inter}}={format_param(combo.K_inter)}$",
            r"$\Sigma=I,\ B=A^TA$",
        ]
    )


def draw_order_panel(ax: plt.Axes, result: dict[str, object]) -> None:
    t = np.asarray(result["t"], dtype=float)
    r_total = np.asarray(result["r_total"], dtype=float)
    r_groups = [np.asarray(values, dtype=float) for values in result["r_groups"]]
    ax.plot(t, r_total, color="0.05", linewidth=1.2, label="overall")
    group_colors = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")
    for idx, r_group in enumerate(r_groups):
        ax.plot(t, r_group, color=group_colors[idx % len(group_colors)], linewidth=1.0, label=f"group {idx + 1}")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("time")
    ax.grid(axis="y", color="0.90", linewidth=0.4)


def draw_spectrum_panel(ax: plt.Axes, values: np.ndarray, top_k: int) -> None:
    values = np.asarray(values, dtype=float)[:top_k]
    x = np.arange(1, len(values) + 1)
    ax.bar(x, values, color="#2f6f8f", edgecolor="0.25", linewidth=0.35, width=SPECTRUM_BAR_WIDTH)
    ax.plot(x, values, color="0.25", linewidth=0.65, marker="o", markersize=2.1)
    ax.set_yscale("log" if np.all(values > 0) else "symlog")
    ax.set_ylim(spectrum_limits(values))
    ax.set_xticks(x, [rf"$s_{i}$" for i in x])
    ax.tick_params(axis="y", labelleft=True)
    ax.grid(axis="y", color="0.90", linewidth=0.4)


def sparse_feature_labels(feature_names: list[str], step: int = 6) -> list[str]:
    labels = []
    for idx, name in enumerate(feature_names):
        labels.append(name if idx % step == 0 else "")
    return labels


def draw_vector_panel(ax: plt.Axes, result: dict[str, object], top_vectors: int, show_ylabels: bool) -> mpl.image.AxesImage:
    vectors = np.asarray(result["vectors"], dtype=float)[:, :top_vectors]
    image = ax.imshow(vectors, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(vectors.shape[1]), [rf"$u_{idx + 1}$" for idx in range(vectors.shape[1])])
    feature_names = list(result["feature_names"])
    if show_ylabels:
        ax.set_yticks(np.arange(len(feature_names)), sparse_feature_labels(feature_names))
    else:
        ax.set_yticks([])
    for _, start, _ in result["feature_blocks"][1:]:
        ax.axhline(start - 0.5, color="0.15", linewidth=0.45)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_experiment(spec: KuramotoSpec, output_dir: Path, config: KuramotoConfig) -> tuple[Path, pd.DataFrame]:
    results = [analyze_combo(combo, config=config) for combo in spec.combos]
    n_cols = len(results)
    fig = plt.figure(figsize=(max(10.5, 2.05 * n_cols), 8.0), constrained_layout=True)
    gs = fig.add_gridspec(3, n_cols, height_ratios=(1.0, 0.72, 1.9), hspace=0.08, wspace=0.18)

    vector_axes = []
    vector_image = None
    for col, result in enumerate(results):
        combo = result["combo"]
        ax_order = fig.add_subplot(gs[0, col])
        ax_spectrum = fig.add_subplot(gs[1, col])
        ax_vector = fig.add_subplot(gs[2, col])

        draw_order_panel(ax_order, result)
        draw_spectrum_panel(ax_spectrum, result["singular_values"], top_k=config.top_spectrum)
        vector_image = draw_vector_panel(ax_vector, result, top_vectors=config.top_vectors, show_ylabels=(col == 0))
        vector_axes.append(ax_vector)

        ax_order.set_title(combo_title(combo), loc="left", pad=4)
        ax_spectrum.set_ylabel(r"$s(A^TA)$", labelpad=1.5)
        if col == 0:
            ax_order.set_ylabel("order r")
            ax_vector.set_ylabel("ATA left SV")
            ax_order.legend(loc="lower right", frameon=False)
        else:
            ax_order.tick_params(labelleft=False)

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
        singular_values = np.asarray(result["singular_values"], dtype=float)
        vectors = np.asarray(result["vectors"], dtype=float)
        cluster_masks = list(result["cluster_masks"])
        u1_abs = np.abs(vectors[:, 0])
        row = {
            "figure": spec.key,
            "combo": combo.label,
            "K_intra": combo.K_intra,
            "K_inter": combo.K_inter,
            "N": config.N,
            "n_clusters": config.n_clusters,
            "T": config.T,
            "dt": config.dt,
            "noise": config.noise,
            "random_seed1": config.random_seed1,
            "random_seed2": config.random_seed2,
            "burn_in_steps": config.burn_in_steps,
            "sample_stride": config.sample_stride,
            "lag_steps": config.lag_steps,
            "observable_dim": vectors.shape[0],
            "r_total_mean": float(np.mean(result["r_total"])),
            "r_group1_mean": float(np.mean(result["r_groups"][0])),
            "r_group2_mean": float(np.mean(result["r_groups"][1])),
        }
        row["order_gap"] = float(0.5 * (row["r_group1_mean"] + row["r_group2_mean"]) - row["r_total_mean"])
        for idx, value in enumerate(singular_values[: min(10, len(singular_values))], start=1):
            row[f"sv{idx}"] = float(value)
        for block_name, start, end in result["feature_blocks"]:
            row[f"u1_{block_name.replace(' ', '')}_mass"] = float(np.sum(u1_abs[start:end]))
        for idx, mask in enumerate(cluster_masks, start=1):
            row[f"u1_cluster{idx}_mass"] = float(np.sum(u1_abs[mask]))
        rows.append(row)

    return output_path, pd.DataFrame(rows)


def run_experiments(
    output_dir: str | Path,
    config: KuramotoConfig | None = None,
    **spec_kwargs,
) -> tuple[list[Path], pd.DataFrame]:
    configure_style()
    output_dir = Path(output_dir)
    config = config or KuramotoConfig()
    paths = []
    summary_frames = []
    for spec in make_experiment_specs(**spec_kwargs):
        path, summary = plot_experiment(spec, output_dir=output_dir, config=config)
        paths.append(path)
        summary_frames.append(summary)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_path = output_dir / "case_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    paths.append(summary_path)
    return paths, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ana_pll Kuramoto ATA grids.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/analysitic/figs/ana_pllkur"),
        help="Directory for generated PNG grids and the summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths, _ = run_experiments(output_dir=args.output_dir)
    print("Generated ana_pllkur outputs:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
