import argparse
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


REPO_ROOT = None
for candidate in [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent.parent,
    Path(__file__).resolve().parent.parent.parent,
]:
    if (candidate / "tools.py").exists():
        REPO_ROOT = candidate
        break
if REPO_ROOT is None:
    raise RuntimeError("Could not locate repository root containing tools.py")
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

if "plotly" not in sys.modules:
    plotly_module = types.ModuleType("plotly")
    plotly_module.express = types.ModuleType("plotly.express")
    plotly_module.graph_objects = types.ModuleType("plotly.graph_objects")
    sys.modules["plotly"] = plotly_module
    sys.modules["plotly.express"] = plotly_module.express
    sys.modules["plotly.graph_objects"] = plotly_module.graph_objects

from tools import (  # noqa: E402
    apply_coarse_graining,
    compute_ce_from_micro_macro,
    compute_gis_metrics,
    compute_prediction_errors,
    make_step_system_matrix,
    observable_step,
    simulate_discrete_system,
    step_map,
)


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 15

FEATURE_NAMES = ["$x$", "$y$", "$x^2$"]
A_SCAN_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0, 5.0, 8.0, 10.0]
A_SUMMARY_VALUES = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
B_SCAN_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
AB_SCAN_VALUES = [
    (0.1, 0.0),
    (0.2, 0.05),
    (0.4, 0.1),
    (0.5, 0.2),
    (0.6, 0.2),
    (0.8, 0.2),
    (1.0, 0.2),
    (1.0, 0.4),
    (2.0, 0.5),
    (2.0, 0.8),
    (5.0, 1.0),
    (8.0, 1.0),
    (10.0, 1.0),
]
C_A_VALUES = [0.5, 1.0, 2.0, 5.0, 8.0, 10.0]
C_B_VALUES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
DEFAULT_CONFIG = {
    "lam": 0.1,
    "mu": 0.9,
    "initial_state": [5.0, 5.0],
    "steps": 600,
    "dt": 1.0,
    "tau": 1,
    "alpha": 1.0,
    "noise_seed": 42,
    "metrics_eps": 1e-10,
    "manual_r": 1,
    "horizons": (1, 3, 5),
}


def sparse_labels(labels, step=1):
    if labels is None:
        return False
    if step <= 1:
        return labels
    return [label if i % step == 0 else "" for i, label in enumerate(labels)]


def choose_heatmap_cmap(matrix, cmap=None):
    if cmap is not None:
        return cmap
    matrix = np.asarray(matrix, dtype=float)
    if np.any(matrix < 0) and np.any(matrix > 0):
        return "vlag"
    return "Blues"


def standardize_for_plot(x):
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def format_value(value):
    return f"{value:.3f}".rstrip("0").rstrip(".") if value != 0 else "0"


def save_figure(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_manual_sigma_matrix(a, b):
    sigma_diag = np.diag([a, a, a])
    sigma_cross = np.array(
        [
            [0.0, 0.0, b],
            [0.0, 0.0, 0.0],
            [b, 0.0, 0.0],
        ],
        dtype=float,
    )
    sigma = sigma_diag + sigma_cross
    return 0.5 * (sigma + sigma.T)


def sample_gaussian_noise_from_sigma(n_samples, sigma, random_state=None):
    rng = np.random.default_rng(random_state)
    return rng.multivariate_normal(
        mean=np.zeros(sigma.shape[0]),
        cov=sigma,
        size=n_samples,
        check_valid="warn",
    )


def plot_matrix_heatmap_to_file(
    matrix,
    title,
    output_path,
    row_labels=None,
    col_labels=None,
    center=0.0,
    figsize=(6.2, 5.6),
    label_step=1,
    cmap=None,
):
    matrix_arr = np.asarray(matrix, dtype=float)
    final_cmap = choose_heatmap_cmap(matrix_arr, cmap=cmap)
    if center is None and final_cmap == "vlag":
        center = 0.0
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix_arr,
        ax=ax,
        cmap=final_cmap,
        center=center,
        square=matrix_arr.shape[0] == matrix_arr.shape[1],
        xticklabels=sparse_labels(col_labels, label_step),
        yticklabels=sparse_labels(row_labels, label_step),
    )
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_blue_singular_value_bars_to_file(values, title, output_path):
    values = np.asarray(values, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(np.arange(1, len(values) + 1), values, color="tab:blue", alpha=0.92, width=0.72, edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular Value")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_sorted_svd_spectrum_to_file(
    forward_values,
    backward_values,
    title,
    output_path,
    forward_label="$\\Sigma^{-1}$",
    backward_label="$A^T\\Sigma^{-1}A$",
):
    forward_values = np.asarray(forward_values, dtype=float).ravel()
    backward_values = np.asarray(backward_values, dtype=float).ravel()
    combined_data = [(value, "forward") for value in forward_values]
    combined_data.extend((value, "backward") for value in backward_values)
    combined_data.sort(key=lambda item: item[0], reverse=True)
    sorted_values = [item[0] for item in combined_data]
    sorted_labels = [item[1] for item in combined_data]
    x = np.arange(1, len(sorted_values) + 1)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for i, value in enumerate(sorted_values):
        color = "tab:blue" if sorted_labels[i] == "forward" else "tab:orange"
        ax.bar(x[i], value, color=color, alpha=0.5, edgecolor="none")

    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=10, label=forward_label),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=10, label=backward_label),
        ]
    )
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_prediction_curves_to_file(series_names, predictions, targets, title, output_path, sample_count=80):
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    limit = min(sample_count, len(predictions))
    x = np.arange(limit)
    for idx, name in enumerate(series_names):
        ax.plot(x, targets[:limit, idx], label=f"true {name}")
        ax.plot(x, predictions[:limit, idx], "--", linewidth=1.8, label=f"pred {name}")
    ax.set_title(title)
    ax.set_xlabel("Pair Index")
    ax.set_ylabel("Value")
    ax.legend(ncol=max(1, min(3, len(series_names))))
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_micro_macro_curve_compare_to_file(micro_series, macro_series, macro_names, title, output_path, sample_count=120):
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    limit = min(sample_count, len(micro_series), len(macro_series))
    x = np.arange(limit)
    for idx, name in enumerate(FEATURE_NAMES):
        ax.plot(x, standardize_for_plot(micro_series[:limit, idx]), linewidth=1.6, label=f"micro: {name}")
    for idx, name in enumerate(macro_names):
        ax.plot(x, standardize_for_plot(macro_series[:limit, idx]), "--", linewidth=2.3, label=f"macro: {name}")
    ax.set_title(title)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Standardized Value")
    ax.legend(ncol=2)
    fig.tight_layout()
    save_figure(fig, output_path)


def build_w_from_two_stage(A, Sigma, metrics_eps=1e-10, manual_r=1, stage1_threshold=0.0):
    metrics = compute_gis_metrics(A, Sigma, alpha=DEFAULT_CONFIG["alpha"], eps=metrics_eps)
    sigma_inv = metrics["Sigma_inv"]
    backward = metrics["A_t_Sigma_inv_A"]

    u_det, s_det, _ = np.linalg.svd(sigma_inv, full_matrices=False)
    u_nondeg, s_nondeg, _ = np.linalg.svd(backward, full_matrices=False)

    combined_scores = np.concatenate([s_nondeg, s_det], axis=0)
    combined_vectors = np.concatenate([u_nondeg, u_det], axis=1)
    source_labels = np.array(["nondeg"] * len(s_nondeg) + ["det"] * len(s_det), dtype=object)
    order = np.argsort(-combined_scores)
    s_all = combined_scores[order]
    u_all = combined_vectors[:, order]
    source_all = source_labels[order]

    keep_stage1 = s_all >= stage1_threshold
    u_bar = u_all[:, keep_stage1]
    s_bar = s_all[keep_stage1]
    source_bar = source_all[keep_stage1]
    weighted_matrix = u_bar @ np.diag(s_bar)

    u2, s2, _ = np.linalg.svd(weighted_matrix, full_matrices=False)
    r_final = max(1, min(int(manual_r), u2.shape[1]))
    basis = u2[:, :r_final]

    return {
        "W": basis.T,
        "r": r_final,
        "sv_info": {
            "sv_forward": metrics["sv_forward"],
            "sv_backward": metrics["sv_backward"],
            "sv_det": s_det,
            "sv_nondeg": s_nondeg,
            "sv_all": s_all,
            "sv_stage2": s2,
        },
        "basis_info": {
            "weighted_matrix": weighted_matrix,
            "source_all": source_all.tolist(),
            "source_bar": source_bar.tolist(),
        },
    }


def plot_summary_ce_j_to_file(results_df, param_col, output_path):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = results_df[param_col].to_numpy()
    ax.plot(x, results_df["CE"], marker="o", linewidth=2.0, label="CE")
    ax.plot(x, results_df["micro_J_alpha"], marker="s", linewidth=1.8, label="micro J_alpha")
    ax.plot(x, results_df["macro_J_alpha"], marker="^", linewidth=1.8, label="macro J_alpha")
    ax.set_title("CE and J_alpha")
    ax.set_xlabel(param_col)
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_summary_d_n_to_file(results_df, param_col, output_path):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = results_df[param_col].to_numpy()
    ax.plot(x, results_df["micro_D"], marker="o", linewidth=1.8, label="micro D")
    ax.plot(x, results_df["micro_N"], marker="s", linewidth=1.8, label="micro N")
    ax.plot(x, results_df["macro_D"], marker="^", linewidth=1.8, label="macro D")
    ax.plot(x, results_df["macro_N"], marker="d", linewidth=1.8, label="macro N")
    ax.set_title("D and N")
    ax.set_xlabel(param_col)
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_summary_prediction_errors_to_file(results_df, param_col, horizons, output_path):
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    x = results_df[param_col].to_numpy()
    markers = {1: "o", 3: "s", 5: "^"}
    for horizon in horizons:
        ax.plot(x, results_df[f"micro_E{horizon}"], marker=markers.get(horizon, "o"), linewidth=1.8, label=f"micro E{horizon}")
        ax.plot(x, results_df[f"macro_E{horizon}"], marker=markers.get(horizon, "o"), linewidth=1.8, linestyle="--", label=f"macro E{horizon}")
    ax.set_title("Prediction Errors")
    ax.set_xlabel(param_col)
    ax.set_ylabel("Mean Squared Error")
    ax.legend(ncol=2)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_3d_bars_series(results_df, x_col, value_cols, y_labels, title, x_label, y_label, z_label, output_path):
    fig = plt.figure(figsize=(10.8, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    x_values = results_df[x_col].to_numpy()
    norm = Normalize(vmin=0, vmax=max(1, len(y_labels) - 1))
    cmap = cm.Blues

    for yi, col in enumerate(value_cols):
        xs = x_values
        ys = np.full_like(xs, yi, dtype=float)
        zs = np.zeros_like(xs, dtype=float)
        dx = np.full_like(xs, max(0.03, (xs.max() - xs.min()) / max(20, len(xs) * 4)), dtype=float)
        dy = np.full_like(xs, 0.6, dtype=float)
        dz = results_df[col].to_numpy()
        color = cmap(0.35 + 0.5 * norm(yi))
        ax.bar3d(xs - dx / 2, ys - dy / 2, zs, dx, dy, dz, color=color, edgecolor="none", shade=True, alpha=0.95)

    ax.set_title(title, pad=18)
    ax.set_xlabel(x_label, labelpad=12)
    ax.set_ylabel(y_label, labelpad=12)
    ax.set_zlabel(z_label, labelpad=12)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.view_init(elev=24, azim=-128)
    save_figure(fig, output_path)


def plot_3d_ab_panels(results_df, value_cols, titles, x_col, y_col, z_label, super_title, output_path, ncols=3):
    n_panels = len(value_cols)
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig = plt.figure(figsize=(5.2 * ncols, 4.6 * nrows))
    norm = Normalize(vmin=results_df[value_cols].to_numpy().min(), vmax=results_df[value_cols].to_numpy().max())
    cmap = cm.Blues

    for idx, (col, panel_title) in enumerate(zip(value_cols, titles), start=1):
        ax = fig.add_subplot(nrows, ncols, idx, projection="3d")
        xs = results_df[x_col].to_numpy()
        ys = results_df[y_col].to_numpy()
        dz = results_df[col].to_numpy()
        dx = np.full_like(xs, 0.18, dtype=float)
        dy = np.full_like(xs, 0.08, dtype=float)
        colors = cmap(0.25 + 0.65 * norm(dz))
        ax.bar3d(xs - dx / 2, ys - dy / 2, np.zeros_like(dz), dx, dy, dz, color=colors, edgecolor="none", shade=True, alpha=0.96)
        ax.set_title(panel_title, pad=10)
        ax.set_xlabel("a", labelpad=6)
        ax.set_ylabel("b", labelpad=6)
        ax.set_zlabel(z_label, labelpad=6)
        ax.view_init(elev=24, azim=-130)

    fig.suptitle(super_title, y=0.99)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_heatmap_from_grid(results_df, value_col, title, output_path, x_col="a", y_col="b", cmap="Blues"):
    pivot = results_df.pivot(index=y_col, columns=x_col, values=value_col).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    sns.heatmap(pivot, ax=ax, cmap=cmap, mask=pivot.isna(), cbar=True)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    save_figure(fig, output_path)


def dataframe_to_markdown(df):
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_single_experiment(a, b, label, figs_root, config):
    single_dir = figs_root / label
    single_dir.mkdir(parents=True, exist_ok=True)

    clean_sim = simulate_discrete_system(
        step_map,
        config["initial_state"],
        steps=config["steps"],
        system_kwargs={"lam": config["lam"], "mu": config["mu"]},
        dt=config["dt"],
    )
    clean_xy = clean_sim["trajectories"][0]
    obs_clean = observable_step(clean_xy, mode="default")

    sigma_micro = make_manual_sigma_matrix(a, b)
    obs_noise = sample_gaussian_noise_from_sigma(len(obs_clean), sigma_micro, random_state=config["noise_seed"])
    obs_noisy = obs_clean + obs_noise

    a_micro = make_step_system_matrix(config["lam"], config["mu"])
    micro_metrics = compute_gis_metrics(a_micro, sigma_micro, alpha=config["alpha"], eps=config["metrics_eps"])
    micro_errors = compute_prediction_errors(a_micro, obs_noisy, tau=config["tau"], horizons=config["horizons"])

    w_result = build_w_from_two_stage(a_micro, sigma_micro, metrics_eps=config["metrics_eps"], manual_r=config["manual_r"], stage1_threshold=0.0)
    w_matrix = w_result["W"]
    macro_names = [f"$z_{i+1}$" for i in range(w_result["r"])]
    z_series = apply_coarse_graining(w_matrix, obs_noisy)

    a_macro = w_matrix @ a_micro @ w_matrix.T
    sigma_macro = w_matrix @ sigma_micro @ w_matrix.T
    macro_metrics = compute_gis_metrics(a_macro, sigma_macro, alpha=config["alpha"], eps=config["metrics_eps"])
    macro_errors = compute_prediction_errors(a_macro, z_series, tau=config["tau"], horizons=config["horizons"])
    ce_result = compute_ce_from_micro_macro(micro_metrics, macro_metrics)

    plot_matrix_heatmap_to_file(
        sigma_micro,
        f"Micro Sigma, {label}",
        single_dir / f"micro_sigma_{label}.png",
        row_labels=FEATURE_NAMES,
        col_labels=FEATURE_NAMES,
        center=0.0,
    )
    plot_prediction_curves_to_file(
        FEATURE_NAMES,
        micro_errors[1]["predictions"],
        micro_errors[1]["targets"],
        f"Micro Prediction, {label}",
        single_dir / f"micro_prediction_{label}.png",
    )
    plot_sorted_svd_spectrum_to_file(
        micro_metrics["sv_forward"],
        micro_metrics["sv_backward"],
        f"Sorted SVD Spectrum, {label}",
        single_dir / f"sorted_svd_spectrum_{label}.png",
    )
    plot_matrix_heatmap_to_file(
        w_result["basis_info"]["weighted_matrix"],
        f"Stage-2 Matrix, {label}",
        single_dir / f"stage2_matrix_{label}.png",
        row_labels=FEATURE_NAMES,
        col_labels=[f"$c_{i+1}$" for i in range(w_result["basis_info"]["weighted_matrix"].shape[1])],
        center=0.0,
        figsize=(6.2, 4.8),
        cmap="vlag",
    )
    plot_blue_singular_value_bars_to_file(
        w_result["sv_info"]["sv_stage2"],
        f"Stage-2 Singular Values, {label}",
        single_dir / f"stage2_spectrum_{label}.png",
    )
    plot_matrix_heatmap_to_file(
        np.abs(w_matrix),
        f"Absolute W, {label}",
        single_dir / f"W_heatmap_{label}.png",
        row_labels=macro_names,
        col_labels=FEATURE_NAMES,
        center=None,
        figsize=(5.4, 3.4),
        cmap="Blues",
    )
    plot_prediction_curves_to_file(
        macro_names,
        macro_errors[1]["predictions"],
        macro_errors[1]["targets"],
        f"Macro Prediction, {label}",
        single_dir / f"macro_prediction_{label}.png",
    )
    plot_micro_macro_curve_compare_to_file(
        obs_noisy,
        z_series,
        macro_names,
        f"Micro and Macro Curves, {label}",
        single_dir / f"micro_macro_curve_{label}.png",
    )

    stage2_sv = np.asarray(w_result["sv_info"]["sv_stage2"], dtype=float)
    w_abs = np.abs(np.asarray(w_matrix, dtype=float).ravel())

    row = {
        "label": label,
        "a": float(a),
        "b": float(b),
        "micro_dim": micro_metrics["dimension"],
        "macro_dim": macro_metrics["dimension"],
        "selected_r": w_result["r"],
        "CE": ce_result["CE"],
        "micro_J_alpha": micro_metrics["J_alpha"],
        "macro_J_alpha": macro_metrics["J_alpha"],
        "micro_D": micro_metrics["D"],
        "micro_N": micro_metrics["N"],
        "macro_D": macro_metrics["D"],
        "macro_N": macro_metrics["N"],
        "stage2_sv1": float(stage2_sv[0]) if len(stage2_sv) > 0 else np.nan,
        "stage2_sv2": float(stage2_sv[1]) if len(stage2_sv) > 1 else np.nan,
        "stage2_sv3": float(stage2_sv[2]) if len(stage2_sv) > 2 else np.nan,
        "W_abs_x": float(w_abs[0]) if len(w_abs) > 0 else np.nan,
        "W_abs_y": float(w_abs[1]) if len(w_abs) > 1 else np.nan,
        "W_abs_x2": float(w_abs[2]) if len(w_abs) > 2 else np.nan,
    }
    for horizon in config["horizons"]:
        row[f"micro_E{horizon}"] = micro_errors[horizon]["mean_error"]
        row[f"macro_E{horizon}"] = macro_errors[horizon]["mean_error"]

    detail = {
        "label": label,
        "micro_metrics": micro_metrics,
        "macro_metrics": macro_metrics,
        "micro_errors": micro_errors,
        "macro_errors": macro_errors,
        "ce_result": ce_result,
        "stage2_sv": stage2_sv,
        "w_abs": w_abs,
    }
    return row, detail


def detail_by_label(details):
    return {detail["label"]: detail for detail in details}


def render_detail_block(root_name, label, detail):
    return "\n".join(
        [
            f"#### {label}",
            "",
            f"![Micro Sigma](./figs/{root_name}/{label}/micro_sigma_{label}.png)",
            "",
            f"![Micro Prediction](./figs/{root_name}/{label}/micro_prediction_{label}.png)",
            "",
            f"![Sorted SVD Spectrum](./figs/{root_name}/{label}/sorted_svd_spectrum_{label}.png)",
            "",
            f"![Stage2 Matrix](./figs/{root_name}/{label}/stage2_matrix_{label}.png)",
            "",
            f"![Stage2 Singular Values](./figs/{root_name}/{label}/stage2_spectrum_{label}.png)",
            "",
            f"![Absolute W](./figs/{root_name}/{label}/W_heatmap_{label}.png)",
            "",
            f"![Macro Prediction](./figs/{root_name}/{label}/macro_prediction_{label}.png)",
            "",
            f"![Micro Macro Curves](./figs/{root_name}/{label}/micro_macro_curve_{label}.png)",
            "",
            f"- Stage-2 singular values: `{np.array2string(detail['stage2_sv'], precision=6, separator=', ')}`",
            f"- Absolute W: `{np.array2string(detail['w_abs'], precision=6, separator=', ')}`",
            "",
        ]
    )


def select_best_b_row(results_df):
    df = results_df.copy()
    df["sv_ratio"] = df["stage2_sv1"] / np.maximum(df["stage2_sv2"], 1e-12)
    mask = (df["W_abs_y"] > df["W_abs_x"]) & (df["W_abs_y"] > df["W_abs_x2"])
    candidate_df = df[mask].copy()
    if candidate_df.empty:
        candidate_df = df
    candidate_df["score"] = candidate_df["sv_ratio"] + 0.5 * candidate_df["CE"]
    best_idx = candidate_df["score"].idxmax()
    return candidate_df.loc[best_idx]


def build_section_markdown_a(results_df, details):
    detail_map = detail_by_label(details)
    summary_df = results_df[results_df["a"].isin(A_SUMMARY_VALUES)].copy()
    extra_labels = [label for label in results_df["label"].tolist() if label not in set(summary_df["label"])]
    lines = [
        "## 实验一：固定 b=0，扫描 a",
        "",
        "固定 `b = 0`，扫描 `a`。汇总图仅展示 `a ∈ [0.2, 1.0]` 的结果，`a=0`、极小噪声和大噪声点保留为单次实验图。",
        "",
        "### 汇总图",
        "",
        "![CE and J_alpha](./figs/a/all/summary_ce_j_alpha.png)",
        "",
        "![D and N](./figs/a/all/summary_d_n.png)",
        "",
        "![Prediction Errors](./figs/a/all/summary_prediction_errors.png)",
        "",
        "![Stage2 Singular Values](./figs/a/all/summary_stage2_singular_values.png)",
        "",
        "![Absolute W](./figs/a/all/summary_w_abs.png)",
        "",
        "### 数值汇总",
        "",
        dataframe_to_markdown(summary_df),
        "",
        "### 说明",
        "",
        "- 当 `b=0` 且 `a>0` 时，`Sigma = aI`，其逆就是 `Sigma^{-1} = (1/a)I`。",
        "- 当 `a=0, b=0` 时，`Sigma` 退化为零矩阵；当前代码在 `compute_gis_metrics` 中会加上极小正则项并使用正则化伪逆，因此会出现非常大的奇异值。这也是本次把 `a=0` 从汇总图中移除的原因。",
        "",
        "### 单独结论",
        "",
        "- 在 `b=0` 的条件下，`CE` 在汇总区间内基本保持稳定，说明纯对角噪声主要改变整体尺度，而没有明显改变宏观层相对微观层的单位维度优势。",
        "- `micro_J_alpha` 与 `macro_J_alpha` 会随着 `a` 增大而整体下降，说明噪声增强会同时压低宏观和微观的可逆性，但二者的差值变化很小。",
        "- Stage-2 奇异值随着 `a` 增大快速衰减，说明两阶段结构的谱强度主要受对角噪声幅值控制。",
        "",
        "### 单次实验图（不在汇总图中展示的点）",
        "",
    ]
    for label in extra_labels:
        lines.append(render_detail_block("a", label, detail_map[label]))
    return "\n".join(lines)


def build_section_markdown_b(results_df, details):
    best_row = select_best_b_row(results_df)
    lines = [
        "## 实验二：固定 a=1，扫描 b",
        "",
        "固定 `a = 1`，扫描 `b`。由于全部 `b` 值都进入汇总图，因此本节不再重复展示单次图。",
        "",
        "### 汇总图",
        "",
        "![CE and J_alpha](./figs/b/all/summary_ce_j_alpha.png)",
        "",
        "![D and N](./figs/b/all/summary_d_n.png)",
        "",
        "![Prediction Errors](./figs/b/all/summary_prediction_errors.png)",
        "",
        "![Stage2 Singular Values](./figs/b/all/summary_stage2_singular_values.png)",
        "",
        "![Absolute W](./figs/b/all/summary_w_abs.png)",
        "",
        "### 数值汇总",
        "",
        dataframe_to_markdown(results_df),
        "",
        "### 单独结论",
        "",
        "- 随着 `b` 增大，`CE` 整体下降，说明交叉噪声会持续侵蚀宏观优势。",
        "- `macro_J_alpha` 对 `b` 的变化更敏感，表明交叉项首先破坏的是宏观压缩后的有效组织效率。",
        f"- 按“`sv1` 显著大于其余奇异值且 `|W_y|` 最大”这一标准，本轮最优点出现在 `b={best_row['b']:.2f}`，对应 `sv = ({best_row['stage2_sv1']:.6f}, {best_row['stage2_sv2']:.6f}, {best_row['stage2_sv3']:.6f})`，`|W| = ({best_row['W_abs_x']:.6f}, {best_row['W_abs_y']:.6f}, {best_row['W_abs_x2']:.6f})`，`CE = {best_row['CE']:.6f}`。",
        "",
    ]
    return "\n".join(lines)


def build_section_markdown_c(results_df):
    lines = [
        "## 实验三：更规整的联合扫描网格 c",
        "",
        "使用更规则的 `(a, b)` 网格，并保持 `a > b`。本节只展示热力图，不重复展示已纳入汇总的单次实验图。",
        "",
        "### 汇总图",
        "",
        "![CE Heatmap](./figs/c/all/summary_ce_j_alpha.png)",
        "",
        "![Micro J_alpha Heatmap](./figs/c/all/summary_d_n.png)",
        "",
        "![Macro J_alpha Heatmap](./figs/c/all/summary_prediction_errors.png)",
        "",
        "### 数值汇总",
        "",
        dataframe_to_markdown(results_df[["label", "a", "b", "CE", "micro_J_alpha", "macro_J_alpha"]]),
        "",
        "### 单独结论",
        "",
        "- 在规则网格上，`CE` 的高值区集中在小 `a`、小 `b` 的角落。",
        "- `micro_J_alpha` 和 `macro_J_alpha` 都会随着 `a`、`b` 的增大而整体下降，但 `macro_J_alpha` 对交叉项 `b` 更敏感。",
        "- 这说明在联合噪声下，交叉耦合是压低宏观收益的关键因素，而纯尺度放大主要由 `a` 控制。",
        "",
    ]
    return "\n".join(lines)


def build_report(a_df, a_details, b_df, b_details, c_df):
    return "\n".join(
        [
            "# 第三部分噪音批量实验",
            "",
            build_section_markdown_a(a_df, a_details),
            "",
            build_section_markdown_b(b_df, b_details),
            "",
            build_section_markdown_c(c_df),
            "",
        ]
    )


def run_scan_a(figs_root, config):
    root = figs_root / "a"
    summary_dir = root / "all"
    rows, details = [], []
    for a in A_SCAN_VALUES:
        label = format_value(a)
        row, detail = run_single_experiment(a, 0.0, label, root, config)
        rows.append(row)
        details.append(detail)
    df = pd.DataFrame(rows).sort_values("a").reset_index(drop=True)
    summary_df = df[df["a"].isin(A_SUMMARY_VALUES)].copy()
    plot_summary_ce_j_to_file(summary_df, "a", summary_dir / "summary_ce_j_alpha.png")
    plot_summary_d_n_to_file(summary_df, "a", summary_dir / "summary_d_n.png")
    plot_summary_prediction_errors_to_file(summary_df, "a", config["horizons"], summary_dir / "summary_prediction_errors.png")
    plot_3d_bars_series(
        summary_df,
        "a",
        ["stage2_sv3", "stage2_sv2", "stage2_sv1"],
        ["sv3", "sv2", "sv1"],
        "Stage-2 Singular Values vs a",
        "a",
        "Singular Index",
        "Singular Value",
        summary_dir / "summary_stage2_singular_values.png",
    )
    plot_3d_bars_series(
        summary_df,
        "a",
        ["W_abs_x", "W_abs_y", "W_abs_x2"],
        ["|W_x|", "|W_y|", "|W_x2|"],
        "Absolute W vs a",
        "a",
        "Component",
        "Absolute Value",
        summary_dir / "summary_w_abs.png",
    )
    return df, details


def run_scan_b(figs_root, config):
    root = figs_root / "b"
    summary_dir = root / "all"
    rows, details = [], []
    for b in B_SCAN_VALUES:
        label = format_value(b)
        row, detail = run_single_experiment(1.0, b, label, root, config)
        rows.append(row)
        details.append(detail)
    df = pd.DataFrame(rows).sort_values("b").reset_index(drop=True)
    plot_summary_ce_j_to_file(df, "b", summary_dir / "summary_ce_j_alpha.png")
    plot_summary_d_n_to_file(df, "b", summary_dir / "summary_d_n.png")
    plot_summary_prediction_errors_to_file(df, "b", config["horizons"], summary_dir / "summary_prediction_errors.png")
    plot_3d_bars_series(
        df,
        "b",
        ["stage2_sv3", "stage2_sv2", "stage2_sv1"],
        ["sv3", "sv2", "sv1"],
        "Stage-2 Singular Values vs b",
        "b",
        "Singular Index",
        "Singular Value",
        summary_dir / "summary_stage2_singular_values.png",
    )
    plot_3d_bars_series(
        df,
        "b",
        ["W_abs_x", "W_abs_y", "W_abs_x2"],
        ["|W_x|", "|W_y|", "|W_x2|"],
        "Absolute W vs b",
        "b",
        "Component",
        "Absolute Value",
        summary_dir / "summary_w_abs.png",
    )
    return df, details


def run_scan_ab(figs_root, config):
    root = figs_root / "ab"
    summary_dir = root / "all"
    rows, details = [], []
    for a, b in AB_SCAN_VALUES:
        label = f"a_{format_value(a)}_b_{format_value(b)}"
        row, detail = run_single_experiment(a, b, label, root, config)
        rows.append(row)
        details.append(detail)
    df = pd.DataFrame(rows).sort_values(["a", "b"]).reset_index(drop=True)
    plot_3d_ab_panels(
        df,
        ["CE", "micro_J_alpha", "macro_J_alpha"],
        ["CE", "micro J_alpha", "macro J_alpha"],
        "a",
        "b",
        "Value",
        "CE and J_alpha on (a, b)",
        summary_dir / "summary_ce_j_alpha.png",
        ncols=3,
    )
    plot_3d_ab_panels(
        df,
        ["micro_D", "micro_N", "macro_D", "macro_N"],
        ["micro D", "micro N", "macro D", "macro N"],
        "a",
        "b",
        "Value",
        "D and N on (a, b)",
        summary_dir / "summary_d_n.png",
        ncols=2,
    )
    plot_3d_ab_panels(
        df,
        ["micro_E1", "macro_E1", "micro_E3", "macro_E3", "micro_E5", "macro_E5"],
        ["micro E1", "macro E1", "micro E3", "macro E3", "micro E5", "macro E5"],
        "a",
        "b",
        "Mean Squared Error",
        "Prediction Errors on (a, b)",
        summary_dir / "summary_prediction_errors.png",
        ncols=3,
    )
    plot_3d_ab_panels(
        df,
        ["stage2_sv1", "stage2_sv2", "stage2_sv3"],
        ["sv1", "sv2", "sv3"],
        "a",
        "b",
        "Singular Value",
        "Stage-2 Singular Values on (a, b)",
        summary_dir / "summary_stage2_singular_values.png",
    )
    plot_3d_ab_panels(
        df,
        ["W_abs_x", "W_abs_y", "W_abs_x2"],
        ["|W_x|", "|W_y|", "|W_x2|"],
        "a",
        "b",
        "Absolute Value",
        "Absolute W on (a, b)",
        summary_dir / "summary_w_abs.png",
    )
    return df, details


def run_scan_c(figs_root, config):
    root = figs_root / "c"
    summary_dir = root / "all"
    rows, details = [], []
    for a in C_A_VALUES:
        for b in C_B_VALUES:
            if not (a > b):
                continue
            label = f"a_{format_value(a)}_b_{format_value(b)}"
            row, detail = run_single_experiment(a, b, label, root, config)
            rows.append(row)
            details.append(detail)
    df = pd.DataFrame(rows).sort_values(["a", "b"]).reset_index(drop=True)
    plot_heatmap_from_grid(df, "CE", "CE Heatmap", summary_dir / "summary_ce_j_alpha.png", cmap="vlag")
    plot_heatmap_from_grid(df, "micro_J_alpha", "Micro J_alpha Heatmap", summary_dir / "summary_d_n.png", cmap="vlag")
    plot_heatmap_from_grid(df, "macro_J_alpha", "Macro J_alpha Heatmap", summary_dir / "summary_prediction_errors.png", cmap="vlag")
    return df, details


def parse_args():
    parser = argparse.ArgumentParser(description="Run Part-3 batch noise scans.")
    parser.add_argument("--skip-a", action="store_true")
    parser.add_argument("--skip-b", action="store_true")
    parser.add_argument("--skip-ab", action="store_true")
    parser.add_argument("--skip-c", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    workdir = Path(__file__).resolve().parent
    figs_root = workdir / "figs"
    figs_root.mkdir(parents=True, exist_ok=True)

    a_df = pd.DataFrame()
    b_df = pd.DataFrame()
    c_df = pd.DataFrame()
    a_details = []
    b_details = []
    c_details = []

    if not args.skip_a:
        a_df, a_details = run_scan_a(figs_root, DEFAULT_CONFIG)
    if not args.skip_b:
        b_df, b_details = run_scan_b(figs_root, DEFAULT_CONFIG)
    if not args.skip_c:
        c_df, c_details = run_scan_c(figs_root, DEFAULT_CONFIG)

    report_path = workdir / "噪音实验.md"
    report_path.write_text(build_report(a_df, a_details, b_df, b_details, c_df), encoding="utf-8")

    print("Batch noise scan completed.")
    print(report_path)


if __name__ == "__main__":
    main()
