import argparse
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


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

# tools.py 顶层依赖 plotly，但本实验不需要它；这里用轻量 stub 避免环境缺包阻塞。
if "plotly" not in sys.modules:
    plotly_module = types.ModuleType("plotly")
    plotly_module.express = types.ModuleType("plotly.express")
    plotly_module.graph_objects = types.ModuleType("plotly.graph_objects")
    sys.modules["plotly"] = plotly_module
    sys.modules["plotly.express"] = plotly_module.express
    sys.modules["plotly.graph_objects"] = plotly_module.graph_objects

from tools import (  # noqa: E402
    add_gaussian_noise,
    apply_coarse_graining,
    build_w_from_svd,
    compute_ce_from_micro_macro,
    compute_gis_metrics,
    compute_prediction_errors,
    estimate_covariance_from_residuals,
    make_step_system_matrix,
    observable_step,
    simulate_discrete_system,
    step_map,
)


plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 160
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
sns.set_theme(style="whitegrid")

HEATMAP_CMAP = "vlag"
FEATURE_NAMES = ["$x$", "$y$", "$x^2$"]
DEFAULT_CONFIG = {
    "experiment_name": "exp_ana_gis_part3_noise_scan",
    "lam": 0.1,
    "mu": 0.9,
    "initial_state": [5.0, 5.0],
    "steps": 600,
    "dt": 1.0,
    "tau": 1,
    "delta": None,
    "alpha": 1.0,
    "noise_seed": 42,
    "eps": 1e-10,
    "ridge": 1e-10,
    "manual_r": 1,
    "horizons": (1, 3, 5),
}
DEFAULT_NOISE_VALUES = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6]


def sparse_labels(labels, step=1):
    if labels is None:
        return False
    if step <= 1:
        return labels
    return [label if i % step == 0 else "" for i, label in enumerate(labels)]


def standardize_for_plot(x):
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def noise_to_tag(noise_value):
    return f"{noise_value:.3f}"


def save_figure(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_heatmap_to_file(
    matrix,
    title,
    output_path,
    row_labels=None,
    col_labels=None,
    center=0.0,
    figsize=(6, 6),
    label_step=1,
    cmap=HEATMAP_CMAP,
):
    fig, ax = plt.subplots(figsize=figsize)
    matrix_arr = np.asarray(matrix)
    sns.heatmap(
        matrix_arr,
        ax=ax,
        cmap=cmap,
        center=center,
        square=matrix_arr.shape[0] == matrix_arr.shape[1],
        xticklabels=sparse_labels(col_labels, label_step),
        yticklabels=sparse_labels(row_labels, label_step),
    )
    ax.set_title(title)
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

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, value in enumerate(sorted_values):
        color = "tab:blue" if sorted_labels[i] == "forward" else "tab:orange"
        ax.bar(x[i], value, color=color, alpha=0.45)

    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=10, label=forward_label),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=10, label=backward_label),
        ]
    )
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Values")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_stage2_spectrum_to_file(singular_values, title, output_path):
    singular_values = np.asarray(singular_values, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        np.arange(1, len(singular_values) + 1),
        singular_values,
        marker="o",
        color="tab:green",
        linewidth=1.8,
    )
    ax.set_title(title)
    ax.set_xlabel("奇异值序号")
    ax.set_ylabel("奇异值")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_prediction_curves_to_file(series_names, predictions, targets, title, output_path, sample_count=80):
    fig, ax = plt.subplots(figsize=(10, 4))
    limit = min(sample_count, len(predictions))
    x = np.arange(limit)
    for idx, name in enumerate(series_names):
        ax.plot(x, targets[:limit, idx], label=f"true {name}")
        ax.plot(x, predictions[:limit, idx], "--", linewidth=1.6, label=f"pred {name}")
    ax.set_title(title)
    ax.set_xlabel("Pair index")
    ax.set_ylabel("Value")
    ax.legend(ncol=max(1, min(3, len(series_names))))
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_micro_macro_curve_compare_to_file(micro_series, macro_series, macro_names, title, output_path, sample_count=120):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    limit = min(sample_count, len(micro_series), len(macro_series))
    x = np.arange(limit)
    for idx, name in enumerate(FEATURE_NAMES):
        ax.plot(x, standardize_for_plot(micro_series[:limit, idx]), linewidth=1.4, label=f"micro: {name}")
    for idx, name in enumerate(macro_names):
        ax.plot(x, standardize_for_plot(macro_series[:limit, idx]), "--", linewidth=2.2, label=f"macro: {name}")
    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Standardized value")
    ax.legend(ncol=2)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_summary_ce_j_to_file(results_df, output_path):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = results_df["noise_scale"].to_numpy()
    ax.plot(x, results_df["CE"], marker="o", linewidth=2.0, label="CE")
    ax.plot(x, results_df["micro_J_alpha"], marker="s", linewidth=1.8, label="micro J_alpha")
    ax.plot(x, results_df["macro_J_alpha"], marker="^", linewidth=1.8, label="macro J_alpha")
    ax.set_title("CE、微观/宏观 J_alpha 随噪音变化")
    ax.set_xlabel("noise_scale")
    ax.set_ylabel("value")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_summary_d_n_to_file(results_df, output_path):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = results_df["noise_scale"].to_numpy()
    ax.plot(x, results_df["micro_D"], marker="o", linewidth=1.8, label="micro D")
    ax.plot(x, results_df["micro_N"], marker="s", linewidth=1.8, label="micro N")
    ax.plot(x, results_df["macro_D"], marker="^", linewidth=1.8, label="macro D")
    ax.plot(x, results_df["macro_N"], marker="d", linewidth=1.8, label="macro N")
    ax.set_title("微观/宏观 D 和 N 随噪音变化")
    ax.set_xlabel("noise_scale")
    ax.set_ylabel("value")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_summary_prediction_errors_to_file(results_df, horizons, output_path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = results_df["noise_scale"].to_numpy()
    markers = {1: "o", 3: "s", 5: "^"}
    for horizon in horizons:
        ax.plot(
            x,
            results_df[f"micro_E{horizon}"],
            marker=markers.get(horizon, "o"),
            linewidth=1.8,
            label=f"micro E{horizon}",
        )
        ax.plot(
            x,
            results_df[f"macro_E{horizon}"],
            marker=markers.get(horizon, "o"),
            linewidth=1.8,
            linestyle="--",
            label=f"macro E{horizon}",
        )
    ax.set_title("微观/宏观单步与多步预测误差随噪音变化")
    ax.set_xlabel("noise_scale")
    ax.set_ylabel("mean squared error")
    ax.legend(ncol=2)
    fig.tight_layout()
    save_figure(fig, output_path)


def build_markdown_report(results_df, details, summary_dir):
    best_idx = results_df["CE"].idxmax()
    worst_idx = results_df["CE"].idxmin()
    best_row = results_df.loc[best_idx]
    worst_row = results_df.loc[worst_idx]

    ce_trend = np.polyfit(results_df["noise_scale"], results_df["CE"], deg=1)[0]
    micro_e1_trend = np.polyfit(results_df["noise_scale"], results_df["micro_E1"], deg=1)[0]
    macro_e1_trend = np.polyfit(results_df["noise_scale"], results_df["macro_E1"], deg=1)[0]

    trend_text = "整体下降" if ce_trend < -1e-9 else "整体上升" if ce_trend > 1e-9 else "整体较平"
    error_text = (
        "微观和宏观单步误差都随噪音增大而上升"
        if micro_e1_trend > 0 and macro_e1_trend > 0
        else "预测误差随噪音变化并非完全单调"
    )

    lines = [
        "# 第三部分噪音强度扫描实验",
        "",
        "## 实验设置",
        "",
        f"- 扫描对象：第三部分含噪实验的 `noise_scale`",
        f"- 噪音取值：{', '.join(noise_to_tag(v) for v in results_df['noise_scale'])}",
        f"- 固定参数：`lam={DEFAULT_CONFIG['lam']}`，`mu={DEFAULT_CONFIG['mu']}`，`tau={DEFAULT_CONFIG['tau']}`，`alpha={DEFAULT_CONFIG['alpha']}`，`manual_r={DEFAULT_CONFIG['manual_r']}`，`horizons={DEFAULT_CONFIG['horizons']}`",
        "",
        "## 汇总图",
        "",
        f"![CE 与 J_alpha](./figs/{summary_dir.name}/summary_ce_j_alpha.png)",
        "",
        f"![D 与 N](./figs/{summary_dir.name}/summary_d_n.png)",
        "",
        f"![预测误差](./figs/{summary_dir.name}/summary_prediction_errors.png)",
        "",
        "## 结果结论",
        "",
        f"- `CE` 随噪音变化的总体趋势为：{trend_text}。",
        f"- `CE` 最大值出现在 `noise_scale={noise_to_tag(best_row['noise_scale'])}`，此时 `CE={best_row['CE']:.6f}`。",
        f"- `CE` 最小值出现在 `noise_scale={noise_to_tag(worst_row['noise_scale'])}`，此时 `CE={worst_row['CE']:.6f}`。",
        f"- {error_text}。",
        "- 是否出现“因果涌现”以 `CE > 0`、宏观 `J_alpha` 高于微观 `J_alpha`，且宏观预测误差没有明显恶化为主要判断依据。",
        "",
        "## 数值汇总",
        "",
        dataframe_to_markdown(results_df),
        "",
        "## 各噪音值结果",
        "",
    ]

    for detail in details:
        noise_tag = detail["noise_tag"]
        exp_dir = Path("figs") / noise_tag
        micro_metrics = detail["micro_metrics"]
        macro_metrics = detail["macro_metrics"]
        micro_errors = detail["micro_errors"]
        macro_errors = detail["macro_errors"]
        ce_result = detail["ce_result"]
        lines.extend(
            [
                f"### noise_scale = {noise_tag}",
                "",
                "#### 图",
                "",
                f"![微观Sigma](./{exp_dir.as_posix()}/micro_sigma_noise_{noise_tag}.png)",
                "",
                f"![微观预测](./{exp_dir.as_posix()}/micro_prediction_noise_{noise_tag}.png)",
                "",
                f"![Sorted SVD Spectrum](./{exp_dir.as_posix()}/sorted_svd_spectrum_noise_{noise_tag}.png)",
                "",
                f"![第二次矩阵热力图](./{exp_dir.as_posix()}/stage2_matrix_noise_{noise_tag}.png)",
                "",
                f"![第二次矩阵奇异值谱](./{exp_dir.as_posix()}/stage2_spectrum_noise_{noise_tag}.png)",
                "",
                f"![W热力图](./{exp_dir.as_posix()}/W_heatmap_noise_{noise_tag}.png)",
                "",
                f"![宏观预测](./{exp_dir.as_posix()}/macro_prediction_noise_{noise_tag}.png)",
                "",
                f"![宏微观曲线对比](./{exp_dir.as_posix()}/micro_macro_curve_noise_{noise_tag}.png)",
                "",
                "#### 数值",
                "",
                f"- 微观 GIS 指标：`Gamma={micro_metrics['Gamma']:.6f}`，`log_Gamma={micro_metrics['log_Gamma']:.6f}`，`J_alpha={micro_metrics['J_alpha']:.6f}`，`D={micro_metrics['D']:.6f}`，`N={micro_metrics['N']:.6f}`",
                f"- 宏观 GIS 指标：`Gamma={macro_metrics['Gamma']:.6f}`，`log_Gamma={macro_metrics['log_Gamma']:.6f}`，`J_alpha={macro_metrics['J_alpha']:.6f}`，`D={macro_metrics['D']:.6f}`，`N={macro_metrics['N']:.6f}`",
                f"- CE：`{ce_result['CE']:.6f}`",
                f"- 微观预测误差：`E1={micro_errors[1]['mean_error']:.6f}`，`E3={micro_errors[3]['mean_error']:.6f}`，`E5={micro_errors[5]['mean_error']:.6f}`",
                f"- 宏观预测误差：`E1={macro_errors[1]['mean_error']:.6f}`，`E3={macro_errors[3]['mean_error']:.6f}`，`E5={macro_errors[5]['mean_error']:.6f}`",
                "",
            ]
        )

    return "\n".join(lines)


def dataframe_to_markdown(df):
    headers = list(df.columns)
    align = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
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


def run_single_noise_experiment(noise_scale, config, figs_root):
    noise_tag = noise_to_tag(noise_scale)
    noise_dir = figs_root / noise_tag
    noise_dir.mkdir(parents=True, exist_ok=True)

    clean_sim = simulate_discrete_system(
        step_map,
        config["initial_state"],
        steps=config["steps"],
        system_kwargs={"lam": config["lam"], "mu": config["mu"]},
        dt=config["dt"],
    )
    clean_xy = clean_sim["trajectories"][0]

    noisy_xy = add_gaussian_noise(
        clean_xy,
        noise_scale=noise_scale,
        cov=None,
        random_state=config["noise_seed"],
    )["noisy_data"]

    obs_clean = observable_step(clean_xy, mode="default")
    obs_noisy = observable_step(noisy_xy, mode="default")
    obs_noise = obs_noisy - obs_clean

    sigma_micro = estimate_covariance_from_residuals(
        obs_noise,
        center=True,
        regularization=config["eps"],
    )
    a_micro = make_step_system_matrix(config["lam"], config["mu"])
    micro_metrics = compute_gis_metrics(
        a_micro,
        sigma_micro,
        alpha=config["alpha"],
        eps=config["eps"],
    )
    micro_errors = compute_prediction_errors(
        a_micro,
        obs_noisy,
        tau=config["tau"],
        horizons=config["horizons"],
    )

    w_result = build_w_from_svd(
        a_micro,
        sigma_micro,
        r=config["manual_r"],
        alpha=config["alpha"],
        eps=config["eps"],
        mode="two_stage",
    )
    w_matrix = w_result["W"]
    macro_names = [f"$z_{i+1}$" for i in range(w_result["r"])]
    z_series = apply_coarse_graining(w_matrix, obs_noisy)

    a_macro = w_matrix @ a_micro @ w_matrix.T
    sigma_macro = w_matrix @ sigma_micro @ w_matrix.T
    macro_metrics = compute_gis_metrics(
        a_macro,
        sigma_macro,
        alpha=config["alpha"],
        eps=config["eps"],
    )
    macro_errors = compute_prediction_errors(
        a_macro,
        z_series,
        tau=config["tau"],
        horizons=config["horizons"],
    )
    ce_result = compute_ce_from_micro_macro(micro_metrics, macro_metrics)

    plot_matrix_heatmap_to_file(
        sigma_micro,
        f"第三部分：微观层 Sigma，noise={noise_tag}",
        noise_dir / f"micro_sigma_noise_{noise_tag}.png",
        row_labels=FEATURE_NAMES,
        col_labels=FEATURE_NAMES,
        center=None,
        label_step=1,
    )
    plot_prediction_curves_to_file(
        FEATURE_NAMES,
        micro_errors[1]["predictions"],
        micro_errors[1]["targets"],
        f"第三部分：微观层单步预测，noise={noise_tag}",
        noise_dir / f"micro_prediction_noise_{noise_tag}.png",
    )
    plot_sorted_svd_spectrum_to_file(
        micro_metrics["sv_forward"],
        micro_metrics["sv_backward"],
        f"第三部分：Sorted SVD Spectrum，noise={noise_tag}",
        noise_dir / f"sorted_svd_spectrum_noise_{noise_tag}.png",
    )
    plot_matrix_heatmap_to_file(
        w_result["basis_info"]["weighted_matrix"],
        f"第三部分：第二次分解矩阵，noise={noise_tag}",
        noise_dir / f"stage2_matrix_noise_{noise_tag}.png",
        row_labels=FEATURE_NAMES,
        col_labels=[f"$c_{i+1}$" for i in range(w_result["basis_info"]["weighted_matrix"].shape[1])],
        center=0.0,
        figsize=(6, 4),
        cmap=HEATMAP_CMAP,
    )
    plot_stage2_spectrum_to_file(
        w_result["sv_info"]["sv_stage2"],
        f"第三部分：第二次分解奇异值谱，noise={noise_tag}",
        noise_dir / f"stage2_spectrum_noise_{noise_tag}.png",
    )
    plot_matrix_heatmap_to_file(
        np.abs(w_matrix),
        f"第三部分：W 热力图，noise={noise_tag}",
        noise_dir / f"W_heatmap_noise_{noise_tag}.png",
        row_labels=macro_names,
        col_labels=FEATURE_NAMES,
        center=None,
        figsize=(5, 3),
        cmap="Blues",
    )
    plot_prediction_curves_to_file(
        macro_names,
        macro_errors[1]["predictions"],
        macro_errors[1]["targets"],
        f"第三部分：宏观层单步预测，noise={noise_tag}",
        noise_dir / f"macro_prediction_noise_{noise_tag}.png",
    )
    plot_micro_macro_curve_compare_to_file(
        obs_noisy,
        z_series,
        macro_names,
        f"第三部分：宏微观曲线对比，noise={noise_tag}",
        noise_dir / f"micro_macro_curve_noise_{noise_tag}.png",
    )

    row = {
        "noise_scale": float(noise_scale),
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
    }
    for horizon in config["horizons"]:
        row[f"micro_E{horizon}"] = micro_errors[horizon]["mean_error"]
        row[f"macro_E{horizon}"] = macro_errors[horizon]["mean_error"]

    detail = {
        "noise_tag": noise_tag,
        "noise_dir": noise_dir,
        "micro_metrics": micro_metrics,
        "macro_metrics": macro_metrics,
        "micro_errors": micro_errors,
        "macro_errors": macro_errors,
        "ce_result": ce_result,
    }
    return row, detail


def parse_args():
    parser = argparse.ArgumentParser(description="Run part-3 noise-strength scan for exp_ana_gis.")
    parser.add_argument(
        "--noise-values",
        type=float,
        nargs="*",
        default=DEFAULT_NOISE_VALUES,
        help="Noise scale values to scan.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    workdir = Path(__file__).resolve().parent
    figs_root = workdir / "figs"
    summary_dir = figs_root / "噪音汇总"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    details = []
    for noise_scale in args.noise_values:
        row, detail = run_single_noise_experiment(noise_scale, DEFAULT_CONFIG, figs_root)
        rows.append(row)
        details.append(detail)

    results_df = pd.DataFrame(rows).sort_values("noise_scale").reset_index(drop=True)

    plot_summary_ce_j_to_file(results_df, summary_dir / "summary_ce_j_alpha.png")
    plot_summary_d_n_to_file(results_df, summary_dir / "summary_d_n.png")
    plot_summary_prediction_errors_to_file(
        results_df,
        DEFAULT_CONFIG["horizons"],
        summary_dir / "summary_prediction_errors.png",
    )

    report_md = build_markdown_report(results_df, details, summary_dir)
    output_md = workdir / "噪音实验.md"
    output_md.write_text(report_md, encoding="utf-8")

    print("Noise scan completed.")
    print(f"Markdown report: {output_md}")
    print(f"Summary figures: {summary_dir}")


if __name__ == "__main__":
    main()
