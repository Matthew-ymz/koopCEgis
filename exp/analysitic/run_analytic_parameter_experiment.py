from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from tools import (
    make_step_system_matrix,
    step_map,
    observable_step,
    simulate_discrete_system,
    compute_gis_metrics,
    compute_prediction_errors,
    compute_ce_from_gis_metrics,
    build_w_from_svd,
    apply_coarse_graining,
    make_analytic_sigma_matrix,
    check_analytic_sigma_validity,
    compute_macro_true_matrices,
)


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "DejaVu Serif"

BASE_DIR = REPO_ROOT / "exp" / "analysitic"
FIG_DIR = BASE_DIR / "figs"
SUMMARY_DIR = FIG_DIR / "summary"
REPORT_PATH = BASE_DIR / "参数实验.md"
CSV_PATH = SUMMARY_DIR / "parameter_scan_results.csv"

FEATURE_NAMES = ["x", "y", "x^2"]
STATE_NAMES = ["x", "y"]
INITIAL_STATE = np.array([0.35, -0.20], dtype=float)
STEPS = 90
TAU = 1
HORIZONS = (1, 3, 5)
ALPHA = 1.0
EPS = 1e-10
R_FIXED = 1


def ensure_dirs() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def fmt_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(round(float(x))))
    text = f"{float(x):.6g}"
    return text.replace(".", "p")


def case_dir_name(row: dict) -> str:
    return f"lam_{fmt_num(row['lam'])}_mu_{fmt_num(row['mu'])}_b_{fmt_num(row['b'])}_c_{fmt_num(row['c'])}"


def to_markdown_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if columns is not None:
        df = df.loc[:, columns]
    if max_rows is not None:
        df = df.head(max_rows)
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        items = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                items.append(f"{float(value):.6f}")
            else:
                items.append(str(value))
        lines.append("| " + " | ".join(items) + " |")
    return "\n".join(lines)


def plot_dual_spectrum_save(forward_values, backward_values, title: str, save_path: Path) -> None:
    forward_values = np.asarray(forward_values, dtype=float).ravel()
    backward_values = np.asarray(backward_values, dtype=float).ravel()
    k = max(len(forward_values), len(backward_values))
    x = np.arange(1, k + 1)
    width = 0.36

    forward_plot = np.full(k, np.nan)
    backward_plot = np.full(k, np.nan)
    forward_plot[: len(forward_values)] = forward_values
    backward_plot[: len(backward_values)] = backward_values

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x - width / 2, forward_plot, width=width, color="tab:blue", alpha=0.45, label="$\\Sigma^{-1}$")
    ax.bar(x + width / 2, backward_plot, width=width, color="tab:orange", alpha=0.45, label="$A^T\\Sigma^{-1}A$")
    ax.plot(x - width / 2, forward_plot, color="tab:blue", marker="o", linewidth=1.8)
    ax.plot(x + width / 2, backward_plot, color="tab:orange", marker="s", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("Singular value")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_save(
    matrix,
    title: str,
    save_path: Path,
    row_labels: list[str],
    col_labels: list[str],
    center: float | None = 0.0,
    cmap: str = "vlag",
    annot: bool = True,
    decimals: int = 3,
    figsize=(6, 4),
) -> None:
    matrix_arr = np.asarray(matrix, dtype=float)
    final_center = center if cmap == "vlag" else None
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix_arr,
        ax=ax,
        cmap=cmap,
        center=final_center,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=annot,
        fmt=f".{decimals}f",
        square=matrix_arr.shape[0] == matrix_arr.shape[1],
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def standardize_for_plot(x):
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def plot_macro_observation_compare(obs, macro, title: str, save_path: Path, n_plot: int = 120) -> None:
    n_plot = int(min(n_plot, len(obs), len(macro)))
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for idx, name in enumerate(FEATURE_NAMES):
        ax.plot(np.arange(n_plot), standardize_for_plot(obs[:n_plot, idx]), linewidth=1.5, label=f"obs: {name}")
    ax.plot(np.arange(n_plot), standardize_for_plot(macro[:n_plot, 0]), "--", linewidth=2.4, label="macro: z1")
    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Standardized value")
    ax.legend(ncol=2)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_case(lam: float, mu: float, b: float, c: float) -> dict | None:
    valid = check_analytic_sigma_validity(b, c, allow_singular=False)
    if not valid["is_valid"]:
        return None

    A = make_step_system_matrix(lam, mu)
    Sigma = make_analytic_sigma_matrix(b, c)

    sim = simulate_discrete_system(
        step_map,
        INITIAL_STATE,
        steps=STEPS,
        system_kwargs={"lam": lam, "mu": mu},
        dt=1.0,
    )
    xy = sim["trajectories"][0]
    obs = observable_step(xy, mode="default")
    if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(obs)):
        return None
    if np.max(np.abs(obs)) > 1e8:
        return None

    metrics = compute_gis_metrics(A, Sigma, alpha=ALPHA, eps=EPS)
    ce_result = compute_ce_from_gis_metrics(metrics, r_eps=R_FIXED, eps=EPS)
    w_result = build_w_from_svd(A, Sigma, r=R_FIXED, alpha=ALPHA, eps=EPS, mode="two_stage")
    W = np.asarray(w_result["W"], dtype=float)
    z = apply_coarse_graining(W, obs)
    macro_pack = compute_macro_true_matrices(A, Sigma, W)
    A_macro = macro_pack["A_macro"]

    micro_errors = compute_prediction_errors(A, obs, tau=TAU, horizons=HORIZONS)
    macro_errors = compute_prediction_errors(A_macro, z, tau=TAU, horizons=HORIZONS)

    sv_forward = np.asarray(metrics["sv_forward"], dtype=float)
    sv_backward = np.asarray(metrics["sv_backward"], dtype=float)
    stage2_sv = np.asarray(w_result["sv_info"]["sv_stage2"], dtype=float)
    w_abs = np.abs(W[0])

    forward_gap12 = float(sv_forward[0] / max(sv_forward[1], EPS))
    backward_gap12 = float(sv_backward[0] / max(sv_backward[1], EPS))
    min_primary_gap = float(min(forward_gap12, backward_gap12))
    y_dominance_ratio = float(w_abs[1] / max(max(w_abs[0], w_abs[2]), EPS))
    slow_ratio = float(mu / max(lam, EPS))
    ce_value = float(ce_result["CE"])
    macro_e1 = float(macro_errors[1]["mean_error"])
    micro_e1 = float(micro_errors[1]["mean_error"])

    pass_gap = min_primary_gap >= 2.1
    pass_w = y_dominance_ratio >= 0.85
    pass_slow = slow_ratio >= 1.5
    pass_ce = ce_value > -0.05
    pass_macro = np.isfinite(macro_e1) and macro_e1 < 0.1
    selected = pass_gap and pass_w and pass_slow and pass_ce and pass_macro

    composite_score = (
        2.5 * min_primary_gap
        + 2.0 * y_dominance_ratio
        + 0.8 * slow_ratio
        + 1.2 * max(ce_value, -0.2)
        - 0.5 * math.log10(1.0 + max(micro_e1, 0.0))
        - 0.8 * math.log10(1.0 + max(macro_e1, 0.0))
    )

    row = {
        "lam": lam,
        "mu": mu,
        "b": b,
        "c": c,
        "slow_ratio_mu_over_lam": slow_ratio,
        "forward_sv1": sv_forward[0],
        "forward_sv2": sv_forward[1],
        "forward_sv3": sv_forward[2],
        "backward_sv1": sv_backward[0],
        "backward_sv2": sv_backward[1],
        "backward_sv3": sv_backward[2],
        "forward_gap12": forward_gap12,
        "backward_gap12": backward_gap12,
        "min_primary_gap": min_primary_gap,
        "ce": ce_value,
        "gamma_hat": ce_result["gamma_hat"],
        "gamma_hat_eps": ce_result["gamma_hat_eps"],
        "stage2_sv1": stage2_sv[0] if len(stage2_sv) > 0 else np.nan,
        "stage2_sv2": stage2_sv[1] if len(stage2_sv) > 1 else np.nan,
        "w_abs_x": w_abs[0],
        "w_abs_y": w_abs[1],
        "w_abs_x2": w_abs[2],
        "w_y_is_max": pass_w,
        "w_y_ratio": y_dominance_ratio,
        "micro_E1": micro_errors[1]["mean_error"],
        "micro_E3": micro_errors[3]["mean_error"],
        "micro_E5": micro_errors[5]["mean_error"],
        "macro_E1": macro_errors[1]["mean_error"],
        "macro_E3": macro_errors[3]["mean_error"],
        "macro_E5": macro_errors[5]["mean_error"],
        "pass_gap": pass_gap,
        "pass_w": pass_w,
        "pass_slow": pass_slow,
        "pass_ce": pass_ce,
        "pass_macro": pass_macro,
        "selected": selected,
        "composite_score": composite_score,
        "A": A,
        "Sigma": Sigma,
        "obs": obs,
        "xy": xy,
        "z": z,
        "metrics": metrics,
        "w_result": w_result,
    }
    return row


def save_case_outputs(row: dict) -> dict[str, str]:
    folder = FIG_DIR / case_dir_name(row)
    folder.mkdir(parents=True, exist_ok=True)

    spectrum_path = folder / "dual_spectrum.png"
    stage2_matrix_path = folder / "stage2_weighted_matrix.png"
    w_heatmap_path = folder / "W_heatmap.png"
    curve_path = folder / "macro_observation_compare.png"

    title_suffix = (
        f"$\\lambda$={row['lam']:.3g}, $\\mu$={row['mu']:.3g}, "
        f"b={row['b']:.3g}, c={row['c']:.3g}"
    )
    plot_dual_spectrum_save(
        row["metrics"]["sv_forward"],
        row["metrics"]["sv_backward"],
        f"Dual spectrum ({title_suffix})",
        spectrum_path,
    )
    weighted = row["w_result"]["basis_info"]["weighted_matrix"]
    plot_heatmap_save(
        weighted,
        f"Stage-2 weighted matrix ({title_suffix})",
        stage2_matrix_path,
        row_labels=FEATURE_NAMES,
        col_labels=[f"c{i+1}" for i in range(weighted.shape[1])],
        center=0.0,
        cmap="vlag",
        annot=True,
        decimals=3,
        figsize=(6, 4),
    )
    plot_heatmap_save(
        np.abs(row["w_result"]["W"]),
        f"|W| heatmap ({title_suffix})",
        w_heatmap_path,
        row_labels=["z1"],
        col_labels=FEATURE_NAMES,
        center=None,
        cmap="Blues",
        annot=True,
        decimals=3,
        figsize=(5, 2.6),
    )
    plot_macro_observation_compare(
        row["obs"],
        row["z"],
        f"Macro vs observables ({title_suffix})",
        curve_path,
    )
    return {
        "folder": folder.relative_to(BASE_DIR).as_posix(),
        "dual_spectrum": spectrum_path.relative_to(BASE_DIR).as_posix(),
        "stage2_matrix": stage2_matrix_path.relative_to(BASE_DIR).as_posix(),
        "W_heatmap": w_heatmap_path.relative_to(BASE_DIR).as_posix(),
        "macro_curve": curve_path.relative_to(BASE_DIR).as_posix(),
    }


def save_summary_figures(df: pd.DataFrame) -> dict[str, str]:
    paths: dict[str, str] = {}

    top = df.sort_values("composite_score", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [case_dir_name(row) for _, row in top.iterrows()]
    ax.bar(np.arange(len(top)), top["composite_score"], color="tab:blue", alpha=0.8)
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_title("Top composite-score parameter cases")
    ax.set_ylabel("Composite score")
    plt.tight_layout()
    path = SUMMARY_DIR / "summary_top_composite.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths["top_composite"] = path.relative_to(BASE_DIR).as_posix()

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    scatter = ax.scatter(
        df["min_primary_gap"],
        df["w_abs_y"],
        c=df["ce"],
        s=40 + 50 * np.clip(df["slow_ratio_mu_over_lam"], 0, 6),
        cmap="coolwarm",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.axvline(1.8, color="gray", linestyle="--", linewidth=1.0, label="gap threshold")
    ax.set_xlabel("Minimum primary spectral gap")
    ax.set_ylabel("|W_y|")
    ax.set_title("Spectral gap vs slow-variable dominance")
    ax.legend(loc="lower right")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("CE")
    plt.tight_layout()
    path = SUMMARY_DIR / "summary_gap_vs_wy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths["gap_vs_wy"] = path.relative_to(BASE_DIR).as_posix()

    selected = df[df["selected"]].copy()
    heat_source = selected if not selected.empty else df.sort_values("composite_score", ascending=False).head(30)
    heat = (
        heat_source.pivot_table(index="mu", columns="lam", values="composite_score", aggfunc="max")
        .sort_index(ascending=False)
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    sns.heatmap(heat, ax=ax, cmap="YlGnBu", annot=True, fmt=".2f")
    ax.set_title("Best composite score by $(\\lambda, \\mu)$")
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("$\\mu$")
    plt.tight_layout()
    path = SUMMARY_DIR / "summary_lambda_mu_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths["lambda_mu_heatmap"] = path.relative_to(BASE_DIR).as_posix()

    return paths


def write_report(df: pd.DataFrame, summary_paths: dict[str, str], detailed_rows: pd.DataFrame) -> None:
    selected = df[df["selected"]].sort_values("composite_score", ascending=False)
    top = df.sort_values("composite_score", ascending=False).head(12)

    report_lines = [
        "# 参数实验",
        "",
        "本轮实验针对解析有噪声动力学扫描参数 `lam / mu / b / c`，固定宏观维度 `r = 1`，并重点寻找同时满足以下要求的参数：",
        "",
        "- 谱截断清晰：`Sigma^{-1}` 与 `A^T Sigma^{-1} A` 的第一奇异值显著大于后两个。",
        "- 粗粒化方向偏向慢变量：`|W_y|` 尽量大，并优先大于 `|W_x|` 与 `|W_{x^2}|`。",
        "- 动力学具有快慢变量结构：优先保留 `mu > lam` 且 `mu / lam` 较大的情形。",
        "- `CE` 不过低，同时微观与宏观的一步预测误差保持在可接受范围内。",
        "",
        "本轮筛选约定：",
        "",
        "- 约束：`lam > 0`, `mu > 0`, `b > c >= 0`。",
        "- 固定 `r = 1`。",
        "- 经验筛选阈值：`min(forward_gap12, backward_gap12) >= 2.1`，也就是两组谱都能用同一条截断线稳定截出 1 维；同时要求 `|W_y| / max(|W_x|, |W_{x^2}|) >= 0.85`，`mu/lam >= 1.5`，`CE > -0.05`，且 `macro_E1 < 0.1`。",
        "",
        f"- 共评估参数组数：`{len(df)}`。",
        f"- 满足筛选条件的参数组数：`{len(selected)}`。",
        "",
        "## 汇总图",
        "",
        f"![Top Composite](./{summary_paths['top_composite']})",
        "",
        f"![Gap vs Wy](./{summary_paths['gap_vs_wy']})",
        "",
        f"![Lambda Mu Heatmap](./{summary_paths['lambda_mu_heatmap']})",
        "",
        "## 最优参数候选",
        "",
        to_markdown_table(
            top.drop(columns=["A", "Sigma", "obs", "xy", "z", "metrics", "w_result"], errors="ignore"),
            columns=[
                "lam",
                "mu",
                "b",
                "c",
                "min_primary_gap",
                "w_abs_y",
                "w_y_ratio",
                "ce",
                "micro_E1",
                "macro_E1",
                "composite_score",
                "selected",
            ],
            max_rows=12,
        ),
        "",
        "## 通过筛选的参数",
        "",
        to_markdown_table(
            selected.drop(columns=["A", "Sigma", "obs", "xy", "z", "metrics", "w_result"], errors="ignore"),
            columns=[
                "lam",
                "mu",
                "b",
                "c",
                "slow_ratio_mu_over_lam",
                "forward_gap12",
                "backward_gap12",
                "w_abs_x",
                "w_abs_y",
                "w_abs_x2",
                "ce",
                "micro_E1",
                "macro_E1",
                "composite_score",
            ],
        ) if not selected.empty else "本轮没有参数组同时满足全部筛选条件。",
        "",
        "## 详细图保存",
        "",
        "下面列出已保存详细图片的参数组。每组都保留四张图：双谱图、第二步加权矩阵热力图、`W` 热力图、宏观观测数据对比图。",
        "",
    ]

    for _, row in detailed_rows.iterrows():
        report_lines.extend(
            [
                f"### `{case_dir_name(row)}`",
                "",
                f"- 参数：`lam={row['lam']:.6g}`, `mu={row['mu']:.6g}`, `b={row['b']:.6g}`, `c={row['c']:.6g}`",
                f"- 指标：`gap={row['min_primary_gap']:.4f}`, `|W_y|={row['w_abs_y']:.4f}`, `CE={row['ce']:.4f}`, `micro_E1={row['micro_E1']:.4f}`, `macro_E1={row['macro_E1']:.4f}`",
                "",
                f"![Dual Spectrum](./{row['dual_spectrum_path']})",
                "",
                f"![Stage2 Matrix](./{row['stage2_matrix_path']})",
                "",
                f"![W Heatmap](./{row['w_heatmap_path']})",
                "",
                f"![Macro Curve](./{row['macro_curve_path']})",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()

    lam_values = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.20]
    mu_values = [0.30, 0.50, 0.70, 0.90, 1.20, 1.60, 2.00, 3.00]
    b_values = [0.20, 0.50, 1.00, 2.00, 5.00]
    c_fraction_values = [0.0, 0.25, 0.50, 0.70, 0.80, 0.90]

    results: list[dict] = []
    for lam in lam_values:
        for mu in mu_values:
            for b in b_values:
                for frac in c_fraction_values:
                    c = b * frac
                    if not (lam > 0 and mu > 0 and b > c >= 0):
                        continue
                    row = evaluate_case(lam=lam, mu=mu, b=b, c=c)
                    if row is not None:
                        results.append(row)

    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError("没有有效参数组合被成功评估。")

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    CSV_PATH.write_text(df.drop(columns=["A", "Sigma", "obs", "xy", "z", "metrics", "w_result"]).to_csv(index=False), encoding="utf-8")

    selected = df[df["selected"]].copy()
    detailed = selected.head(12) if not selected.empty else df.head(12)

    detail_rows = []
    for _, row in detailed.iterrows():
        save_info = save_case_outputs(row.to_dict())
        info = row.drop(labels=["A", "Sigma", "obs", "xy", "z", "metrics", "w_result"]).to_dict()
        info["dual_spectrum_path"] = save_info["dual_spectrum"]
        info["stage2_matrix_path"] = save_info["stage2_matrix"]
        info["w_heatmap_path"] = save_info["W_heatmap"]
        info["macro_curve_path"] = save_info["macro_curve"]
        detail_rows.append(info)

    detail_df = pd.DataFrame(detail_rows)
    summary_paths = save_summary_figures(df)
    write_report(df, summary_paths, detail_df)

    print(f"Evaluated {len(df)} parameter cases.")
    print(f"Selected {int(df['selected'].sum())} cases.")
    print(f"Wrote report to {REPORT_PATH}")
    print(f"Wrote csv to {CSV_PATH}")


if __name__ == "__main__":
    main()
