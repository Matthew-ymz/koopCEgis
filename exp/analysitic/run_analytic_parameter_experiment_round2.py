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
import numpy as np
import pandas as pd
import seaborn as sns

from tools import (
    make_step_system_matrix,
    step_map,
    observable_step,
    simulate_discrete_system,
    compute_gis_metrics,
    compute_ce_from_gis_metrics,
    build_w_from_svd,
    apply_coarse_graining,
    make_analytic_sigma_matrix,
    check_analytic_sigma_validity,
)


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "DejaVu Serif"

BASE_DIR = REPO_ROOT / "exp" / "analysitic"
FIG_DIR = BASE_DIR / "figs"
SUMMARY_DIR = FIG_DIR / "summary_round2"
REPORT_PATH = BASE_DIR / "参数实验_第二次.md"
CSV_PATH = SUMMARY_DIR / "parameter_scan_round2.csv"

FEATURE_NAMES = ["x", "y", "x^2"]
INITIAL_STATE = np.array([0.25, -0.15], dtype=float)
STEPS = 80
ALPHA = 1.0
EPS = 1e-10
R_FIXED = 1


def ensure_dirs() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def fmt_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(round(float(x))))
    return f"{float(x):.6g}".replace(".", "p")


def case_dir_name(row: dict) -> str:
    return f"lam_{fmt_num(row['lam'])}_mu_{fmt_num(row['mu'])}_b_{fmt_num(row['b'])}_c_{fmt_num(row['c'])}"


def standardize_for_plot(x):
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def plot_dual_spectrum_save(forward_values, backward_values, title: str, save_path: Path) -> None:
    forward_values = np.asarray(forward_values, dtype=float).ravel()
    backward_values = np.asarray(backward_values, dtype=float).ravel()
    x = np.arange(1, max(len(forward_values), len(backward_values)) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - 0.18, np.pad(forward_values, (0, len(x) - len(forward_values)), constant_values=np.nan), width=0.36, color="tab:blue", alpha=0.45, label="$\\Sigma^{-1}$")
    ax.bar(x + 0.18, np.pad(backward_values, (0, len(x) - len(backward_values)), constant_values=np.nan), width=0.36, color="tab:orange", alpha=0.45, label="$A^T\\Sigma^{-1}A$")
    ax.plot(x - 0.18, np.pad(forward_values, (0, len(x) - len(forward_values)), constant_values=np.nan), color="tab:blue", marker="o", linewidth=1.6)
    ax.plot(x + 0.18, np.pad(backward_values, (0, len(x) - len(backward_values)), constant_values=np.nan), color="tab:orange", marker="s", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("Singular value")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_save(matrix, title: str, save_path: Path, row_labels, col_labels, cmap="vlag", center=0.0, figsize=(6, 4), annot=True):
    matrix_arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix_arr,
        ax=ax,
        cmap=cmap,
        center=center if cmap == "vlag" else None,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=annot,
        fmt=".3f",
        square=matrix_arr.shape[0] == matrix_arr.shape[1],
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_macro_observation_compare(obs, macro, title: str, save_path: Path, n_plot: int = 80) -> None:
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


def compute_common_cut_metrics(sv_forward: np.ndarray, sv_backward: np.ndarray) -> dict:
    s1, s2 = float(sv_forward[0]), float(sv_forward[1])
    k1, k2 = float(sv_backward[0]), float(sv_backward[1])
    low = max(s2, k2)
    high = min(s1, k1)
    feasible = high > low
    ratio = high / max(low, EPS)
    midpoint = 0.5 * (low + high)
    return {
        "common_cut_low": low,
        "common_cut_high": high,
        "common_cut_mid": midpoint,
        "common_cut_feasible": feasible,
        "common_cut_ratio": ratio,
        "forward_gap12": s1 / max(s2, EPS),
        "backward_gap12": k1 / max(k2, EPS),
    }


def evaluate_case_matrix_only(lam: float, mu: float, b: float, c: float) -> dict | None:
    valid = check_analytic_sigma_validity(b, c, allow_singular=False)
    if not valid["is_valid"]:
        return None

    A = make_step_system_matrix(lam, mu)
    Sigma = make_analytic_sigma_matrix(b, c)
    metrics = compute_gis_metrics(A, Sigma, alpha=ALPHA, eps=EPS)
    ce_result = compute_ce_from_gis_metrics(metrics, r_eps=R_FIXED, eps=EPS)
    w_result = build_w_from_svd(A, Sigma, r=R_FIXED, alpha=ALPHA, eps=EPS, mode="two_stage")

    sv_forward = np.asarray(metrics["sv_forward"], dtype=float)
    sv_backward = np.asarray(metrics["sv_backward"], dtype=float)
    common = compute_common_cut_metrics(sv_forward, sv_backward)
    w_abs = np.abs(np.asarray(w_result["W"], dtype=float)[0])
    y_ratio = float(w_abs[1] / max(max(w_abs[0], w_abs[2]), EPS))
    slow_ratio = float(max(lam, mu) / max(min(lam, mu), EPS))

    score = (
        3.0 * math.log10(max(common["common_cut_ratio"], 1.0))
        + 0.8 * math.log10(max(common["common_cut_high"], 1.0 + EPS))
        + 0.8 * y_ratio
        + 0.4 * slow_ratio
        + 0.3 * float(ce_result["CE"])
    )

    return {
        "lam": lam,
        "mu": mu,
        "b": b,
        "c": c,
        "forward_sv1": sv_forward[0],
        "forward_sv2": sv_forward[1],
        "forward_sv3": sv_forward[2],
        "backward_sv1": sv_backward[0],
        "backward_sv2": sv_backward[1],
        "backward_sv3": sv_backward[2],
        "common_cut_low": common["common_cut_low"],
        "common_cut_high": common["common_cut_high"],
        "common_cut_mid": common["common_cut_mid"],
        "common_cut_feasible": common["common_cut_feasible"],
        "common_cut_ratio": common["common_cut_ratio"],
        "forward_gap12": common["forward_gap12"],
        "backward_gap12": common["backward_gap12"],
        "w_abs_x": w_abs[0],
        "w_abs_y": w_abs[1],
        "w_abs_x2": w_abs[2],
        "w_y_ratio": y_ratio,
        "ce": float(ce_result["CE"]),
        "score": score,
        "A": A,
        "Sigma": Sigma,
        "metrics": metrics,
        "w_result": w_result,
    }


def simulate_case(row: dict) -> dict | None:
    sim = simulate_discrete_system(
        step_map,
        INITIAL_STATE,
        steps=STEPS,
        system_kwargs={"lam": row["lam"], "mu": row["mu"]},
        dt=1.0,
    )
    xy = sim["trajectories"][0]
    obs = observable_step(xy, mode="default")
    if not np.all(np.isfinite(obs)) or np.max(np.abs(obs)) > 1e8:
        return None
    z = apply_coarse_graining(row["w_result"]["W"], obs)
    out = dict(row)
    out["xy"] = xy
    out["obs"] = obs
    out["z"] = z
    return out


def save_case_outputs(row: dict) -> dict[str, str]:
    folder = FIG_DIR / case_dir_name(row)
    folder.mkdir(parents=True, exist_ok=True)

    dual = folder / "dual_spectrum.png"
    stage2 = folder / "stage2_weighted_matrix.png"
    wheat = folder / "W_heatmap.png"
    curve = folder / "macro_observation_compare.png"
    suffix = f"$\\lambda$={row['lam']:.3g}, $\\mu$={row['mu']:.3g}, b={row['b']:.3g}, c={row['c']:.3g}"
    plot_dual_spectrum_save(row["metrics"]["sv_forward"], row["metrics"]["sv_backward"], f"Dual spectrum ({suffix})", dual)
    weighted = row["w_result"]["basis_info"]["weighted_matrix"]
    plot_heatmap_save(weighted, f"Stage-2 weighted matrix ({suffix})", stage2, FEATURE_NAMES, [f"c{i+1}" for i in range(weighted.shape[1])], cmap="vlag", center=0.0)
    plot_heatmap_save(np.abs(row["w_result"]["W"]), f"|W| heatmap ({suffix})", wheat, ["z1"], FEATURE_NAMES, cmap="Blues", center=None, figsize=(5, 2.6))
    plot_macro_observation_compare(row["obs"], row["z"], f"Macro vs observables ({suffix})", curve)
    return {
        "dual_spectrum_path": dual.relative_to(BASE_DIR).as_posix(),
        "stage2_matrix_path": stage2.relative_to(BASE_DIR).as_posix(),
        "w_heatmap_path": wheat.relative_to(BASE_DIR).as_posix(),
        "macro_curve_path": curve.relative_to(BASE_DIR).as_posix(),
    }


def save_summary_figures(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}

    top = df.sort_values("score", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(top)), top["score"], color="tab:blue", alpha=0.8)
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels([case_dir_name(row) for _, row in top.iterrows()], rotation=60, ha="right")
    ax.set_title("Round-2 top parameter scores")
    ax.set_ylabel("Score")
    plt.tight_layout()
    path = SUMMARY_DIR / "round2_top_scores.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    out["top_scores"] = path.relative_to(BASE_DIR).as_posix()

    feasible = df[df["common_cut_feasible"]].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    scatter = ax.scatter(
        feasible["common_cut_ratio"],
        feasible["w_abs_y"],
        c=feasible["ce"],
        s=45 + 35 * np.clip(feasible["w_y_ratio"], 0, 2.0),
        cmap="coolwarm",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel("Common cutoff ratio")
    ax.set_ylabel("|W_y|")
    ax.set_title("Round-2 common cutoff vs slow-variable weight")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("CE")
    plt.tight_layout()
    path = SUMMARY_DIR / "round2_cutoff_vs_wy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    out["cutoff_vs_wy"] = path.relative_to(BASE_DIR).as_posix()

    heat = feasible.pivot_table(index="mu", columns="lam", values="common_cut_ratio", aggfunc="max").sort_index(ascending=False).sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heat, ax=ax, cmap="YlGnBu", annot=True, fmt=".2f")
    ax.set_title("Best common-cutoff ratio by $(\\lambda, \\mu)$")
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("$\\mu$")
    plt.tight_layout()
    path = SUMMARY_DIR / "round2_lambda_mu_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    out["lambda_mu_heatmap"] = path.relative_to(BASE_DIR).as_posix()
    return out


def to_markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    df = df.loc[:, columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(f"{float(value):.6f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(df: pd.DataFrame, detailed_rows: pd.DataFrame, summary_paths: dict[str, str]) -> None:
    feasible = df[df["common_cut_feasible"]].sort_values("score", ascending=False)
    w_friendly = feasible.sort_values(["w_abs_y", "common_cut_ratio", "ce"], ascending=[False, False, False])
    lines = [
        "# 参数实验第二次",
        "",
        "第二次参数实验继续扩大 `lam / mu / b / c` 的搜索范围，并使用新的评价口径：",
        "",
        "- 不再预设某个固定截断阈值，而是直接判断两个矩阵是否存在共同截断线。",
        "- 共同截断线存在的条件为：",
        "",
        "  `max(σ2(Σ^{-1}), σ2(A^T Σ^{-1} A)) < min(σ1(Σ^{-1}), σ1(A^T Σ^{-1} A))`",
        "",
        "- 当上式成立时，就说明两个矩阵都能被同一条线截成 1 维。",
        "- 在共同截断线基础上，再观察 `W` 是否偏向慢变量 `y`，以及 `CE` 是否较高。",
        "",
        f"- 共评估参数组数：`{len(df)}`。",
        f"- 存在共同截断线的参数组数：`{int(df['common_cut_feasible'].sum())}`。",
        "",
        "## 汇总图",
        "",
        f"![Top Scores](./{summary_paths['top_scores']})",
        "",
        f"![Cutoff vs Wy](./{summary_paths['cutoff_vs_wy']})",
        "",
        f"![Lambda Mu Heatmap](./{summary_paths['lambda_mu_heatmap']})",
        "",
        "## 共同截断线最强的参数",
        "",
        to_markdown_table(
            feasible,
            columns=[
                "lam",
                "mu",
                "b",
                "c",
                "common_cut_low",
                "common_cut_high",
                "common_cut_ratio",
                "w_abs_x",
                "w_abs_y",
                "w_abs_x2",
                "w_y_ratio",
                "ce",
                "score",
            ],
            max_rows=20,
        ),
        "",
        "## 更偏向慢变量 y 的共同截断参数",
        "",
        to_markdown_table(
            w_friendly,
            columns=[
                "lam",
                "mu",
                "b",
                "c",
                "common_cut_ratio",
                "w_abs_x",
                "w_abs_y",
                "w_abs_x2",
                "w_y_ratio",
                "ce",
                "score",
            ],
            max_rows=20,
        ),
        "",
        "## 已保存详细图的参数",
        "",
    ]
    for _, row in detailed_rows.iterrows():
        lines.extend(
            [
                f"### `{case_dir_name(row)}`",
                "",
                f"- 参数：`lam={row['lam']:.6g}`, `mu={row['mu']:.6g}`, `b={row['b']:.6g}`, `c={row['c']:.6g}`",
                f"- 共同截断线区间：`({row['common_cut_low']:.6f}, {row['common_cut_high']:.6f})`",
                f"- 指标：`common_cut_ratio={row['common_cut_ratio']:.4f}`, `|W_y|={row['w_abs_y']:.4f}`, `CE={row['ce']:.4f}`",
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
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()

    lam_values = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.20, 2.0, 3.0, 5.0]
    mu_values = [0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.20, 2.0, 3.0, 5.0, 8.0]
    b_values = [0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0]
    c_fraction_values = [0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]

    rows = []
    for lam in lam_values:
        for mu in mu_values:
            for b in b_values:
                for frac in c_fraction_values:
                    c = b * frac
                    if not (lam > 0 and mu > 0 and b > c >= 0):
                        continue
                    row = evaluate_case_matrix_only(lam, mu, b, c)
                    if row is not None:
                        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("第二次参数实验没有得到有效参数。")

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.drop(columns=["A", "Sigma", "metrics", "w_result"]).to_csv(CSV_PATH, index=False)

    feasible = df[df["common_cut_feasible"]].copy()
    feasible["sim_priority"] = np.where(feasible["mu"] <= 1.2, 0, 1) + np.where(feasible["lam"] <= 1.2, 0, 0.25)
    feasible = feasible.sort_values(["sim_priority", "score"], ascending=[True, False])

    detailed_candidates = []
    for _, row in feasible.head(300).iterrows():
        sim_row = simulate_case(row.to_dict())
        if sim_row is not None:
            detailed_candidates.append(sim_row)
        if len(detailed_candidates) >= 12:
            break
    if not detailed_candidates:
        raise RuntimeError("第二次参数实验没有可保存的详细参数。")

    detailed_rows = []
    for row in detailed_candidates[:12]:
        save_info = save_case_outputs(row)
        base = {k: v for k, v in row.items() if k not in {"A", "Sigma", "metrics", "w_result", "xy", "obs", "z"}}
        base.update(save_info)
        detailed_rows.append(base)

    detailed_df = pd.DataFrame(detailed_rows)
    summary_paths = save_summary_figures(df)
    write_report(df, detailed_df, summary_paths)

    print(f"Evaluated {len(df)} round-2 parameter cases.")
    print(f"Feasible common-cutoff cases: {int(df['common_cut_feasible'].sum())}")
    print(f"Wrote report to {REPORT_PATH}")
    print(f"Wrote csv to {CSV_PATH}")


if __name__ == "__main__":
    main()
