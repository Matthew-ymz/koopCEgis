from __future__ import annotations

import math
import sys
from itertools import permutations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools import (
    apply_coarse_graining,
    build_w_from_svd,
    compute_ce2_from_singular_values,
    compute_ce_from_gis_metrics,
    compute_gis_metrics,
    compute_prediction_errors,
    fit_linear_gis_from_pairs,
    prepare_time_pairs,
)
from data.data_func import (
    compute_cluster_order_parameters,
    compute_order_parameter,
    generate_kuramoto_cluster_data_sin_cos,
)


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["font.size"] = 12


BASE_CONFIG = {
    "N": 10,
    "n_clusters": 2,
    "T": 100.0,
    "dt": 0.01,
}

PIPELINE_CONFIG = {
    "observable_mode": "identity + fourier",
    "n_fourier_frequencies": 1,
    "burn_in_steps": 1000,
    "sample_stride": 10,
    "lag_steps": 1,
    "eps": 1e-10,
    "ridge": 1e-10,
    "manual_r": 2,
    "curve_window": 180,
    "horizons": (1, 3, 5),
    "micro_label_step": 4,
}

ORDER_THRESHOLDS = {
    "group_min": 0.80,
    "overall_max": 0.72,
    "gap_min": 0.10,
}

FINAL_THRESHOLDS = {
    "stage_ratio_23_min": 2.0,
    "curve_corr_min": 0.35,
    "macro_distinct_min": 0.08,
    "w_contrast_min": 0.15,
    "cluster_purity_min": 0.70,
}

SEED_TRIPLETS = [
    (41, 2600),
    (51, 2701),
    (61, 2802),
]
K_INTRA_VALUES = [4.0, 6.0, 8.0, 10.0]
K_INTER_VALUES = [0.01, 0.03, 0.05, 0.08]
NOISE_VALUES = [0.002, 0.005]

FIG_ROOT = REPO_ROOT / "exp" / "kuramoto" / "figs"
SUMMARY_DIR = FIG_ROOT / "summary_param_scan"
DOC_PATH = REPO_ROOT / "exp" / "kuramoto" / "参数实验.md"


def ensure_dirs() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def safe_to_float(value) -> float:
    return float(np.real_if_close(value))


def sparse_labels(labels: list[str] | None, step: int = 1) -> list[str] | bool:
    if labels is None:
        return False
    if step <= 1:
        return labels
    return [label if i % step == 0 else "" for i, label in enumerate(labels)]


def standardize_for_plot(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def normalize_series(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min() + 1e-12)


def format_float_tag(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def build_run_id(run_cfg: dict) -> str:
    return (
        f"ki{format_float_tag(run_cfg['K_intra'])}"
        f"_ke{format_float_tag(run_cfg['K_inter'])}"
        f"_nz{format_float_tag(run_cfg['noise'])}"
        f"_s{run_cfg['random_seed1']}_{run_cfg['random_seed2']}"
    )


def generate_runs() -> list[dict]:
    runs: list[dict] = []
    for seed1, seed2 in SEED_TRIPLETS:
        for K_intra in K_INTRA_VALUES:
            for K_inter in K_INTER_VALUES:
                for noise in NOISE_VALUES:
                    run_cfg = {
                        "K_intra": K_intra,
                        "K_inter": K_inter,
                        "noise": noise,
                        "random_seed1": seed1,
                        "random_seed2": seed2,
                        "ridge": PIPELINE_CONFIG["ridge"],
                    }
                    run_cfg["run_id"] = build_run_id(run_cfg)
                    runs.append(run_cfg)
    return runs


def observable_identity(data: np.ndarray, state_names: list[str]) -> dict:
    library = ps.IdentityLibrary()
    library.fit(data)
    lifted = np.asarray(library.transform(data), dtype=float)
    feature_names = library.get_feature_names(input_features=list(state_names))
    return {"library": library, "data": lifted, "feature_names": feature_names}


def observable_identity_fourier(data: np.ndarray, state_names: list[str], n_frequencies: int = 1) -> dict:
    library = ps.IdentityLibrary() + ps.FourierLibrary(n_frequencies=n_frequencies)
    library.fit(data)
    lifted = np.asarray(library.transform(data), dtype=float)
    feature_names = library.get_feature_names(input_features=list(state_names))
    return {"library": library, "data": lifted, "feature_names": feature_names}


def gaussian_mutual_information(A: np.ndarray, B: np.ndarray, eps: float = 1e-10) -> float:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    cov_A = np.cov(A, rowvar=False)
    cov_B = np.cov(B, rowvar=False)
    cov_AB = np.cov(np.hstack([A, B]), rowvar=False)
    sign_A, logdet_A = np.linalg.slogdet(cov_A + eps * np.eye(cov_A.shape[0]))
    sign_B, logdet_B = np.linalg.slogdet(cov_B + eps * np.eye(cov_B.shape[0]))
    sign_AB, logdet_AB = np.linalg.slogdet(cov_AB + eps * np.eye(cov_AB.shape[0]))
    if sign_A <= 0 or sign_B <= 0 or sign_AB <= 0:
        return 0.0
    return float(0.5 * (logdet_A + logdet_B - logdet_AB))


def save_matrix_heatmap(
    matrix: np.ndarray,
    title: str,
    out_path: Path,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    cmap: str = "vlag",
    center: float | None = 0.0,
    label_step: int = 1,
    figsize: tuple[float, float] = (8.0, 4.0),
) -> None:
    matrix_arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix_arr,
        ax=ax,
        cmap=cmap,
        center=center,
        square=False,
        xticklabels=sparse_labels(col_labels, label_step),
        yticklabels=sparse_labels(row_labels, label_step),
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_order_overview(
    t_fit: np.ndarray,
    raw_refs: np.ndarray,
    raw_ref_names: list[str],
    r_total: np.ndarray,
    r_groups: list[np.ndarray],
    out_path: Path,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    for idx, name in enumerate(raw_ref_names):
        ax.plot(t_fit, raw_refs[:, idx], linewidth=1.8, label=name)
    ax.set_title(f"{title_prefix}: representative raw channels")
    ax.set_xlabel("Time")
    ax.legend()

    ax = axes[1]
    ax.plot(t_fit, r_total, color="black", linewidth=2.0, label="overall")
    for idx, r_group in enumerate(r_groups):
        ax.plot(t_fit, r_group, linewidth=2.0, label=f"group {idx + 1}")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{title_prefix}: order parameters")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_three_spectra(
    value_triplet: list[np.ndarray],
    title_triplet: list[str],
    out_path: Path,
    top_k: int | None = None,
    ylabel: str = "Singular value",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for ax, values, title in zip(axes, value_triplet, title_triplet):
        vals = np.asarray(values, dtype=float).ravel()
        if top_k is not None:
            vals = vals[:top_k]
        x = np.arange(1, len(vals) + 1)
        ax.bar(x, vals, color="#4C78A8", alpha=0.85)
        ax.plot(x, vals, color="#1F3B75", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Index")
        ax.set_ylabel(ylabel)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_macro_raw_comparison(
    raw_refs: np.ndarray,
    raw_ref_names: list[str],
    z_aligned: np.ndarray,
    macro_names: list[str],
    out_path: Path,
    curve_window: int,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    raw_window = raw_refs[:curve_window]
    macro_window = z_aligned[:curve_window]

    for idx, ax in enumerate(axes):
        ax.plot(standardize_for_plot(raw_window[:, idx]), linewidth=2.0, color="#4C78A8", label=raw_ref_names[idx])
        ax.plot(
            standardize_for_plot(macro_window[:, idx]),
            "--",
            linewidth=2.0,
            color="#F58518",
            label=macro_names[idx],
        )
        ax.set_title(f"{title}: cluster {idx + 1}")
        ax.set_xlabel("Time index")
        ax.set_ylabel("Standardized value")
        ax.legend()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    fmt_df = df.copy()
    for col in fmt_df.columns:
        if pd.api.types.is_float_dtype(fmt_df[col]):
            fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.{digits}f}")
    headers = list(fmt_df.columns)
    rows = [headers]
    for _, row in fmt_df.iterrows():
        rows.append([str(row[col]) for col in headers])

    widths = [max(len(row[col_idx]) for row in rows) for col_idx in range(len(headers))]

    def format_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"
    lines = [format_row(rows[0]), sep]
    for row in rows[1:]:
        lines.append(format_row(row))
    return "\n".join(lines)


def build_run_config(run_cfg: dict) -> dict:
    return {
        **BASE_CONFIG,
        "K_intra": run_cfg["K_intra"],
        "K_inter": run_cfg["K_inter"],
        "noise": run_cfg["noise"],
        "random_seed1": run_cfg["random_seed1"],
        "random_seed2": run_cfg["random_seed2"],
    }


def select_representative_raw_channels(x_data_fit: np.ndarray, n_clusters: int, n_osc: int) -> tuple[np.ndarray, list[str], list[int]]:
    cluster_size = n_osc // n_clusters
    rep_indices = [cluster_id * cluster_size for cluster_id in range(n_clusters)]
    raw_refs = x_data_fit[:, rep_indices]
    raw_names = [f"cos_theta_{idx}" for idx in rep_indices]
    return raw_refs, raw_names, rep_indices


def compute_order_metrics(theta_hist: np.ndarray, n_clusters: int) -> dict:
    theta_eval = theta_hist[PIPELINE_CONFIG["burn_in_steps"] :]
    r_total = compute_order_parameter(theta_eval)
    r_groups = compute_cluster_order_parameters(theta_eval, n_clusters)
    group_means = [safe_to_float(np.mean(r)) for r in r_groups]
    overall_mean = safe_to_float(np.mean(r_total))
    avg_group_mean = safe_to_float(np.mean(group_means))
    order_pass = (
        min(group_means) >= ORDER_THRESHOLDS["group_min"]
        and overall_mean <= ORDER_THRESHOLDS["overall_max"]
        and (avg_group_mean - overall_mean) >= ORDER_THRESHOLDS["gap_min"]
    )
    return {
        "r_total_series": r_total,
        "r_group_series": r_groups,
        "r_total_mean": overall_mean,
        "r_group1_mean": group_means[0],
        "r_group2_mean": group_means[1],
        "r_group_avg_mean": avg_group_mean,
        "order_gap": safe_to_float(avg_group_mean - overall_mean),
        "order_pass": bool(order_pass),
    }


def find_max_gap_rank(values: np.ndarray, eps: float = 1e-10) -> int:
    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[vals > eps]
    if vals.size <= 1:
        return 1
    log_vals = np.log(vals + eps)
    gaps = log_vals[:-1] - log_vals[1:]
    return int(np.argmax(gaps) + 1)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return safe_to_float(np.dot(a, b) / denom)


def align_macro_to_raw(z_data: np.ndarray, raw_refs: np.ndarray) -> dict:
    raw_std = np.column_stack([standardize_for_plot(raw_refs[:, idx]) for idx in range(raw_refs.shape[1])])
    best = None
    for perm in permutations(range(z_data.shape[1])):
        for signs in product([-1.0, 1.0], repeat=z_data.shape[1]):
            z_perm = np.asarray(z_data[:, perm], dtype=float).copy()
            z_perm = z_perm * np.asarray(signs)[None, :]
            z_std = np.column_stack([standardize_for_plot(z_perm[:, idx]) for idx in range(z_perm.shape[1])])
            corrs = []
            mses = []
            for idx in range(z_std.shape[1]):
                corr = np.corrcoef(raw_std[:, idx], z_std[:, idx])[0, 1]
                if not np.isfinite(corr):
                    corr = 0.0
                mse = safe_to_float(np.mean((raw_std[:, idx] - z_std[:, idx]) ** 2))
                corrs.append(abs(corr))
                mses.append(mse)
            score = float(np.mean(corrs) - 0.25 * np.mean(mses))
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "perm": perm,
                    "signs": signs,
                    "z_aligned": z_perm,
                    "corrs": corrs,
                    "mses": mses,
                }
    assert best is not None
    return best


def build_cluster_masks(feature_names: list[str], n_osc: int, n_clusters: int) -> list[np.ndarray]:
    cluster_size = n_osc // n_clusters
    cluster_index_sets = []
    for cluster_id in range(n_clusters):
        start = cluster_id * cluster_size
        end = n_osc if cluster_id == n_clusters - 1 else (cluster_id + 1) * cluster_size
        cluster_index_sets.append(set(range(start, end)))

    masks = [np.zeros(len(feature_names), dtype=bool) for _ in range(n_clusters)]
    for feat_idx, feat_name in enumerate(feature_names):
        assigned = False
        for osc_idx in range(n_osc):
            token_cos = f"cos_theta_{osc_idx}"
            token_sin = f"sin_theta_{osc_idx}"
            if token_cos in feat_name or token_sin in feat_name:
                for cluster_id, idx_set in enumerate(cluster_index_sets):
                    if osc_idx in idx_set:
                        masks[cluster_id][feat_idx] = True
                        assigned = True
                        break
            if assigned:
                break
    return masks


def evaluate_w_metrics(W: np.ndarray, cluster_masks: list[np.ndarray]) -> dict:
    W = np.asarray(W, dtype=float)
    absW = np.abs(W)
    w1 = absW[0]
    w2 = absW[1]
    contrast = safe_to_float(np.mean(np.abs(w1 - w2)) / (np.mean(w1 + w2) + 1e-12))
    overlap = safe_to_float(cosine_similarity(w1, w2))
    top_k = min(12, W.shape[1])
    top1 = set(np.argsort(-w1)[:top_k].tolist())
    top2 = set(np.argsort(-w2)[:top_k].tolist())
    top_overlap_ratio = len(top1 & top2) / top_k

    best_cluster_purity = -np.inf
    best_cluster_margin = -np.inf
    for perm in permutations(range(len(cluster_masks)), len(cluster_masks)):
        purities = []
        margins = []
        for row_idx, mask_idx in enumerate(perm):
            mask = cluster_masks[mask_idx]
            in_mass = np.sum(absW[row_idx, mask])
            out_mass = np.sum(absW[row_idx, ~mask])
            total = in_mass + out_mass + 1e-12
            purities.append(safe_to_float(in_mass / total))
            margins.append(safe_to_float((in_mass - out_mass) / total))
        purity = safe_to_float(np.mean(purities))
        margin = safe_to_float(np.mean(margins))
        if purity > best_cluster_purity:
            best_cluster_purity = purity
            best_cluster_margin = margin
    return {
        "w_contrast": contrast,
        "w_overlap": overlap,
        "top_overlap_ratio": safe_to_float(top_overlap_ratio),
        "cluster_purity": best_cluster_purity,
        "cluster_margin": best_cluster_margin,
    }


def build_branch_payloads(
    A_micro: np.ndarray,
    Sigma_micro: np.ndarray,
    obs_data: np.ndarray,
    raw_refs: np.ndarray,
    feature_names: list[str],
    cluster_masks: list[np.ndarray],
) -> tuple[list[dict], list[dict]]:
    r = PIPELINE_CONFIG["manual_r"]
    eps = PIPELINE_CONFIG["eps"]

    metrics_b = compute_gis_metrics(A_micro, Sigma_micro, alpha=0.0, eps=eps)
    ce_b = compute_ce_from_gis_metrics(metrics_b, r_eps=r, alpha=0.0, eps=eps)
    w_b = build_w_from_svd(A_micro, Sigma_micro, r=r, alpha=0.0, eps=eps, mode="backward_only")
    stage_b = np.asarray(metrics_b["A_t_Sigma_inv_A"], dtype=float)
    stage_sv_b = np.linalg.svd(stage_b, compute_uv=False)
    U_b, _, Vt_b = np.linalg.svd(stage_b, full_matrices=False)
    ce2_b = compute_ce2_from_singular_values(stage_sv_b, r=r, eps=eps)

    metrics_t = compute_gis_metrics(A_micro, Sigma_micro, alpha=1.0, eps=eps)
    ce_t = compute_ce_from_gis_metrics(metrics_t, r_eps=r, alpha=1.0, eps=eps)
    w_t = build_w_from_svd(A_micro, Sigma_micro, r=r, alpha=1.0, eps=eps, mode="two_stage")
    stage_t = np.asarray(w_t["basis_info"]["weighted_matrix"], dtype=float)
    stage_sv_t = np.linalg.svd(stage_t, compute_uv=False)
    U_t, _, Vt_t = np.linalg.svd(stage_t, full_matrices=False)
    U_bar_t = np.asarray(w_t["basis_info"]["U_bar"], dtype=float)
    ce2_t = compute_ce2_from_singular_values(stage_sv_t, r=r, eps=eps)

    U_a, S_a, Vt_a = np.linalg.svd(np.asarray(A_micro, dtype=float), full_matrices=False)
    ce2_a = compute_ce2_from_singular_values(S_a, r=r, eps=eps)

    branches = [
        {
            "branch_key": "backward_only_alpha0",
            "display_name": "backward-only (A^T Sigma^{-1} A, alpha=0)",
            "preferred_bonus": 0.08,
            "primary_spectrum": np.asarray(metrics_b["sv_backward"], dtype=float),
            "primary_title": r"Primary spectrum: $A^\top \Sigma^{-1} A$",
            "stage_matrix": stage_b,
            "stage_singular_values": stage_sv_b,
            "stage_title": r"Final W-source matrix: $A^\top \Sigma^{-1} A$",
            "ce": safe_to_float(ce_b["CE"]),
            "ce2": safe_to_float(ce2_b["CE2"]),
            "W_left": np.asarray(w_b["W"], dtype=float),
            "W_right_current": np.asarray(Vt_b[:r, :], dtype=float),
            "W_right_direct": np.asarray(Vt_b[:, :r].T, dtype=float),
        },
        {
            "branch_key": "two_stage_alpha1",
            "display_name": "two-stage (Sigma^{-1} + A^T Sigma^{-1} A, alpha=1)",
            "preferred_bonus": 0.03,
            "primary_spectrum": np.asarray(w_t["sv_info"]["sv_all"], dtype=float),
            "primary_title": "Primary spectrum: jointly sorted values",
            "stage_matrix": stage_t,
            "stage_singular_values": stage_sv_t,
            "stage_title": "Final W-source matrix: stage-2 weighted matrix",
            "ce": safe_to_float(ce_t["CE"]),
            "ce2": safe_to_float(ce2_t["CE2"]),
            "W_left": np.asarray(w_t["W"], dtype=float),
            "W_right_current": np.asarray(Vt_t[:r, :] @ U_bar_t.T, dtype=float),
            "W_right_direct": np.asarray(Vt_t[:, :r].T, dtype=float),
        },
        {
            "branch_key": "A_only_alpha1",
            "display_name": "A-only (A, alpha=1)",
            "preferred_bonus": 0.00,
            "primary_spectrum": np.asarray(S_a, dtype=float),
            "primary_title": "Primary spectrum: A",
            "stage_matrix": np.asarray(A_micro, dtype=float),
            "stage_singular_values": np.asarray(S_a, dtype=float),
            "stage_title": "Final W-source matrix: A",
            "ce": safe_to_float(ce2_a["CE2"]),
            "ce2": safe_to_float(ce2_a["CE2"]),
            "W_left": np.asarray(U_a[:, :r].T, dtype=float),
            "W_right_current": np.asarray(Vt_a[:r, :], dtype=float),
            "W_right_direct": np.asarray(Vt_a[:, :r].T, dtype=float),
        },
    ]

    macro_names = [f"z_{idx + 1}" for idx in range(r)]
    candidates: list[dict] = []
    for branch in branches:
        for method_key, method_label in [
            ("W_left", "left"),
            ("W_right_current", "right_current"),
            ("W_right_direct", "right_direct"),
        ]:
            W = np.asarray(branch[method_key], dtype=float)
            z = apply_coarse_graining(W, obs_data)
            aligned = align_macro_to_raw(z, raw_refs)
            w_metrics = evaluate_w_metrics(W, cluster_masks)
            z_std = np.column_stack([standardize_for_plot(aligned["z_aligned"][:, idx]) for idx in range(r)])
            macro_corr = np.corrcoef(z_std[:, 0], z_std[:, 1])[0, 1]
            if not np.isfinite(macro_corr):
                macro_corr = 0.0
            candidate = {
                "branch_key": branch["branch_key"],
                "branch_display": branch["display_name"],
                "method_key": method_key,
                "method_label": method_label,
                "W": W,
                "z_aligned": aligned["z_aligned"],
                "stage_matrix": branch["stage_matrix"],
                "stage_singular_values": branch["stage_singular_values"],
                "primary_spectrum": branch["primary_spectrum"],
                "ce": branch["ce"],
                "ce2": branch["ce2"],
                "primary_title": branch["primary_title"],
                "stage_title": branch["stage_title"],
                "curve_corr_mean": safe_to_float(np.mean(aligned["corrs"])),
                "curve_mse_mean": safe_to_float(np.mean(aligned["mses"])),
                "macro_distinct": safe_to_float(1.0 - abs(macro_corr)),
                "stage_ratio_23": safe_to_float(
                    branch["stage_singular_values"][1] / (branch["stage_singular_values"][2] + eps)
                )
                if len(branch["stage_singular_values"]) >= 3
                else float("inf"),
                "max_gap_rank": int(find_max_gap_rank(branch["stage_singular_values"], eps=eps)),
                **w_metrics,
                "preferred_bonus": branch["preferred_bonus"],
            }
            candidates.append(candidate)
    return branches, candidates


def score_candidates(candidates: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{k: v for k, v in item.items() if k not in {"W", "z_aligned", "stage_matrix", "stage_singular_values", "primary_spectrum", "primary_title", "stage_title"}} for item in candidates])
    score = pd.Series(0.0, index=df.index)
    score += 0.30 * normalize_series(df["stage_ratio_23"].replace([np.inf, -np.inf], np.nan).fillna(df["stage_ratio_23"].replace([np.inf, -np.inf], np.nan).max()))
    score += 0.20 * normalize_series(df["curve_corr_mean"])
    score += 0.15 * (1 - normalize_series(df["curve_mse_mean"]))
    score += 0.15 * normalize_series(df["w_contrast"])
    score += 0.10 * normalize_series(df["cluster_purity"])
    score += 0.05 * normalize_series(df["cluster_margin"])
    score += 0.03 * (1 - normalize_series(df["top_overlap_ratio"]))
    score += 0.02 * normalize_series(df["macro_distinct"])
    score += df["preferred_bonus"]
    df["score"] = score
    df["desired_pass"] = (
        (df["stage_ratio_23"] >= FINAL_THRESHOLDS["stage_ratio_23_min"])
        & (df["curve_corr_mean"] >= FINAL_THRESHOLDS["curve_corr_min"])
        & (df["macro_distinct"] >= FINAL_THRESHOLDS["macro_distinct_min"])
        & (df["w_contrast"] >= FINAL_THRESHOLDS["w_contrast_min"])
        & (df["cluster_purity"] >= FINAL_THRESHOLDS["cluster_purity_min"])
    )
    return df.sort_values(["desired_pass", "score"], ascending=[False, False]).reset_index(drop=True)


def save_branch_overview(branches: list[dict], run_dir: Path) -> None:
    save_three_spectra(
        [branch["primary_spectrum"] for branch in branches],
        [branch["display_name"] for branch in branches],
        run_dir / "primary_spectra.png",
        top_k=20,
        ylabel="Primary spectrum value",
    )
    save_three_spectra(
        [branch["stage_singular_values"] for branch in branches],
        [branch["display_name"] for branch in branches],
        run_dir / "stage_spectra.png",
        top_k=20,
        ylabel="Final W-source singular value",
    )


def save_selected_candidate_artifacts(
    selected: dict,
    feature_names: list[str],
    raw_refs: np.ndarray,
    raw_ref_names: list[str],
    run_dir: Path,
) -> None:
    macro_names = [f"z_{idx + 1}" for idx in range(PIPELINE_CONFIG["manual_r"])]
    save_matrix_heatmap(
        np.abs(selected["stage_matrix"]),
        title=f"{selected['branch_display']} | {selected['method_label']} | final W-source matrix",
        out_path=run_dir / "selected_stage_matrix.png",
        row_labels=feature_names,
        col_labels=feature_names if selected["stage_matrix"].shape[1] == len(feature_names) else [f"s{i + 1}" for i in range(selected["stage_matrix"].shape[1])],
        cmap="Blues",
        center=None,
        label_step=PIPELINE_CONFIG["micro_label_step"],
        figsize=(8.6, 5.0),
    )
    save_matrix_heatmap(
        np.abs(selected["W"]),
        title=f"{selected['branch_display']} | {selected['method_label']} | |W|",
        out_path=run_dir / "selected_W_heatmap.png",
        row_labels=macro_names,
        col_labels=feature_names,
        cmap="Blues",
        center=None,
        label_step=PIPELINE_CONFIG["micro_label_step"],
        figsize=(8.4, 3.8),
    )
    save_macro_raw_comparison(
        raw_refs=raw_refs,
        raw_ref_names=raw_ref_names,
        z_aligned=selected["z_aligned"],
        macro_names=macro_names,
        out_path=run_dir / "selected_macro_raw_comparison.png",
        curve_window=PIPELINE_CONFIG["curve_window"],
        title=f"{selected['branch_display']} | {selected['method_label']}",
    )


def run_single_experiment(run_cfg: dict) -> dict:
    print(f"Running {run_cfg['run_id']} ...", flush=True)
    kuramoto_cfg = build_run_config(run_cfg)
    run_dir = FIG_ROOT / run_cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    x_clean, theta_hist, t_data_raw, _ = generate_kuramoto_cluster_data_sin_cos(**kuramoto_cfg)
    x_data = np.asarray(x_clean, dtype=float)

    x_data_fit = x_data[PIPELINE_CONFIG["burn_in_steps"] :: PIPELINE_CONFIG["sample_stride"]].copy()
    x_clean_fit = x_clean[PIPELINE_CONFIG["burn_in_steps"] :: PIPELINE_CONFIG["sample_stride"]].copy()
    t_fit = t_data_raw[PIPELINE_CONFIG["burn_in_steps"] :: PIPELINE_CONFIG["sample_stride"]].copy()
    raw_refs, raw_ref_names, rep_indices = select_representative_raw_channels(
        x_data_fit=x_clean_fit,
        n_clusters=kuramoto_cfg["n_clusters"],
        n_osc=kuramoto_cfg["N"],
    )

    order_metrics = compute_order_metrics(theta_hist, kuramoto_cfg["n_clusters"])
    save_order_overview(
        t_fit=t_fit,
        raw_refs=raw_refs,
        raw_ref_names=raw_ref_names,
        r_total=order_metrics["r_total_series"][:: PIPELINE_CONFIG["sample_stride"]],
        r_groups=[r_group[:: PIPELINE_CONFIG["sample_stride"]] for r_group in order_metrics["r_group_series"]],
        out_path=run_dir / "data_order_overview.png",
        title_prefix=run_cfg["run_id"],
    )

    state_names_raw = [f"cos_theta_{i}" for i in range(kuramoto_cfg["N"])] + [f"sin_theta_{i}" for i in range(kuramoto_cfg["N"])]
    dt_fit = kuramoto_cfg["dt"] * PIPELINE_CONFIG["sample_stride"]
    tau_phys = PIPELINE_CONFIG["lag_steps"] * dt_fit
    obs_result = observable_identity_fourier(
        x_data_fit,
        state_names_raw,
        n_frequencies=PIPELINE_CONFIG["n_fourier_frequencies"],
    )
    obs_data = obs_result["data"]
    feature_names = obs_result["feature_names"]
    cluster_masks = build_cluster_masks(feature_names, kuramoto_cfg["N"], kuramoto_cfg["n_clusters"])

    X_now, X_next = prepare_time_pairs(obs_data, tau=PIPELINE_CONFIG["lag_steps"], burn_in=0, stride=1)
    fit = fit_linear_gis_from_pairs(
        X_now,
        X_next,
        fit_intercept=False,
        ridge=run_cfg["ridge"],
        regularization=PIPELINE_CONFIG["eps"],
    )
    A_micro = fit["A"]
    Sigma_micro = fit["Sigma"]
    errors_micro = compute_prediction_errors(
        A_micro,
        obs_data,
        tau=PIPELINE_CONFIG["lag_steps"],
        horizons=PIPELINE_CONFIG["horizons"],
    )

    branch_rows_path = run_dir / "candidate_scores.csv"
    if not order_metrics["order_pass"]:
        fail_row = {
            "run_id": run_cfg["run_id"],
            "K_intra": run_cfg["K_intra"],
            "K_inter": run_cfg["K_inter"],
            "noise": run_cfg["noise"],
            "random_seed1": run_cfg["random_seed1"],
            "random_seed2": run_cfg["random_seed2"],
            "tau_phys": tau_phys,
            "order_pass": False,
            "desired_pass": False,
            "r_total_mean": order_metrics["r_total_mean"],
            "r_group1_mean": order_metrics["r_group1_mean"],
            "r_group2_mean": order_metrics["r_group2_mean"],
            "order_gap": order_metrics["order_gap"],
            "best_branch": "not_evaluated",
            "best_method": "not_evaluated",
            "best_score": np.nan,
            "selected_branch_preferred": False,
            "best_cluster_purity": np.nan,
            "figure_dir": run_dir.relative_to(REPO_ROOT).as_posix(),
        }
        pd.DataFrame(
            [{
                "branch_display": "screening_failed",
                "method_label": "screening_failed",
                "score": np.nan,
                "desired_pass": False,
                "note": "Failed hard order-parameter screening; later steps skipped.",
            }]
        ).to_csv(branch_rows_path, index=False, encoding="utf-8-sig")
        return fail_row

    branches, candidates = build_branch_payloads(
        A_micro=A_micro,
        Sigma_micro=Sigma_micro,
        obs_data=obs_data,
        raw_refs=raw_refs,
        feature_names=feature_names,
        cluster_masks=cluster_masks,
    )
    save_branch_overview(branches, run_dir)

    candidate_df = score_candidates(candidates)
    candidate_df.to_csv(branch_rows_path, index=False, encoding="utf-8-sig")
    best_row = candidate_df.iloc[0].to_dict()
    best_candidate = next(
        item for item in candidates
        if item["branch_key"] == best_row["branch_key"] and item["method_label"] == best_row["method_label"]
    )
    save_selected_candidate_artifacts(
        selected=best_candidate,
        feature_names=feature_names,
        raw_refs=raw_refs,
        raw_ref_names=raw_ref_names,
        run_dir=run_dir,
    )

    metrics_micro_alpha0 = compute_gis_metrics(A_micro, Sigma_micro, alpha=0.0, eps=PIPELINE_CONFIG["eps"])
    metrics_micro_alpha1 = compute_gis_metrics(A_micro, Sigma_micro, alpha=1.0, eps=PIPELINE_CONFIG["eps"])
    ce_alpha0 = compute_ce_from_gis_metrics(metrics_micro_alpha0, r_eps=PIPELINE_CONFIG["manual_r"], alpha=0.0, eps=PIPELINE_CONFIG["eps"])
    ce_alpha1 = compute_ce_from_gis_metrics(metrics_micro_alpha1, r_eps=PIPELINE_CONFIG["manual_r"], alpha=1.0, eps=PIPELINE_CONFIG["eps"])
    mi_denominator = gaussian_mutual_information(X_now, X_next, eps=PIPELINE_CONFIG["eps"])
    z_best_now, _ = prepare_time_pairs(best_candidate["z_aligned"], tau=PIPELINE_CONFIG["lag_steps"], burn_in=0, stride=1)
    mi_ratio_best = gaussian_mutual_information(z_best_now, X_next, eps=PIPELINE_CONFIG["eps"]) / max(mi_denominator, 1e-12)

    return {
        "run_id": run_cfg["run_id"],
        "K_intra": run_cfg["K_intra"],
        "K_inter": run_cfg["K_inter"],
        "noise": run_cfg["noise"],
        "random_seed1": run_cfg["random_seed1"],
        "random_seed2": run_cfg["random_seed2"],
        "tau_phys": tau_phys,
        "order_pass": True,
        "desired_pass": bool(best_row["desired_pass"]),
        "r_total_mean": order_metrics["r_total_mean"],
        "r_group1_mean": order_metrics["r_group1_mean"],
        "r_group2_mean": order_metrics["r_group2_mean"],
        "order_gap": order_metrics["order_gap"],
        "pred_err_h1": safe_to_float(errors_micro[1]["mean_error"]),
        "pred_err_h3": safe_to_float(errors_micro[3]["mean_error"]),
        "pred_err_h5": safe_to_float(errors_micro[5]["mean_error"]),
        "ce_alpha0": safe_to_float(ce_alpha0["CE"]),
        "ce_alpha1": safe_to_float(ce_alpha1["CE"]),
        "best_branch": best_row["branch_display"],
        "best_method": best_row["method_label"],
        "best_score": safe_to_float(best_row["score"]),
        "best_ce": safe_to_float(best_row["ce"]),
        "best_ce2": safe_to_float(best_row["ce2"]),
        "best_stage_ratio_23": safe_to_float(best_row["stage_ratio_23"]),
        "best_gap_rank": int(best_row["max_gap_rank"]),
        "best_curve_corr": safe_to_float(best_row["curve_corr_mean"]),
        "best_curve_mse": safe_to_float(best_row["curve_mse_mean"]),
        "best_w_contrast": safe_to_float(best_row["w_contrast"]),
        "best_cluster_purity": safe_to_float(best_row["cluster_purity"]),
        "best_top_overlap_ratio": safe_to_float(best_row["top_overlap_ratio"]),
        "best_macro_distinct": safe_to_float(best_row["macro_distinct"]),
        "mi_ratio_best": safe_to_float(mi_ratio_best),
        "selected_branch_preferred": bool(best_row["branch_key"] == "backward_only_alpha0"),
        "figure_dir": run_dir.relative_to(REPO_ROOT).as_posix(),
    }


def build_summary_visual(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    scatter = ax.scatter(df["r_total_mean"], df["order_gap"], c=df["best_score"].fillna(0), s=120, cmap="viridis")
    ax.set_title("Order screening map")
    ax.set_xlabel("Mean overall order parameter")
    ax.set_ylabel("Mean(group) - overall")
    for _, row in df.head(12).iterrows():
        ax.annotate(row["run_id"], (row["r_total_mean"], row["order_gap"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Best score")

    ax = axes[0, 1]
    branch_counts = df.loc[df["order_pass"], "best_branch"].value_counts()
    if branch_counts.empty:
        ax.text(0.5, 0.5, "No order-pass runs", ha="center", va="center")
        ax.axis("off")
    else:
        ax.bar(branch_counts.index, branch_counts.values, color=["#4C78A8", "#72B7B2", "#F58518"][: len(branch_counts)])
        ax.set_title("Best branch among order-pass runs")
        ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 0]
    score_df = df.loc[df["order_pass"], ["run_id", "best_stage_ratio_23", "best_curve_corr", "best_w_contrast", "best_macro_distinct", "best_ce"]].copy()
    if score_df.empty:
        ax.text(0.5, 0.5, "No order-pass runs", ha="center", va="center")
        ax.axis("off")
    else:
        score_df = score_df.sort_values("best_stage_ratio_23", ascending=False).head(15).set_index("run_id")
        sns.heatmap(score_df, cmap="mako", annot=True, fmt=".3f", ax=ax, cbar=False)
        ax.set_title("Top runs: branch-quality metrics")

    ax = axes[1, 1]
    subset = df.loc[df["order_pass"]].copy()
    if subset.empty:
        ax.text(0.5, 0.5, "No order-pass runs", ha="center", va="center")
        ax.axis("off")
    else:
        scatter2 = ax.scatter(subset["best_curve_corr"], subset["best_w_contrast"], c=subset["best_stage_ratio_23"], s=120, cmap="plasma")
        ax.set_title("Curve match vs W contrast")
        ax.set_xlabel("Best curve correlation")
        ax.set_ylabel("Best W contrast")
        cbar2 = fig.colorbar(scatter2, ax=ax)
        cbar2.set_label("Stage ratio s2 / s3")

    plt.tight_layout()
    fig.savefig(SUMMARY_DIR / "summary_overview.png", bbox_inches="tight")
    plt.close(fig)


def choose_recommendation(df: pd.DataFrame) -> str:
    subset = df.loc[df["desired_pass"]].copy()
    if subset.empty:
        subset = df.loc[df["order_pass"]].copy()
    if subset.empty:
        return "none"
    score = pd.Series(0.0, index=subset.index)
    score += 0.30 * normalize_series(subset["best_stage_ratio_23"])
    score += 0.20 * normalize_series(subset["best_curve_corr"])
    score += 0.15 * normalize_series(subset["best_w_contrast"])
    score += 0.15 * normalize_series(subset["best_cluster_purity"])
    score += 0.10 * normalize_series(subset["best_macro_distinct"])
    score += 0.10 * normalize_series(subset["best_ce"])
    score += 0.10 * subset["selected_branch_preferred"].astype(float)
    return str(subset.loc[score.idxmax(), "run_id"])


def summarize_patterns(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    order_pass_df = df.loc[df["order_pass"]].copy()
    desired_df = df.loc[df["desired_pass"]].copy()
    lines.append(f"- 共扫描 `{len(df)}` 组参数，其中通过序参量硬筛选的有 `{len(order_pass_df)}` 组。")
    lines.append(f"- 同时满足谱、曲线和 `W` 分离度要求的共有 `{len(desired_df)}` 组。")
    if not order_pass_df.empty:
        preferred_ratio = safe_to_float(order_pass_df["selected_branch_preferred"].mean())
        lines.append(f"- 在通过硬筛选的样本里，最终最优方法落在第一个分支 `backward-only` 的比例约为 `{preferred_ratio:.2%}`。")
        grouped = order_pass_df.groupby(["K_intra", "K_inter"])["desired_pass"].mean().reset_index()
        best_pair = grouped.sort_values("desired_pass", ascending=False).iloc[0]
        lines.append(
            f"- 从扫描结果看，较稳定的区域集中在 `K_intra={best_pair['K_intra']:.2f}`、`K_inter={best_pair['K_inter']:.2f}` 附近，说明“团内强、团间弱”仍然是最重要的前提。"
        )
        noise_best = order_pass_df.groupby("noise")["desired_pass"].mean().sort_values(ascending=False)
        best_noise = float(noise_best.index[0])
        lines.append(f"- 本轮里更容易得到清晰二维宏观结构的噪音强度集中在 `noise={best_noise:.3f}` 附近。")
    return lines


def write_markdown(df: pd.DataFrame, recommended_run: str) -> None:
    metric_cols = [
        "run_id",
        "K_intra",
        "K_inter",
        "noise",
        "random_seed1",
        "random_seed2",
        "r_total_mean",
        "r_group1_mean",
        "r_group2_mean",
        "order_gap",
        "order_pass",
        "desired_pass",
    ]
    best_cols = [
        "run_id",
        "best_branch",
        "best_method",
        "best_score",
        "best_ce",
        "best_ce2",
        "best_stage_ratio_23",
        "best_gap_rank",
        "best_curve_corr",
        "best_curve_mse",
        "best_w_contrast",
        "best_cluster_purity",
        "best_macro_distinct",
        "selected_branch_preferred",
    ]
    fig_cols = ["run_id", "figure_dir"]

    order_pass_df = df.loc[df["order_pass"]].copy()
    desired_df = df.loc[df["desired_pass"]].copy()

    lines: list[str] = []
    lines.append("# 参数实验")
    lines.append("")
    lines.append("## 实验目标")
    lines.append("")
    lines.append("这一轮批量实验围绕两团 Kuramoto 系统做参数扫描，优先寻找满足以下条件的样本：")
    lines.append("- 两个团各自序参量较高，但整体序参量明显更低。")
    lines.append("- 二次 SVD 谱前两维明显突出。")
    lines.append("- 宏观曲线与代表性原始通道差异较小。")
    lines.append("- `W` 的两列尽量按团分组，而不是按 cos/sin 分组。")
    lines.append("")
    lines.append("## 扫描设置")
    lines.append("")
    lines.append(f"- `K_intra` 扫描集合：`{K_INTRA_VALUES}`")
    lines.append(f"- `K_inter` 扫描集合：`{K_INTER_VALUES}`")
    lines.append(f"- `noise` 扫描集合：`{NOISE_VALUES}`")
    lines.append(f"- 随机种子组合：`{SEED_TRIPLETS}`")
    lines.append(f"- 观测函数：`{PIPELINE_CONFIG['observable_mode']}`")
    lines.append(f"- 固定宏观维度：`r = {PIPELINE_CONFIG['manual_r']}`")
    lines.append("- 本轮不再额外叠加任何人为观测噪声，只使用 Kuramoto 动力学自身的内在噪声。")
    lines.append("")
    lines.append("## 序参量硬筛选条件")
    lines.append("")
    lines.append(f"- 两团序参量均值至少为 `{ORDER_THRESHOLDS['group_min']:.2f}`")
    lines.append(f"- 整体序参量均值不高于 `{ORDER_THRESHOLDS['overall_max']:.2f}`")
    lines.append(f"- 团均值与整体均值之差至少为 `{ORDER_THRESHOLDS['gap_min']:.2f}`")
    lines.append("")
    lines.append("## 本轮观察到的规律")
    lines.append("")
    lines.extend(summarize_patterns(df))
    lines.append("")
    lines.append("## 推荐优先查看的参数组")
    lines.append("")
    lines.append(f"推荐先看：`{recommended_run}`")
    lines.append("")
    if not desired_df.empty:
        lines.append("## 满足全部目标的优选结果")
        lines.append("")
        lines.append(dataframe_to_markdown(desired_df[best_cols + ["figure_dir"]].head(12), digits=4))
        lines.append("")
    if not order_pass_df.empty:
        lines.append("## 通过序参量筛选的结果")
        lines.append("")
        lines.append(dataframe_to_markdown(order_pass_df[metric_cols + ["best_branch", "best_method", "best_score"]].head(20), digits=4))
        lines.append("")
    lines.append("## 全部参数概览")
    lines.append("")
    lines.append(dataframe_to_markdown(df[metric_cols], digits=4))
    lines.append("")
    lines.append("## 最优分支与指标")
    lines.append("")
    lines.append(dataframe_to_markdown(df[best_cols], digits=4))
    lines.append("")
    lines.append("## 图像目录")
    lines.append("")
    lines.append(dataframe_to_markdown(df[fig_cols], digits=4))
    lines.append("")
    lines.append("## 每组参数文件夹内容")
    lines.append("")
    lines.append("- `data_order_overview.png`：代表性原始通道与序参量组合图。")
    lines.append("- `primary_spectra.png`：三个分支的第一层谱对比。")
    lines.append("- `stage_spectra.png`：三个分支的最终 W 来源矩阵奇异值谱。")
    lines.append("- `selected_stage_matrix.png`：当前参数下被选中方法的最终 W 来源矩阵。")
    lines.append("- `selected_W_heatmap.png`：被选中方法的粗粒化矩阵热力图。")
    lines.append("- `selected_macro_raw_comparison.png`：被选中方法的宏观与代表性原始数据对比图。")
    lines.append("- `candidate_scores.csv`：该参数组下九种候选方法的打分与筛选结果。")
    lines.append("")
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    runs = generate_runs()
    rows = [run_single_experiment(run_cfg) for run_cfg in runs]
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["desired_pass", "order_pass", "best_score", "order_gap"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    df.to_csv(SUMMARY_DIR / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    build_summary_visual(df)
    recommended_run = choose_recommendation(df)
    write_markdown(df, recommended_run)
    print(f"Wrote summary markdown to {DOC_PATH}")
    print(f"Recommended run: {recommended_run}")


if __name__ == "__main__":
    main()
