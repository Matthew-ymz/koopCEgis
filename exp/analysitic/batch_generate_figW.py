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

from tools import make_step_system_matrix, observable_step, step_map


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "DejaVu Serif"

BASE_DIR = REPO_ROOT / "exp" / "analysitic"
OUT_DIR = BASE_DIR / "figW"
CSV_PATH = OUT_DIR / "batch_summary.csv"
MD_PATH = OUT_DIR / "result_analysis.md"

STATE_NAMES = ["x", "y"]
OBS_NAMES = ["x", "y", "x^2"]
HEATMAP_CMAP = "vlag"
POSITIVE_CMAP = "Blues"

PARAMETER_PAIRS = [
    (20.0, 380.0),
    (20.0, 200.0),
    (20.0, 0.3),
    (20.0, 2.0),
    (5.0, 50.0),
    (0.5, 5.0),
    (0.5, 0.1),
    (0.1, 0.0001),
    (0.6, 0.45),
    (0.1, 0.05),
    (0.1, 0.9),
    (1.0, 0.3),
    (1.0, 2.6),
    (3.0, 1.0),
    (0.3, 1.0),
]

REP_INITIAL_STATE = np.array([0.80, -0.35], dtype=float)
PHASE_SAMPLES = 18
RNG_SEED = 7
PLOT_CLIP = 1e8


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def choose_heatmap_cmap(matrix: np.ndarray) -> str:
    matrix = np.asarray(matrix, dtype=float)
    if np.any(matrix < 0) and np.any(matrix > 0):
        return HEATMAP_CMAP
    return POSITIVE_CMAP


def canonicalize_svd(U: np.ndarray, Vt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U = np.array(U, dtype=float, copy=True)
    Vt = np.array(Vt, dtype=float, copy=True)
    for k in range(U.shape[1]):
        pivot = int(np.argmax(np.abs(U[:, k])))
        sign = 1.0 if U[pivot, k] >= 0 else -1.0
        U[:, k] *= sign
        Vt[k, :] *= sign
    return U, Vt


def as_real_if_close(values: np.ndarray, tol: float = 1e-10, name: str = "array") -> np.ndarray:
    values = np.asarray(values)
    max_imag = np.max(np.abs(np.imag(values))) if np.iscomplexobj(values) else 0.0
    if max_imag > tol:
        raise ValueError(f"{name} has a non-negligible imaginary part: {max_imag:.3e}")
    return np.real(values)


def canonicalize_eigenvectors(Q: np.ndarray) -> np.ndarray:
    Q = np.array(Q, dtype=float, copy=True)
    for k in range(Q.shape[1]):
        pivot = int(np.argmax(np.abs(Q[:, k])))
        sign = 1.0 if Q[pivot, k] >= 0 else -1.0
        Q[:, k] *= sign
    return Q


def sorted_eigendecomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals = as_real_if_close(eigvals, name="eigenvalues")
    eigvecs = as_real_if_close(eigvecs, name="eigenvectors")
    order = np.lexsort((-eigvals, -np.abs(eigvals)))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvecs = canonicalize_eigenvectors(eigvecs)
    return eigvals, eigvecs


def safe_param_str(x: float) -> str:
    return f"{x:g}"


def plot_name(lam: float, mu: float) -> str:
    return f"lam_{safe_param_str(lam)}__mu_{safe_param_str(mu)}.png"


def auto_axis_limits(points: np.ndarray, pad_ratio: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.asarray(points, dtype=float)
    finite_mask = np.all(np.isfinite(points), axis=1)
    points = points[finite_mask]
    if len(points) == 0:
        return (-1.0, 1.0), (-1.0, 1.0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    pad = spans * pad_ratio
    return (mins[0] - pad[0], maxs[0] + pad[0]), (mins[1] - pad[1], maxs[1] + pad[1])


def maybe_use_symlog(ax: plt.Axes, values: np.ndarray, which: str = "y") -> None:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    vmax = np.max(np.abs(finite))
    nonzero = np.abs(finite[np.abs(finite) > 0])
    if nonzero.size == 0:
        return
    vmin = np.min(nonzero)
    if vmax / max(vmin, 1e-12) >= 1e4:
        if which == "x":
            ax.set_xscale("symlog", linthresh=max(vmin, 1e-3))
        elif which == "y":
            ax.set_yscale("symlog", linthresh=max(vmin, 1e-3))
        else:
            ax.set_xscale("symlog", linthresh=max(vmin, 1e-3))
            ax.set_yscale("symlog", linthresh=max(vmin, 1e-3))


def draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    annot_fmt: str = ".3f",
    center: float | None = None,
) -> None:
    matrix = np.asarray(matrix, dtype=float)
    cmap = choose_heatmap_cmap(matrix)
    heat_center = center if cmap == HEATMAP_CMAP else None
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=annot_fmt,
        cbar=True,
        square=matrix.shape[0] == matrix.shape[1],
        xticklabels=col_labels,
        yticklabels=row_labels,
        center=heat_center,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title)


def choose_steps(lam: float, mu: float) -> tuple[int, int]:
    scale = max(abs(lam), abs(mu))
    if scale >= 1e4:
        return 3, 3
    if scale >= 100:
        return 4, 4
    if scale >= 10:
        return 6, 5
    if scale >= 1:
        return 18, 10
    return 40, 22


def simulate_single_trajectory(initial_state: np.ndarray, lam: float, mu: float, steps: int) -> np.ndarray:
    current = np.asarray(initial_state, dtype=float).copy()
    states = [current.copy()]
    for _ in range(int(steps)):
        current = np.asarray(step_map(current[0], current[1], lam=lam, mu=mu), dtype=float)
        states.append(current.copy())
        if not np.all(np.isfinite(current)):
            break
        if np.max(np.abs(current)) > PLOT_CLIP:
            break
    out = np.asarray(states, dtype=float)
    return np.clip(out, -PLOT_CLIP, PLOT_CLIP)


def simulate_phase_bundle(lam: float, mu: float, steps: int, n_samples: int = PHASE_SAMPLES, seed: int = RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    initial_states = np.column_stack(
        [
            rng.uniform(-0.95, 0.95, n_samples),
            rng.uniform(-0.95, 0.95, n_samples),
        ]
    )
    trajectories = [simulate_single_trajectory(state, lam, mu, steps) for state in initial_states]
    return np.asarray(trajectories, dtype=object)


def to_numeric_phase_points(phase_bundle: np.ndarray) -> np.ndarray:
    points: list[np.ndarray] = []
    for traj in phase_bundle:
        arr = np.asarray(traj, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            points.append(arr)
    if not points:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(points)


def save_full_flow_figure(lam: float, mu: float) -> dict[str, float]:
    main_steps, phase_steps = choose_steps(lam, mu)
    xy = simulate_single_trajectory(REP_INITIAL_STATE, lam, mu, main_steps)
    obs = observable_step(xy, mode="default")
    A = make_step_system_matrix(lam, mu)
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    U, Vt = canonicalize_svd(U, Vt)
    uv_gap_fro = float(np.linalg.norm(U - Vt.T, ord="fro"))
    eigvals, eigvecs = sorted_eigendecomposition(A)
    eigvec_orth_gap = eigvecs.T @ eigvecs - np.eye(eigvecs.shape[1])
    eigvec_nonorth_fro = float(np.linalg.norm(eigvec_orth_gap, ord="fro"))

    phase_bundle = simulate_phase_bundle(lam, mu, phase_steps)
    phase_points = to_numeric_phase_points(phase_bundle)
    xlim, ylim = auto_axis_limits(phase_points)
    eig_labels = [r"$q_1$", r"$q_2$", r"$q_3$"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13.2))

    draw_heatmap(axes[0, 0], A, OBS_NAMES, OBS_NAMES, "$A_o$", center=0.0)

    for traj in phase_bundle:
        traj = np.asarray(traj, dtype=float)
        if traj.ndim != 2 or traj.shape[1] != 2:
            continue
        axes[0, 1].plot(traj[:, 0], traj[:, 1], color="0.75", linewidth=0.9, alpha=0.9)
    axes[0, 1].plot(xy[:, 0], xy[:, 1], color="tab:orange", linewidth=2.0, label="representative trajectory")
    axes[0, 1].scatter(xy[0, 0], xy[0, 1], color="tab:green", s=28, label="start")
    axes[0, 1].scatter(xy[-1, 0], xy[-1, 1], color="tab:red", s=28, label="end")
    axes[0, 1].set_title("Phase portrait")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].set_xlim(xlim)
    axes[0, 1].set_ylim(ylim)
    maybe_use_symlog(axes[0, 1], phase_points, which="both")
    axes[0, 1].legend(loc="best", fontsize=8)

    obs_x = np.arange(len(obs))
    for idx, name in enumerate(OBS_NAMES):
        axes[0, 2].plot(obs_x, obs[:, idx], linewidth=2.0, label=name)
    axes[0, 2].set_title("Lifted observable trajectories")
    axes[0, 2].set_xlabel("step")
    axes[0, 2].set_ylabel("value")
    maybe_use_symlog(axes[0, 2], obs, which="y")
    axes[0, 2].legend(loc="best", fontsize=8)

    draw_heatmap(axes[1, 0], U, OBS_NAMES, [r"$u_1$", r"$u_2$", r"$u_3$"], "Left singular-vector matrix $U$", center=0.0)

    axes[1, 1].bar(np.arange(1, len(s) + 1), s, color="tab:blue", alpha=0.85)
    axes[1, 1].plot(np.arange(1, len(s) + 1), s, color="0.35", linewidth=1.1)
    axes[1, 1].set_title("Singular value spectrum")
    axes[1, 1].set_xlabel("index")
    axes[1, 1].set_ylabel("singular value")
    maybe_use_symlog(axes[1, 1], s, which="y")

    draw_heatmap(axes[1, 2], Vt, [r"$v_1^T$", r"$v_2^T$", r"$v_3^T$"], OBS_NAMES, "Right singular-vector matrix $V^T$", center=0.0)

    draw_heatmap(axes[2, 0], eigvec_orth_gap, eig_labels, eig_labels, "Eigenvector non-orthogonality $Q^TQ-I$", center=0.0)

    eig_index = np.arange(1, len(eigvals) + 1)
    axes[2, 1].bar(eig_index, eigvals, color="tab:green", alpha=0.85)
    axes[2, 1].plot(eig_index, eigvals, color="0.35", linewidth=1.1)
    axes[2, 1].axhline(0.0, color="0.35", linewidth=0.9)
    axes[2, 1].set_title("Eigenvalues sorted by $|\\lambda|$")
    axes[2, 1].set_xlabel("index")
    axes[2, 1].set_ylabel("eigenvalue")
    axes[2, 1].set_xticks(eig_index)
    maybe_use_symlog(axes[2, 1], eigvals, which="y")

    draw_heatmap(axes[2, 2], eigvecs, OBS_NAMES, eig_labels, "Right eigenvector matrix $Q$", center=0.0)

    fig.suptitle(
        f"Full flow summary: $\\lambda$={lam:.3g}, $\\mu$={mu:.3g}, "
        f"main steps={main_steps}, phase steps={phase_steps}",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    save_path = OUT_DIR / plot_name(lam, mu)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "lam": lam,
        "mu": mu,
        "main_steps": main_steps,
        "phase_steps": phase_steps,
        "sigma_1": float(s[0]),
        "sigma_2": float(s[1]),
        "sigma_3": float(s[2]),
        "eig_1": float(eigvals[0]),
        "eig_2": float(eigvals[1]),
        "eig_3": float(eigvals[2]),
        "fro_gap_U_V": uv_gap_fro,
        "fro_norm_QTQ_minus_I": eigvec_nonorth_fro,
        "obs_max_abs": float(np.max(np.abs(obs))),
        "phase_max_abs": float(np.max(np.abs(phase_points))) if len(phase_points) else 0.0,
        "file_name": save_path.name,
    }


def to_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        items: list[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                items.append(f"{float(value):.6g}")
            else:
                items.append(str(value))
        lines.append("| " + " | ".join(items) + " |")
    return "\n".join(lines)


def build_markdown_report(df: pd.DataFrame) -> str:
    largest_uv = df.sort_values("fro_gap_U_V", ascending=False).head(8)
    largest_nonorth = df.sort_values("fro_norm_QTQ_minus_I", ascending=False).head(8)
    largest_sigma = df.sort_values("sigma_1", ascending=False).head(8)
    most_stable = df.sort_values("obs_max_abs", ascending=True).head(8)

    same_scale = df[np.isclose(df["lam"], df["mu"])].sort_values("lam")
    lam_dominant = df[df["lam"] > df["mu"]].sort_values("fro_gap_U_V", ascending=False).head(8)
    mu_dominant = df[df["mu"] > df["lam"]].sort_values("fro_gap_U_V", ascending=False).head(8)

    lines = [
        "# figW batch analysis",
        "",
        "This file summarizes the batch experiment for the `3x3` full-flow figures.",
        "",
        "## Scan setup",
        "",
        f"- Output folder: `{OUT_DIR.relative_to(REPO_ROOT).as_posix()}`",
        f"- Number of cases: `{len(df)}`",
        f"- parameter pairs: `{PARAMETER_PAIRS}`",
        "- The listed parameter pairs are evaluated directly.",
        "- For large parameter values, the script uses fewer time steps, so the plots stay readable.",
        "- When a curve spans many orders of magnitude, the plot uses a symlog axis.",
        "",
        "## Main rough findings",
        "",
        "1. When `lam` and `mu` are both small, the phase portrait and lifted trajectories stay close to the origin, and the singular values are also small.",
        "2. When `lam` becomes much larger than `mu`, the off-diagonal term `lam^2 - mu` becomes very large, and the lifted observable panel changes much more strongly.",
        "3. When `mu` becomes much larger than `lam`, the `y` direction becomes dominant, and the phase portrait changes shape in a different way from the `lam`-dominant cases.",
        "4. When `lam` and `mu` are on the same scale, the figures usually look more balanced, and the left/right singular-vector matrices tend to change more smoothly across neighboring coarse scales.",
        "5. Very large values such as `1e4` do not just rescale the plots. They also change the relative structure of `A_o`, so the singular-value spectrum and the singular-vector heatmaps can look qualitatively different.",
        "",
        "## Cases with the largest difference between U and V",
        "",
        to_markdown_table(largest_uv, ["lam", "mu", "fro_gap_U_V", "sigma_1", "sigma_2", "sigma_3", "file_name"]),
        "",
        "## Cases with the largest first singular value",
        "",
        to_markdown_table(largest_sigma, ["lam", "mu", "sigma_1", "sigma_2", "sigma_3", "file_name"]),
        "",
        "## Cases with the largest eigenvector non-orthogonality",
        "",
        to_markdown_table(largest_nonorth, ["lam", "mu", "fro_norm_QTQ_minus_I", "eig_1", "eig_2", "eig_3", "file_name"]),
        "",
        "## Cases with the smallest observable amplitude",
        "",
        to_markdown_table(most_stable, ["lam", "mu", "obs_max_abs", "phase_max_abs", "file_name"]),
        "",
        "## Same-scale cases: lam = mu",
        "",
        to_markdown_table(same_scale, ["lam", "mu", "sigma_1", "sigma_2", "sigma_3", "fro_gap_U_V", "file_name"]),
        "",
        "## lam-dominant cases with large U-V gap",
        "",
        to_markdown_table(lam_dominant, ["lam", "mu", "fro_gap_U_V", "sigma_1", "sigma_2", "sigma_3", "file_name"]),
        "",
        "## mu-dominant cases with large U-V gap",
        "",
        to_markdown_table(mu_dominant, ["lam", "mu", "fro_gap_U_V", "sigma_1", "sigma_2", "sigma_3", "file_name"]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    rows: list[dict[str, float]] = []
    total = len(PARAMETER_PAIRS)
    for counter, (lam, mu) in enumerate(PARAMETER_PAIRS, start=1):
        print(f"[{counter:03d}/{total:03d}] building lam={lam:g}, mu={mu:g}")
        rows.append(save_full_flow_figure(lam, mu))

    df = pd.DataFrame(rows).sort_values(["lam", "mu"]).reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)
    MD_PATH.write_text(build_markdown_report(df), encoding="utf-8")
    print(f"Saved {len(df)} figures to {OUT_DIR}")
    print(f"Saved summary csv to {CSV_PATH}")
    print(f"Saved markdown report to {MD_PATH}")


if __name__ == "__main__":
    main()
