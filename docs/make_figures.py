# -*- coding: utf-8 -*-
"""docs/make_figures.py

Regenerates the method figures and the results chart embedded in README.md.

The method figures are schematic: they are drawn from a small hand-made episode
(the competition data is not redistributed in this repository), but every
data-dependent step is computed with the real functions in `kleague_v7_core.py`
(coordinate unification, event features, possession split, prefix heatmap,
fine-grid targets, spatial soft labels), so the pictures follow the code.
The results chart uses the final CV fold scores and the DACON leaderboard scores.
The two EDA plots (`eda_*.png`) come from the solution deck and are not generated here.

Usage (from the repository root):
    pip install -r requirements.txt matplotlib
    python docs/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import kleague_v7_core as K  # noqa: E402

OUT_DIR = ROOT / "docs" / "figures"
CFG = K.CFG()

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
        "savefig.facecolor": "white",
    }
)

# ----------------------------
# Palette
# ----------------------------
C_ATT = "#1f6feb"  # reference team (owner of the pass to predict)
C_DEF = "#e5534b"  # opponent
C_TGT = "#f2b90f"  # pass to predict
C_INK = "#1f2328"
C_MUTED = "#57606a"
C_GRID = "#d0d7de"
C_GRASS = "#3f8a4e"
C_GRASS_STRIPE = "#468f55"
C_COARSE = "#8250df"
C_FINE = "#fb8f44"

# ----------------------------
# Illustrative episode
# ----------------------------
TEAM_A, TEAM_B = 101, 202  # A = reference team (it owns the last pass)

# (team, type, start_x, start_y, end_x, end_y, time_seconds) in team A's frame.
_EPISODE_IN_A_FRAME = [
    (TEAM_B, "Pass", 70.0, 20.0, 60.0, 28.0, 0.0),
    (TEAM_B, "Pass", 60.0, 28.0, 52.0, 40.0, 2.1),
    (TEAM_B, "Pass", 52.0, 40.0, 45.0, 30.0, 4.0),
    (TEAM_A, "Recovery", 45.0, 30.0, 45.0, 30.0, 4.8),
    (TEAM_A, "Pass", 45.0, 30.0, 55.0, 18.0, 6.3),
    (TEAM_A, "Carry", 55.0, 18.0, 63.0, 16.0, 8.9),
    (TEAM_A, "Pass", 63.0, 16.0, 70.0, 34.0, 10.7),
    (TEAM_A, "Pass", 70.0, 34.0, 78.0, 52.0, 12.6),
    (TEAM_A, "Carry", 78.0, 52.0, 86.0, 55.0, 15.1),
    (TEAM_A, "Pass", 86.0, 55.0, 97.0, 38.0, 16.9),  # <- pass to predict
]


def make_raw_episode() -> pd.DataFrame:
    """Episode in the provider's actor-centric frame: every event is stored in
    the acting team's own attacking direction, so B's events are rotated."""
    rows = []
    for i, (team, typ, sx, sy, ex, ey, t) in enumerate(_EPISODE_IN_A_FRAME):
        if team != TEAM_A:
            sx, sy = K.rotate_180_xy(sx, sy)
            ex, ey = K.rotate_180_xy(ex, ey)
        rows.append(
            dict(
                game_id=1, game_episode="1_1", time_seconds=t, team_id=team,
                player_id=team * 100 + i % 4, action_id=i, type_name=typ,
                result_name="Successful", start_x=sx, start_y=sy, end_x=ex, end_y=ey,
                is_home=float(team == TEAM_A), period_id=1,
            )
        )
    return pd.DataFrame(rows)


RAW = make_raw_episode()
UNI = K.align_episode_to_ref_team(RAW, TEAM_A, CFG)  # the real coordinate-unification step
T_IDX = len(RAW) - 1
assert np.allclose(UNI[["start_x", "start_y"]].values, np.array(_EPISODE_IN_A_FRAME)[:, 2:4].astype(float))


def team_color(team_id) -> str:
    return C_ATT if team_id == TEAM_A else C_DEF


# ----------------------------
# Drawing helpers
# ----------------------------
def draw_pitch(ax, grass=True, line_color="white", lw=1.3, alpha=0.95, zorder=0, margin=2.5, spot_r=0.4):
    L, W = K.PITCH_X, K.PITCH_Y
    if grass:
        ax.add_patch(Rectangle((-margin, -margin), L + 2 * margin, W + 2 * margin, fc=C_GRASS, ec="none", zorder=zorder))
        n = 14
        for i in range(0, n, 2):
            ax.add_patch(Rectangle((i * L / n, 0), L / n, W, fc=C_GRASS_STRIPE, ec="none", zorder=zorder))
    line = dict(ec=line_color, fc="none", lw=lw, alpha=alpha, zorder=zorder + 1)
    dot = dict(fc=line_color, ec="none", alpha=alpha, zorder=zorder + 1)
    ax.add_patch(Rectangle((0, 0), L, W, **line))
    ax.plot([L / 2, L / 2], [0, W], color=line_color, lw=lw, alpha=alpha, zorder=zorder + 1)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, **line))
    ax.add_patch(Circle((L / 2, W / 2), spot_r * 1.1, **dot))
    theta = np.degrees(np.arccos((K.PENALTY_LENGTH - 11.0) / 9.15))
    for left in (True, False):
        x0 = 0.0 if left else L
        s = 1.0 if left else -1.0
        ax.add_patch(Rectangle((x0 if left else x0 - K.PENALTY_LENGTH, W / 2 - K.HALF_PENALTY_WIDTH),
                               K.PENALTY_LENGTH, K.PENALTY_WIDTH, **line))
        ax.add_patch(Rectangle((x0 if left else x0 - 5.5, W / 2 - 9.16), 5.5, 18.32, **line))
        spot = x0 + s * 11.0
        ax.add_patch(Circle((spot, W / 2), spot_r, **dot))
        a0 = -theta if left else 180.0 - theta
        ax.add_patch(Arc((spot, W / 2), 18.3, 18.3, theta1=a0, theta2=a0 + 2 * theta,
                         ec=line_color, lw=lw, alpha=alpha, zorder=zorder + 1))
        ax.add_patch(Rectangle((x0 - 1.6 if left else x0, W / 2 - 3.66), 1.6, 7.32, **line))
    ax.set_aspect("equal")
    ax.axis("off")


def arrow(ax, p0, p1, color, lw=2.0, ls="-", z=5, ms=14, alpha=1.0, shrink=2.0, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color, linestyle=ls,
                        zorder=z, alpha=alpha, shrinkA=shrink, shrinkB=shrink)
    )


def draw_episode(ax, df: pd.DataFrame, target_idx: int = T_IDX):
    for i, r in df.iterrows():
        col = team_color(r.team_id)
        s, e = (r.start_x, r.start_y), (r.end_x, r.end_y)
        if i == target_idx:
            arrow(ax, s, e, C_TGT, lw=2.6, ls=(0, (3, 2)), z=6)
            ax.add_patch(Circle(e, 2.3, fc="white", ec=C_TGT, lw=2.2, zorder=8))
            ax.text(e[0], e[1] - 0.1, "?", ha="center", va="center", fontsize=12, fontweight="bold", color=C_INK, zorder=9)
        elif np.hypot(e[0] - s[0], e[1] - s[1]) > 0.5:
            ls = "-" if r.type_name.startswith("Pass") else (0, (1.2, 1.4))
            arrow(ax, s, e, col, lw=2.2, ls=ls, z=5)
        ax.add_patch(Circle(s, 1.35, fc=col, ec="white", lw=1.3, zorder=7))


def box(ax, x, y, w, h, title, lines=(), fc="#f6f8fa", ec=C_MUTED, fs_title=12.5, fs_line=10.2,
        dashed=False, title_color=C_INK, line_color=C_INK, lw=1.6, title_gap=2.5, line_gap=1.95, valign="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec,
                                lw=lw, ls=(0, (4, 2.5)) if dashed else "-", zorder=2))
    block = title_gap + line_gap * len(lines)
    ty = y + h / 2 + block / 2 - title_gap / 2 if valign == "center" else y + h - 0.8 - title_gap / 2
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=fs_title, fontweight="bold",
            color=title_color, zorder=3)
    for j, ln in enumerate(lines):
        color = line_color
        if isinstance(ln, tuple):
            ln, color = ln
        ax.text(x + w / 2, ty - title_gap / 2 - line_gap * (j + 0.5) - 0.15, ln, ha="center", va="center",
                fontsize=fs_line, color=color, zorder=3)


def flow(ax, p0, p1, color="#6e7781", lw=1.5, ls="-", conn="arc3,rad=0", ms=12):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, linestyle=ls,
                                 connectionstyle=conn, shrinkA=0, shrinkB=0, zorder=1.5))


def save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path.relative_to(ROOT)}")


def prefix_heatmap(uni: pd.DataFrame) -> np.ndarray:
    g = uni.copy()
    g.loc[T_IDX, ["end_x", "end_y"]] = np.nan  # unknown at test time
    g_feat = K.compute_event_features_single_episode(g)
    pos = K.last_possession_start_index(g_feat["team_id"].values, TEAM_A, T_IDX)
    return K.build_episode_heatmap_prefix(g_feat, TEAM_A, T_IDX, CFG, pos_start_idx=pos)


# ----------------------------
# Figure 1: coordinate frames
# ----------------------------
def fig_coordinate_frames():
    mir = UNI.copy()
    mir["start_y"] = K.PITCH_Y - mir["start_y"]
    mir["end_y"] = K.PITCH_Y - mir["end_y"]

    panels = [
        (RAW, "① Raw data: actor-centric",
         "Each event is stored in the acting team's own\nattacking direction, so B's passes look disconnected.",
         [("A attacks →", C_ATT), ("B attacks →", C_DEF)]),
        (UNI, "② Unified to the passer's team",
         "Opponent events rotated by 180°:\n(x, y) → (105 − x, 68 − y). One continuous play.",
         [("A attacks →", C_ATT), ("← B attacks", C_DEF)]),
        (mir, "③ Mirror augmentation / TTA",
         "y → 68 − y on the unified frame.\nTraining: extra sample · Inference: averaged.",
         [("A attacks →", C_ATT), ("← B attacks", C_DEF)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    for ax, (df, title, caption, dirs) in zip(axes, panels):
        draw_pitch(ax)
        draw_episode(ax, df)
        ax.set_xlim(-3, 108)
        ax.set_ylim(-18, 85)
        ax.text(-2.5, 82, title, ha="left", va="center", fontsize=16.5, fontweight="bold", color=C_INK)
        ax.text(-2.5, 74.5, dirs[0][0], ha="left", va="center", fontsize=13.5, fontweight="bold", color=dirs[0][1])
        ax.text(107.5, 74.5, dirs[1][0], ha="right", va="center", fontsize=13.5, fontweight="bold",
                color=dirs[1][1])
        ax.text(52.5, -4.5, caption, ha="center", va="top", fontsize=13, color=C_INK, linespacing=1.4)

    handles = [
        Line2D([], [], color=C_ATT, lw=2.4, marker="o", markersize=7, label="team A (owns the last pass)"),
        Line2D([], [], color=C_DEF, lw=2.4, marker="o", markersize=7, label="team B (opponent)"),
        Line2D([], [], color=C_MUTED, lw=2.2, ls=(0, (1.2, 1.4)), label="carry"),
        Line2D([], [], color=C_TGT, lw=2.6, ls=(0, (3, 2)), label="pass to predict"),
    ]
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.1, wspace=0.04)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=13.5,
               bbox_to_anchor=(0.5, -0.03))
    save(fig, "coordinate_frames.png")


# ----------------------------
# Figure 2: dense supervision
# ----------------------------
def fig_dense_supervision():
    n = len(RAW)
    teams = RAW["team_id"].values
    types = RAW["type_name"].tolist()
    targets = [i for i, t in enumerate(types) if t.lower().startswith("pass")]  # train_only_pass_targets=True

    fig, ax = plt.subplots(figsize=(13, 6.6))
    n_rows = len(targets)
    y_strip = n_rows + 1.35

    for i in range(n):
        col = team_color(teams[i])
        is_target = i in targets
        ax.add_patch(FancyBboxPatch((i - 0.34, y_strip - 0.34), 0.68, 0.68,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=col if is_target else "white", ec=col, lw=2.0, zorder=3))
        ax.text(i, y_strip, str(i), ha="center", va="center", fontsize=12.5, fontweight="bold",
                color="white" if is_target else col, zorder=4)
        ax.text(i, y_strip + 0.5, types[i], ha="center", va="bottom", fontsize=10, color=C_MUTED)
    ax.text(-0.75, y_strip, "episode", ha="right", va="center", fontsize=12, fontweight="bold", color=C_INK)

    for r, t in enumerate(targets):
        y = n_rows - r
        ref = teams[t]
        col = team_color(ref)
        pos = K.last_possession_start_index(teams, ref, t)
        seq_start = pos if (t - pos + 1) >= CFG.min_episode_len else 0  # possession_fallback_full_prefix
        if pos > 0:
            ax.add_patch(Rectangle((-0.42, y - 0.24), pos - 0.16, 0.48, fc="#eaeef2", ec="#afb8c1",
                                   hatch="////", lw=1.0, zorder=2))
        ax.add_patch(FancyBboxPatch((seq_start - 0.42, y - 0.24), t - seq_start + 0.84, 0.48,
                                    boxstyle="round,pad=0,rounding_size=0.1", fc=col, ec="none",
                                    alpha=0.85, zorder=3))
        ax.scatter([t], [y], marker="*", s=330, c=C_TGT, edgecolors=C_INK, linewidths=1.0, zorder=5)
        side = "A" if ref == TEAM_A else "B"
        ax.text(-0.75, y, f"target: pass {t}  ·  ref {side}", ha="right", va="center", fontsize=11.5, color=col,
                fontweight="bold")
        is_last = t == n - 1
        w = CFG.pass_event_weight * (CFG.last_event_weight_multiplier if is_last else 1.0)
        label = f"w = {w:.1f}" + ("   (last pass × 2.5)" if is_last else "")
        ax.text(n - 0.35, y, label, ha="left", va="center", fontsize=11.5,
                color=C_INK, fontweight="bold" if is_last else "normal")

    ax.plot([-4.2, n + 2.4], [n_rows + 0.62] * 2, color=C_GRID, lw=1.0)
    ax.set_xlim(-4.3, n + 2.6)
    ax.set_ylim(0.3, y_strip + 1.25)
    ax.axis("off")
    ax.set_title("Dense supervision: every pass in an episode becomes a training sample",
                 loc="left", fontsize=14.5, x=0.0, pad=34)
    ax.text(0.0, 1.035, "Passes of both teams are used, each re-unified to its own passer's frame; "
                        "every sample also gets a y-mirrored copy.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=11, color=C_MUTED)

    handles = [
        Patch(fc=C_ATT, alpha=0.85, label="BiLSTM input: current possession (also in heatmap, weight 1.0)"),
        Patch(fc="#eaeef2", ec="#afb8c1", hatch="////", label="heatmap only: earlier possession (weight 0.5)"),
        Line2D([], [], marker="*", ls="none", markersize=15, markerfacecolor=C_TGT, markeredgecolor=C_INK,
               label="target step: its end (x, y) is the label; movement features are masked"),
        Patch(fc="white", ec=C_MUTED, label="non-pass event: context only, no sample"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=2, frameon=False,
              fontsize=10.5, handlelength=2.2, columnspacing=1.6)
    save(fig, "dense_supervision.png")


# ----------------------------
# Figure 3: prefix heatmap channels
# ----------------------------
def fig_heatmap_channels():
    hm = prefix_heatmap(UNI)
    titles = [
        "ch0 · attacker start", "ch1 · attacker end", "ch2 · defender start", "ch3 · defender end",
        "ch4 · attacker Σ dx", "ch5 · attacker Σ dy", "ch6 · defender Σ dx", "ch7 · defender Σ dy",
    ]
    vcount = float(hm[:4].max())
    vdisp = float(np.abs(hm[4:]).max())
    prefix = UNI.iloc[:T_IDX]

    fig, axes = plt.subplots(2, 4, figsize=(14.6, 6.1))
    for ch, ax in enumerate(axes.flat):
        if ch < 4:
            cmap, vmin, vmax = ("Blues" if ch < 2 else "Reds"), 0.0, vcount
        else:
            cmap, vmin, vmax = "PuOr_r", -vdisp, vdisp
        ax.imshow(hm[ch], origin="lower", extent=(0, K.PITCH_X, 0, K.PITCH_Y), cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest", zorder=0)
        for gx in range(1, CFG.heatmap_grid_x):
            ax.plot([gx * K.PITCH_X / CFG.heatmap_grid_x] * 2, [0, K.PITCH_Y], color="#afb8c1", lw=0.4, zorder=1)
        for gy in range(1, CFG.heatmap_grid_y):
            ax.plot([0, K.PITCH_X], [gy * K.PITCH_Y / CFG.heatmap_grid_y] * 2, color="#afb8c1", lw=0.4, zorder=1)
        draw_pitch(ax, grass=False, line_color="#424a53", lw=0.9, alpha=0.75, zorder=2)
        team = TEAM_A if ch in (0, 1, 4, 5) else TEAM_B
        for _, r in prefix[prefix.team_id == team].iterrows():
            if np.hypot(r.end_x - r.start_x, r.end_y - r.start_y) > 0.5:
                arrow(ax, (r.start_x, r.start_y), (r.end_x, r.end_y), "#24292f", lw=0.9, ms=7, z=4, alpha=0.55,
                      shrink=1.0)
        if ch == 0:
            row = UNI.iloc[T_IDX]
            ax.add_patch(Circle((row.start_x, row.start_y), 1.8, fc=C_TGT, ec=C_INK, lw=0.9, zorder=5))
        ax.set_xlim(-1.5, K.PITCH_X + 1.5)
        ax.set_ylim(-1.5, K.PITCH_Y + 1.5)
        ax.set_title(titles[ch], fontsize=14, loc="left")

    fig.suptitle("Prefix heatmap for the pass to predict  (8 channels × 8 × 12 cells of 8.75 m × 8.5 m)",
                 x=0.012, ha="left", fontsize=16.5, fontweight="bold", y=1.01)
    fig.text(0.012, 0.955,
             "Counts and summed displacements of all events before the target (earlier possession weighted 0.5),"
             "\nnormalized by #events.  Displacement: orange = positive, purple = negative.  "
             "Gold dot: start of the pass to predict.",
             ha="left", va="top", fontsize=12.5, color=C_MUTED, linespacing=1.45)
    fig.subplots_adjust(left=0.012, right=0.995, bottom=0.01, wspace=0.05, hspace=0.3, top=0.8)
    save(fig, "heatmap_channels.png")


# ----------------------------
# Figure 4: output grids + spatial soft labels
# ----------------------------
def soft_target(cell_id: int) -> np.ndarray:
    """Soft label used by `spatial_soft_ce_loss_vec`, recovered from the loss itself:
    at logits = 0 its gradient is softmax(0) - target = 1/C - target."""
    gx, gy = CFG.fine_grid_x, CFG.fine_grid_y
    logits = torch.zeros(1, gx * gy, requires_grad=True)
    loss = K.spatial_soft_ce_loss_vec(logits, torch.tensor([cell_id]), gx, gy,
                                      kernel=CFG.fine_soft_kernel, sigma=CFG.fine_soft_sigma)
    loss.sum().backward()
    return (1.0 / (gx * gy) - logits.grad[0]).numpy().reshape(gy, gx)


def fig_fine_grid_soft_labels():
    gx, gy = CFG.fine_grid_x, CFG.fine_grid_y
    cw, ch = K.PITCH_X / gx, K.PITCH_Y / gy
    zx, zy = CFG.zone_grid_x, CFG.zone_grid_y
    examples = [("interior", (57.5, 44.8)), ("corner", (103.2, 2.1))]
    cmap = plt.get_cmap("YlOrRd")

    fig = plt.figure(figsize=(16.5, 5.4))
    gs = GridSpec(1, 3, width_ratios=[2.05, 1, 1], wspace=0.14, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax.add_patch(Rectangle((0, 0), K.PITCH_X, K.PITCH_Y, fc="#f6f8fa", ec="none", zorder=0))
    targets = {}
    for name, (x, y) in examples:
        fid, res = K.compute_fine_id_and_residual(x, y, CFG)
        soft = soft_target(fid)
        targets[name] = (x, y, fid, res, soft)
        vmax = soft.max()
        for iy in range(gy):
            for ix in range(gx):
                if soft[iy, ix] > 1e-6:
                    ax.add_patch(Rectangle((ix * cw, iy * ch), cw, ch, fc=cmap(0.15 + 0.8 * soft[iy, ix] / vmax),
                                           ec="none", zorder=1))
        ix0, iy0 = fid % gx, fid // gx
        ax.add_patch(Rectangle((ix0 * cw, iy0 * ch), cw, ch, fc="none", ec=C_INK, lw=1.8, zorder=4))
        ax.scatter([x], [y], s=16, c=C_INK, zorder=5)
    for i in range(1, gx):
        ax.plot([i * cw] * 2, [0, K.PITCH_Y], color="#c3ccd5", lw=0.5, zorder=2)
    for j in range(1, gy):
        ax.plot([0, K.PITCH_X], [j * ch] * 2, color="#c3ccd5", lw=0.5, zorder=2)
    for i in range(1, zx):
        ax.plot([i * K.PITCH_X / zx] * 2, [0, K.PITCH_Y], color=C_COARSE, lw=1.6, alpha=0.75, zorder=3)
    for j in range(1, zy):
        ax.plot([0, K.PITCH_X], [j * K.PITCH_Y / zy] * 2, color=C_COARSE, lw=1.6, alpha=0.75, zorder=3)
    draw_pitch(ax, grass=False, line_color="#57606a", lw=1.0, alpha=0.8, zorder=3)
    ax.set_xlim(-2, K.PITCH_X + 2)
    ax.set_ylim(-2, K.PITCH_Y + 2)
    ax.set_title("(a) Output grids and soft targets", loc="left", fontsize=15)
    ax.legend(handles=[
        Line2D([], [], color=C_COARSE, lw=2.0, label="zone grid 6 × 4 = 24 classes (auxiliary)"),
        Line2D([], [], color="#aab4be", lw=1.4, label="fine grid 24 × 16 = 384 classes"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=12.5,
        columnspacing=1.5, handlelength=1.8)

    def zoom(axz, name, title):
        x, y, fid, res, soft = targets[name]
        ix0, iy0 = fid % gx, fid // gx
        r = 3
        vmax = soft.max()
        for iy in range(iy0 - r, iy0 + r + 1):
            for ix in range(ix0 - r, ix0 + r + 1):
                inside = 0 <= ix < gx and 0 <= iy < gy
                v = soft[iy, ix] if inside else 0.0
                fc = cmap(0.15 + 0.8 * v / vmax) if v > 1e-6 else ("#f6f8fa" if inside else "white")
                axz.add_patch(Rectangle((ix, iy), 1, 1, fc=fc, ec="#afb8c1" if inside else "#eaeef2", lw=0.8,
                                        hatch=None if inside else "////", zorder=1))
                if v > 1e-6:
                    axz.text(ix + 0.5, iy + 0.5, f"{100 * v:.1f}", ha="center", va="center", fontsize=11,
                             color="white" if v / vmax > 0.55 else C_INK, zorder=3)
        axz.add_patch(Rectangle((ix0, iy0), 1, 1, fc="none", ec=C_INK, lw=2.2, zorder=4))
        axz.set_xlim(ix0 - r - 0.1, ix0 + r + 1.1)
        axz.set_ylim(iy0 - r - 0.1, iy0 + r + 1.1)
        axz.set_aspect(ch / cw)
        axz.set_anchor("N")
        axz.axis("off")
        axz.set_title(title, loc="left", fontsize=14.5)

    ax_b = fig.add_subplot(gs[1])
    zoom(ax_b, "interior", "(b) 5×5 Gaussian, σ = 1")
    ax_b.text(0.5, -0.03, "target mass in %; the true cell\n(outlined) keeps 16 %",
              transform=ax_b.transAxes, ha="center", va="top", fontsize=12.5, color=C_MUTED)
    ax_c = fig.add_subplot(gs[2])
    zoom(ax_c, "corner", "(c) Pitch corner")
    ax_c.text(0.5, -0.03, "off-pitch cells are dropped,\nkernel renormalized",
              transform=ax_c.transAxes, ha="center", va="top", fontsize=12.5, color=C_MUTED)
    ax.set_anchor("N")
    save(fig, "fine_grid_soft_labels.png")


# ----------------------------
# Figure 5: top-K expected decoding
# ----------------------------
def fig_topk_decoding():
    gx, gy = CFG.fine_grid_x, CFG.fine_grid_y
    cw, ch = K.PITCH_X / gx, K.PITCH_Y / gy
    cell = np.array([cw, ch])
    start = UNI.iloc[T_IDX][["start_x", "start_y"]].values.astype(float)
    coarse = np.array([91.0, 47.6])  # s + offset head (illustrative output)
    modes = [np.array([97.0, 38.0]), np.array([90.8, 45.6])]

    # Two hypothetical fine-head outputs for the same pass: {(ix, iy): p}; the remaining mass sits outside the top-8.
    cases = [
        ("(a) Uncertain fine head", {(22, 8): 0.25, (20, 10): 0.16, (21, 9): 0.12, (22, 9): 0.10,
                                     (21, 8): 0.08, (20, 9): 0.07, (21, 10): 0.05, (23, 8): 0.04}),
        ("(b) Confident fine head", {(22, 8): 0.70, (22, 9): 0.07, (21, 8): 0.05, (23, 8): 0.04,
                                     (21, 9): 0.03, (22, 7): 0.02, (23, 9): 0.02, (21, 7): 0.01}),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.6))
    for ax, (title, probs) in zip(axes, cases):
        draw_pitch(ax, margin=6, spot_r=0.0)
        cells = list(probs.keys())
        pk = np.array([probs[c] for c in cells])
        w = pk / pk.sum()
        pmax = float(pk.max())
        alpha = 1.0 - pmax  # the gate is regularized toward 1 - p_max
        use_modes = modes if pmax < 0.5 else modes[:1]
        e_k = []
        for k, (ix, iy) in enumerate(cells):
            c = (np.array([ix, iy]) + 0.5) * cell
            m = min(use_modes, key=lambda mm: np.sum((mm - c) ** 2))
            r = np.clip((m - c) / cell * 0.8, -0.5, 0.5)  # residual in cell units, |r| <= 0.5
            e = c + r * cell
            e_k.append(e)
            ax.add_patch(Rectangle((ix * cw, iy * ch), cw, ch, fc=C_FINE, ec="white", lw=0.9,
                                   alpha=0.3 + 0.65 * w[k] / w.max(), zorder=3))
            ax.text(ix * cw + cw - 0.3, iy * ch + 0.3, f"{100 * pk[k]:.0f}%", ha="right", va="bottom",
                    fontsize=11.5, color=C_INK, zorder=6)
            ax.add_patch(Circle(c, 0.3, fc="white", ec=C_INK, lw=0.6, zorder=6))
            arrow(ax, c, e, "white", lw=1.4, ms=9, z=6, shrink=1.0)
        e_k = np.array(e_k)
        fine = (w[:, None] * e_k).sum(0)
        y_hat = alpha * coarse + (1 - alpha) * fine

        for i in range(int(72 // cw), gx + 1):
            ax.plot([i * cw] * 2, [20, 64], color="white", lw=0.5, alpha=0.4, zorder=2)
        for j in range(int(20 // ch), int(64 // ch) + 1):
            ax.plot([72, K.PITCH_X], [j * ch] * 2, color="white", lw=0.5, alpha=0.4, zorder=2)

        ax.add_patch(Circle(start, 0.85, fc=C_ATT, ec="white", lw=1.4, zorder=8))
        arrow(ax, start, coarse, C_COARSE, lw=2.2, ls=(0, (4, 2)), ms=14, z=7, shrink=4)
        ax.plot([coarse[0], fine[0]], [coarse[1], fine[1]], color="white", lw=1.5, ls=(0, (2, 2)), zorder=8)
        ax.scatter(*coarse, marker="s", s=150, c=C_COARSE, edgecolors="white", linewidths=1.5, zorder=9)
        ax.scatter(*fine, marker="D", s=130, c=C_FINE, edgecolors=C_INK, linewidths=1.3, zorder=9)
        ax.scatter(*y_hat, marker="*", s=480, c=C_TGT, edgecolors=C_INK, linewidths=1.2, zorder=10)

        ax.set_xlim(79.0, 107.5)
        ax.set_ylim(27.0, 59.5)
        ax.set_title(f"{title}\np_max = {pmax:.2f}  →  gate α ≈ 1 − p_max = {alpha:.2f}", loc="left",
                     fontsize=15)
        lean = "stays close to the coarse regression" if alpha > 0.5 else "follows the fine-grid decoder"
        ax.text(0.02, 0.025, f"ŷ {lean}", transform=ax.transAxes, fontsize=13, color="white",
                fontweight="bold", zorder=11)

    handles = [
        Line2D([], [], marker="o", ls="none", markersize=10, markerfacecolor=C_ATT, markeredgecolor="white",
               label="s: pass start"),
        Line2D([], [], marker="s", ls="none", markersize=11, markerfacecolor=C_COARSE, markeredgecolor="white",
               label="coarse = s + offset head"),
        Patch(fc=C_FINE, alpha=0.8, label="top-8 fine cells (label = p_k)"),
        Line2D([], [], color="#8c959f", lw=1.3, marker=">", markersize=6, label="residual r_k: cell center → e_k"),
        Line2D([], [], marker="D", ls="none", markersize=10, markerfacecolor=C_FINE, markeredgecolor=C_INK,
               label="fine = Σ w_k e_k"),
        Line2D([], [], marker="*", ls="none", markersize=19, markerfacecolor=C_TGT, markeredgecolor=C_INK,
               label="ŷ = α·coarse + (1 − α)·fine"),
    ]
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.13, top=0.86, wspace=0.05)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=13,
               bbox_to_anchor=(0.5, -0.02), columnspacing=2.0)
    fig.suptitle("Top-K expected decoding (K = 8): one differentiable decoder for the loss and for inference",
                 x=0.012, ha="left", fontsize=16, fontweight="bold", y=0.99)
    save(fig, "topk_decoding.png")


# ----------------------------
# Figure 6: architecture
# ----------------------------
def mini_pitch(ax, x0, y0, w, h, color="#57606a", lw=0.8, z=4):
    """Pitch outline scaled into the box (x0, y0, w, h) of a diagram axes."""
    sx, sy = w / K.PITCH_X, h / K.PITCH_Y
    kw = dict(fc="none", ec=color, lw=lw, zorder=z)
    ax.add_patch(Rectangle((x0, y0), w, h, **kw))
    ax.plot([x0 + w / 2] * 2, [y0, y0 + h], color=color, lw=lw, zorder=z)
    ax.add_patch(Circle((x0 + w / 2, y0 + h / 2), 9.15 * sy, **kw))
    pa_w, pa_h = K.PENALTY_LENGTH * sx, K.PENALTY_WIDTH * sy
    ax.add_patch(Rectangle((x0, y0 + (h - pa_h) / 2), pa_w, pa_h, **kw))
    ax.add_patch(Rectangle((x0 + w - pa_w, y0 + (h - pa_h) / 2), pa_w, pa_h, **kw))


def fig_architecture():
    fig = plt.figure(figsize=(14.0, 8.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 84)
    ax.axis("off")

    SEQ = dict(fc="#ddeaff", ec=C_ATT)
    CTX = dict(fc="#dcf5e3", ec="#1a7f37")
    SCL = dict(fc="#eef1f4", ec=C_MUTED)
    SPA = dict(fc="#ffeedd", ec="#d4700a")
    FUS = dict(fc="#ece2fd", ec=C_COARSE)
    HED = dict(fc="#fff4c2", ec="#9a6700")
    DEC = dict(fc="#ffe4e1", ec="#cf222e")
    LOSS = "#8250df"
    POOL = dict(fs_title=11.5, fs_line=9.8, title_gap=2.2, line_gap=1.8)

    for x, label in [(14, "INPUTS"), (43.5, "ENCODERS"), (66.5, "POOLING"), (81, "FUSION"), (100, "HEADS"),
                     (127.5, "DECODING")]:
        ax.text(x, 80.3, label, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#8c959f")

    # --- inputs
    box(ax, 1, 56, 26, 20, "Event sequence", ["current possession · T ≤ 40", "16 numeric features",
                                               "+ event-type emb (16)", "+ result emb (8)",
                                               "+ attack/defense side emb (4)"], **SEQ)
    box(ax, 1, 37, 26, 15.5, "Context IDs", ["episode cluster (k=4)", "team · opponent",
                                              "team styles (k=4) × 2", "passer · player role (k=5)"], **CTX)
    box(ax, 1, 27.5, 26, 7, "Episode stats (3)", ["mean speed · angle std · #events"], **SCL)
    box(ax, 1, 19, 26, 6.5, "Pass start (x, y)", ["normalized by 105 × 68"], **SCL)
    box(ax, 1, 1.5, 26, 15.5, "Prefix heatmap", ["8 ch × 8 × 12 grid"], valign="top", **SPA)
    hm = prefix_heatmap(UNI)
    thumb = (hm[0] + hm[1]) - (hm[2] + hm[3])  # attacker minus defender occupancy
    v = np.abs(thumb).max()
    tx, ty, tw, th = 7.4, 2.6, 13.2, 13.2 * K.PITCH_Y / K.PITCH_X
    ax.imshow(thumb, origin="lower", extent=(tx, tx + tw, ty, ty + th), cmap="RdBu", vmin=-v, vmax=v,
              interpolation="nearest", aspect="auto", zorder=3)
    mini_pitch(ax, tx, ty, tw, th)

    # --- encoders
    box(ax, 32, 65, 23, 11, "BiLSTM × 3", ["hidden 256 × 2 directions", "packed · dropout 0.3"], **SEQ)
    box(ax, 32, 56, 23, 6.5, "Context gate", ["H ⊙ σ(W · ctx)"], **SEQ)
    box(ax, 32, 39, 23, 11.5, "Embeddings", ["concat → 44-d", ("gate sees 32-d (no player id)", C_MUTED)], **CTX)
    box(ax, 32, 27.5, 23, 7, "MLP", ["LayerNorm → Linear → 32"], **SCL)
    box(ax, 32, 19, 23, 6.5, "MLP", ["Linear → 16"], **SCL)
    box(ax, 32, 3.5, 23, 11.5, "Heatmap CNN", ["Conv3×3 16 → MaxPool", "Conv3×3 32 → GAP → 32"], **SPA)

    # --- pooling
    box(ax, 59, 70, 15, 6, "Attention pool", ["additive · 512"], **POOL, **SEQ)
    box(ax, 59, 62.5, 15, 6, "Last-8 mean", ["window · 512"], **POOL, **SEQ)
    box(ax, 59, 55, 15, 6, "Last step", ["hidden · 512"], **POOL, **SEQ)

    # --- fusion
    ax.add_patch(FancyBboxPatch((78, 1.5), 6, 74.5, boxstyle="round,pad=0,rounding_size=1.0", lw=1.8, zorder=2,
                                **FUS))
    ax.text(81, 38.75, r"concat   $\mathbf{h} \in \mathbb{R}^{1660}$", rotation=90, ha="center", va="center",
            fontsize=13.5, fontweight="bold", color=C_INK, zorder=3)

    # --- heads
    hx, hw = 88.5, 23
    box(ax, hx, 66, hw, 10, "Coarse offset", ["MLP 256 → (Δx, Δy)", ("MSE + coarse distance", LOSS)], **HED)
    box(ax, hx, 53.5, hw, 10, "Zone logits (aux)", ["MLP 128 → 6 × 4 · train only", ("CE, label smoothing", LOSS)],
        dashed=True, **HED)
    box(ax, hx, 41, hw, 10, "Fine-cell logits", ["MLP 256 → 24 × 16", ("spatial soft-label CE", LOSS)], **HED)
    box(ax, hx, 20.5, hw, 13, "Residual", ["[h ; cell emb 16] → MLP 256", "tanh × 0.5 cell, per top-K cell",
                                          ("Smooth-L1 @ true cell", LOSS)], **HED)
    box(ax, hx, 4, hw, 11, "Mix gate α", ["MLP 64 → sigmoid", ("(α − (1 − p_max))²", LOSS)], **HED)

    # --- decoder
    box(ax, 116, 27, 23, 46, "Top-K expected", [
        r"$p = \mathrm{softmax}(\ell_{\mathrm{fine}})$",
        "keep the top K = 8 cells",
        r"$w_k = p_k \,/\, \Sigma_{\mathrm{top}K}\; p_j$",
        r"$e_k = c_k + r_k \odot \mathrm{cell}$",
        r"$\mathrm{coarse} = s + \hat{\Delta} \odot (105, 68)$",
        "",
        r"$\hat{y} = \alpha\;\mathrm{coarse}$",
        r"$\quad +\, (1-\alpha)\; \Sigma_k\, w_k\, e_k$",
        "",
        ("loss: Euclidean distance", LOSS),
        ("(training) = the metric", LOSS),
    ], fs_line=10.8, line_gap=2.75, **DEC)
    ax.add_patch(FancyBboxPatch((116, 13.5), 23, 9, boxstyle="round,pad=0,rounding_size=1.0", fc="#ffd33d",
                                ec="#9a6700", lw=1.8, zorder=2))
    ax.text(127.5, 19.7, "(end_x, end_y)", ha="center", va="center", fontsize=12.5, fontweight="bold", zorder=3)
    ax.text(127.5, 16.3, "clipped to the pitch", ha="center", va="center", fontsize=10, color=C_INK, zorder=3)
    ax.text(127.5, 7.0, "inference: 5 folds × mirror TTA,\nweighted by 1 / CV score", ha="center", va="center",
            fontsize=10, color=C_MUTED, linespacing=1.4)

    # --- arrows: inputs -> encoders -> pooling -> fusion
    for y in (70.5, 44.75, 31.0, 22.25, 9.25):
        flow(ax, (27, y), (32, y))
    flow(ax, (43.5, 65), (43.5, 62.5))  # BiLSTM -> gate
    flow(ax, (43.5, 50.5), (43.5, 56), color="#1a7f37")  # ctx -> gate
    ax.text(44.5, 53.2, "ctx", fontsize=10, color="#1a7f37", va="center")
    for y in (73, 65.5, 58):
        flow(ax, (55, 59.25), (59, y))
        flow(ax, (74, y), (78, y))
    for y in (44.75, 31.0, 22.25, 9.25):
        flow(ax, (55, y), (78, y))
    # fusion -> heads
    for y in (71, 58.5, 46, 27, 9.5):
        flow(ax, (84, y), (88.5, y))
    flow(ax, (100, 41), (100, 33.5), color="#9a6700")
    ax.text(101, 37.2, "top-K cell ids", fontsize=9.8, color="#9a6700", va="center")
    # heads -> decoder (the zone head only feeds the loss)
    flow(ax, (111.5, 71), (116, 66))
    flow(ax, (111.5, 46), (116, 50))
    flow(ax, (111.5, 27), (116, 38))
    flow(ax, (111.5, 9.5), (116, 31))
    flow(ax, (127.5, 27), (127.5, 22.5), color="#9a6700")
    save(fig, "architecture.png")


# ----------------------------
# Figure 7: results (final v7 training log + DACON leaderboard)
# ----------------------------
FOLD_SCORES = [12.6756, 12.8810, 12.9350, 12.8706, 12.9302]  # best ValLastDist (m) per fold
PUBLIC_LB, PRIVATE_LB = "12.45", "12.56885"  # as reported by DACON


def fig_results():
    fs = np.array(FOLD_SCORES)
    w = 1.0 / (fs + 1e-6)
    w = w / w.sum()  # ensemble weights, as in predict_test_ensemble_from_artifacts
    mean, std = float(fs.mean()), float(fs.std())

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    fold_y = [8.0, 7.2, 6.4, 5.6, 4.8]
    ax.plot([mean, mean], [4.4, 8.4], color=C_COARSE, lw=1.2, ls=(0, (4, 3)), alpha=0.6, zorder=1)
    weight_x = ("axes fraction", "data")
    ax.annotate("ensemble weight", (1.02, 8.75), xycoords=weight_x, va="center", fontsize=12, color=C_MUTED)
    for i, (y, s) in enumerate(zip(fold_y, fs)):
        ax.scatter(s, y, s=110, c=C_ATT, edgecolors="white", linewidths=1.2, zorder=3)
        ax.text(s + 0.018, y, f"{s:.4f}", va="center", fontsize=12.5, color=C_INK)
        ax.annotate(f"{w[i]:.3f}", (1.02, y), xycoords=weight_x, va="center", fontsize=12.5, color=C_INK)
    ax.errorbar(mean, 3.8, xerr=std, fmt="o", ms=11, color=C_COARSE, mec="white", mew=1.2, capsize=6, lw=2,
                zorder=3)
    ax.text(mean + std + 0.018, 3.8, f"{mean:.4f} ± {std:.4f}", va="center", fontsize=12.5, color=C_INK,
            fontweight="bold")
    for y, label in [(1.9, PUBLIC_LB), (1.1, PRIVATE_LB)]:
        ax.scatter(float(label), y, s=260, marker="*", c=C_TGT, edgecolors=C_INK, linewidths=1.0, zorder=3)
        ax.text(float(label) + 0.02, y, label, va="center", fontsize=12.5, color=C_INK, fontweight="bold")

    ax.axhline(2.95, color=C_GRID, lw=1.0)
    ax.text(12.305, 8.75, "single models on their validation folds (no TTA)", fontsize=12, color=C_MUTED,
            va="center")
    ax.text(12.305, 2.55, "5-fold ensemble + mirror TTA on the test set", fontsize=12, color=C_MUTED, va="center")
    ax.set_yticks(fold_y + [3.8, 1.9, 1.1])
    ax.set_yticklabels([f"Fold {i}" for i in range(len(fs))] + ["CV mean ± std", "Public LB", "Private LB"],
                       fontsize=12.5)
    ax.set_xlim(12.3, 13.15)
    ax.set_ylim(0.6, 9.2)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel("mean Euclidean distance on the last pass (m)  —  lower is better", fontsize=12.5)
    ax.grid(axis="x", color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Validation and leaderboard scores (final v7 submission)", loc="left", fontsize=15)
    save(fig, "results.png")


def main():
    fig_coordinate_frames()
    fig_dense_supervision()
    fig_heatmap_channels()
    fig_fine_grid_soft_labels()
    fig_topk_decoding()
    fig_architecture()
    fig_results()


if __name__ == "__main__":
    main()
