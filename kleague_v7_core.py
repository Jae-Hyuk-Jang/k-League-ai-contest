# -*- coding: utf-8 -*-
"""
kleague_v7_core.py

KLeagueAI Open Track 1 - Pass End Coordinate Prediction (v7)
Dense Supervision + Heatmap CNN + Fine-Grid & Residual Head
+ expected TOP-K inference-style distance, EMA, spatial soft labels, attacker/defender indicator, mirror TTA.

This file contains all core utilities (feature engineering, dataset, model, training, inference)
so that `train.py` and `inference.py` can be kept small and clean.

Notes
- All paths should be handled as *relative paths* from your project root by default:
    ./data, ./weights, ./output
- Determinism: we set seeds + cuDNN flags. Full determinism on GPU is not guaranteed
  for all kernels, but this matches the original training environment as closely as possible.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Optional dependency (scikit-learn pulls it in)
try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None


# ----------------------------
# Global constants
# ----------------------------
PITCH_X = 105.0
PITCH_Y = 68.0

GOAL_X = PITCH_X
GOAL_Y = PITCH_Y / 2.0
PENALTY_LENGTH = 16.5
PENALTY_WIDTH = 40.32
HALF_PENALTY_WIDTH = PENALTY_WIDTH / 2.0


# ============================
# Config
# ============================
@dataclass
class CFG:
    # Paths (set in build_cfg / CLI)
    data_root: str = "./data"
    train_csv: str = "./data/train.csv"
    test_meta_csv: str = "./data/test.csv"
    sample_submission_csv: str = "./data/sample_submission.csv"

    # Output dirs (weights / output)
    weights_dir: str = "./weights"
    output_dir: str = "./output"
    submission_filename: str = "exp_dense_lstm_heatmap_v7_ema_softlabel_att_side_tta.csv"

    # Training hyperparams
    n_folds: int = 5
    num_epochs: int = 25
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stopping_patience: int = 6
    num_workers: int = 0  # reproducibility

    # AMP
    use_amp: bool = True

    # Fine-tuning (stage-2)
    do_finetune_last_pass_only: bool = True
    finetune_epochs: int = 6
    finetune_lr: float = 3e-4
    finetune_patience: int = 3
    finetune_freeze_backbone: bool = True
    finetune_use_mirror_aug: bool = True

    # LSTM architecture
    hidden_dim: int = 256
    num_layers: int = 3
    bidirectional: bool = True
    dropout: float = 0.3

    # Embedding dims
    type_emb_dim: int = 16
    result_emb_dim: int = 8
    cluster_emb_dim: int = 4
    team_emb_dim: int = 8
    player_emb_dim: int = 12
    player_role_emb_dim: int = 4
    team_style_emb_dim: int = 4

    # Fine-grid residual head
    fine_grid_x: int = 24
    fine_grid_y: int = 16
    fine_cell_emb_dim: int = 16
    residual_max_norm: float = 0.5  # residual in cell-units range [-0.5,0.5]

    # Inference-style distance computation
    fine_topk: int = 8
    fine_temperature: float = 1.0
    use_mix_gate: bool = True

    # Gate regularization
    gate_loss_weight: float = 0.03  # 0 disables
    gate_target_power: float = 1.0
    gate_detach_pmax: bool = True

    # Extra coarse distance loss
    coarse_dist_weight: float = 0.08  # 0 disables

    # Attention
    attn_dim: int = 64

    # Sequence control
    max_seq_len: int = 40
    min_episode_len: int = 2
    use_last_possession_only: bool = True
    possession_fallback_full_prefix: bool = True

    # Coordinate unification
    use_coord_unification: bool = True

    # Loss weights (stage-1)
    mse_weight: float = 1.0
    dist_weight: float = 0.35
    dist_divisor: float = 40.0
    zone_loss_weight: float = 0.05
    fine_loss_weight: float = 0.35
    residual_loss_weight: float = 0.60

    # Stage-2 loss re-weighting (fine-tune)
    finetune_mse_weight: float = 0.5
    finetune_dist_weight: float = 0.55
    finetune_zone_loss_weight: float = 0.02
    finetune_fine_loss_weight: float = 0.45
    finetune_residual_loss_weight: float = 0.75
    finetune_gate_loss_weight: float = 0.03
    finetune_coarse_dist_weight: float = 0.08

    # Label smoothing
    zone_label_smoothing: float = 0.03
    fine_label_smoothing: float = 0.02

    # Sample weighting
    non_pass_event_weight: float = 0.35
    pass_event_weight: float = 1.0
    last_event_weight_multiplier: float = 2.5

    # Final window size for pooling
    final_window_size: int = 8

    # Data augmentation
    use_mirror_aug: bool = True  # Y-axis flip

    # Zone grid (auxiliary)
    zone_grid_x: int = 6
    zone_grid_y: int = 4

    # Heatmap grid
    heatmap_grid_x: int = 12
    heatmap_grid_y: int = 8
    heatmap_channels: int = 8

    # Possession emphasis in heatmap
    heatmap_weight_pre_possession: float = 0.5
    heatmap_weight_possession: float = 1.0

    # Episode / role / style cluster counts
    ep_n_clusters: int = 4
    player_role_n_clusters: int = 5
    team_style_n_clusters: int = 4

    # Training target filtering
    train_only_pass_targets: bool = True

    # Fine-grid spatial soft labels
    fine_use_spatial_soft: bool = True
    fine_soft_kernel: int = 5
    fine_soft_sigma: float = 1.0

    # Attacking/defending side embedding per timestep
    att_side_emb_dim: int = 4

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.995
    ema_warmup_steps: int = 200
    eval_use_ema: bool = False  # usually False (match original)

    # Test-Time Augmentation (mirror-y)
    use_tta_mirror: bool = True

    # Seed
    seed: int = 42


def build_cfg(
    data_root: str = "./data",
    weights_dir: str = "./weights",
    output_dir: str = "./output",
    submission_filename: Optional[str] = None,
) -> CFG:
    cfg = CFG()
    cfg.data_root = data_root
    cfg.train_csv = os.path.join(data_root, "train.csv")
    cfg.test_meta_csv = os.path.join(data_root, "test.csv")
    cfg.sample_submission_csv = os.path.join(data_root, "sample_submission.csv")
    cfg.weights_dir = weights_dir
    cfg.output_dir = output_dir
    if submission_filename:
        cfg.submission_filename = submission_filename
    return cfg


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================
# Feature engineering base cols
# ============================
SEQ_BASE_COLS = [
    "start_x",
    "start_y",
    "pass_dx",
    "pass_dy",
    "pass_dist",
    "dt",
    "speed",
    "angle",
    "angle_diff",
    "rel_time",
    "is_home",
    "period_id",
    "idx_from_end_norm",
    "dist_goal_start",
    "angle_to_goal",
    "in_box_start",
]


# ============================
# Coordinate unification utils
# ============================
def rotate_180_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return (PITCH_X - x, PITCH_Y - y)


def align_episode_to_ref_team(g: pd.DataFrame, ref_team_id, cfg: CFG) -> pd.DataFrame:
    """
    Coordinate unification:
      - Keep ref_team events as-is
      - Rotate opponent events by 180°: (x,y) -> (105-x, 68-y)
    """
    if (not cfg.use_coord_unification) or (ref_team_id is None) or (len(g) == 0):
        return g.copy()

    g2 = g.copy()
    mask_opp = g2["team_id"] != ref_team_id

    sx = g2.loc[mask_opp, "start_x"].astype(float).values
    sy = g2.loc[mask_opp, "start_y"].astype(float).values
    sx2, sy2 = rotate_180_xy(sx, sy)
    g2.loc[mask_opp, "start_x"] = sx2
    g2.loc[mask_opp, "start_y"] = sy2

    ex = g2.loc[mask_opp, "end_x"].astype(float).values
    ey = g2.loc[mask_opp, "end_y"].astype(float).values
    nan_mask = np.isnan(ex) | np.isnan(ey)
    ex2, ey2 = rotate_180_xy(np.nan_to_num(ex, nan=0.0), np.nan_to_num(ey, nan=0.0))
    ex2[nan_mask] = np.nan
    ey2[nan_mask] = np.nan
    g2.loc[mask_opp, "end_x"] = ex2
    g2.loc[mask_opp, "end_y"] = ey2

    return g2


# ============================
# Grid helpers (zone & fine-grid)
# ============================
def compute_grid_id(x: float, y: float, gx: int, gy: int) -> int:
    cell_w = PITCH_X / gx
    cell_h = PITCH_Y / gy
    ix = int(x // cell_w)
    iy = int(y // cell_h)
    ix = max(0, min(gx - 1, ix))
    iy = max(0, min(gy - 1, iy))
    return iy * gx + ix


def grid_center_from_id(cell_id: int, gx: int, gy: int) -> Tuple[float, float, float, float]:
    cell_w = PITCH_X / gx
    cell_h = PITCH_Y / gy
    ix = int(cell_id % gx)
    iy = int(cell_id // gx)
    cx = (ix + 0.5) * cell_w
    cy = (iy + 0.5) * cell_h
    return float(cx), float(cy), float(cell_w), float(cell_h)


def batch_grid_centers(cell_ids: torch.Tensor, gx: int, gy: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    cell_w = PITCH_X / gx
    cell_h = PITCH_Y / gy
    ix = (cell_ids % gx).to(torch.float32)
    iy = (cell_ids // gx).to(torch.float32)
    cx = (ix + 0.5) * cell_w
    cy = (iy + 0.5) * cell_h
    return cx, cy, float(cell_w), float(cell_h)


def compute_zone_id(x: float, y: float, cfg: CFG) -> int:
    return compute_grid_id(x, y, cfg.zone_grid_x, cfg.zone_grid_y)


def compute_fine_id_and_residual(x: float, y: float, cfg: CFG) -> Tuple[int, np.ndarray]:
    fine_id = compute_grid_id(x, y, cfg.fine_grid_x, cfg.fine_grid_y)
    cx, cy, cell_w, cell_h = grid_center_from_id(fine_id, cfg.fine_grid_x, cfg.fine_grid_y)
    rx = (x - cx) / cell_w
    ry = (y - cy) / cell_h
    rx = float(np.clip(rx, -0.5, 0.5))
    ry = float(np.clip(ry, -0.5, 0.5))
    return int(fine_id), np.array([rx, ry], dtype=np.float32)


# ============================
# Per-episode event feature computation
# ============================
def compute_event_features_single_episode(g: pd.DataFrame) -> pd.DataFrame:
    """
    Compute movement + goal/box features for a single episode dataframe sorted by time.
    Safe for test last row with end_x/end_y NaN.
    """
    g = g.sort_values("time_seconds").reset_index(drop=True).copy()

    g["pass_dx"] = (g["end_x"] - g["start_x"]).astype(float)
    g["pass_dy"] = (g["end_y"] - g["start_y"]).astype(float)
    g["pass_dx"] = g["pass_dx"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    g["pass_dy"] = g["pass_dy"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    g["dt"] = g["time_seconds"].diff().fillna(0.05)
    g["dt"] = g["dt"].clip(lower=0.05)

    g["pass_dist"] = np.sqrt(g["pass_dx"] ** 2 + g["pass_dy"] ** 2)
    g["speed"] = g["pass_dist"] / g["dt"]
    g["speed"] = g["speed"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    g["angle"] = np.arctan2(g["pass_dy"], g["pass_dx"]).fillna(0.0)
    g["angle_diff"] = g["angle"].diff().fillna(0.0)

    t = g["time_seconds"].astype(float).values
    if len(t) <= 1:
        g["rel_time"] = 0.0
    else:
        denom = (t.max() - t.min()) + 1e-6
        g["rel_time"] = (t - t.min()) / denom

    g["is_home"] = g["is_home"].astype(float)

    g["idx_in_ep"] = np.arange(len(g))
    g["len_ep"] = len(g)
    g["idx_from_end"] = g["len_ep"] - 1 - g["idx_in_ep"]
    g["idx_from_end_norm"] = g["idx_from_end"] / max(len(g), 1)

    sx = g["start_x"].astype(float)
    sy = g["start_y"].astype(float)
    g["dist_goal_start"] = np.sqrt((GOAL_X - sx) ** 2 + (GOAL_Y - sy) ** 2)
    g["angle_to_goal"] = np.arctan2(GOAL_Y - sy, GOAL_X - sx).fillna(0.0)
    g["in_box_start"] = (
        (sx >= GOAL_X - PENALTY_LENGTH)
        & (sx <= GOAL_X)
        & (np.abs(sy - GOAL_Y) <= HALF_PENALTY_WIDTH)
    ).astype(float)

    return g


# ============================
# Global event features (full dataset)
# ============================
def add_event_features_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute event features for ALL rows (original coordinates).
    Safe for NaN end coords (fills with 0 in derived cols).
    """
    df = df.sort_values(["game_episode", "time_seconds"]).reset_index(drop=True).copy()

    df["pass_dx"] = (df["end_x"] - df["start_x"]).astype(float)
    df["pass_dy"] = (df["end_y"] - df["start_y"]).astype(float)

    df["dt"] = df.groupby("game_episode")["time_seconds"].diff().fillna(0.05)
    df["dt"] = df["dt"].clip(lower=0.05)

    df["pass_dist"] = np.sqrt(df["pass_dx"] ** 2 + df["pass_dy"] ** 2)
    df["speed"] = df["pass_dist"] / df["dt"]
    df["speed"] = df["speed"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["angle"] = np.arctan2(df["pass_dy"], df["pass_dx"]).fillna(0.0)
    df["angle_diff"] = df.groupby("game_episode")["angle"].diff().fillna(0.0)

    def _rel_time(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)

    df["rel_time"] = df.groupby("game_episode")["time_seconds"].transform(_rel_time)
    df["is_home"] = df["is_home"].astype(float)

    df["idx_in_ep"] = df.groupby("game_episode").cumcount()
    df["len_ep"] = df.groupby("game_episode")["idx_in_ep"].transform("max") + 1
    df["idx_from_end"] = df["len_ep"] - 1 - df["idx_in_ep"]
    df["idx_from_end_norm"] = df["idx_from_end"] / df["len_ep"].clip(lower=1)

    df["dist_goal_start"] = np.sqrt((GOAL_X - df["start_x"]) ** 2 + (GOAL_Y - df["start_y"]) ** 2)
    df["angle_to_goal"] = np.arctan2(GOAL_Y - df["start_y"], GOAL_X - df["start_x"]).fillna(0.0)
    df["in_box_start"] = (
        (df["start_x"] >= GOAL_X - PENALTY_LENGTH)
        & (df["start_x"] <= GOAL_X)
        & (np.abs(df["start_y"] - GOAL_Y) <= HALF_PENALTY_WIDTH)
    ).astype(float)

    for c in ["pass_dx", "pass_dy", "pass_dist", "speed", "angle", "angle_diff"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


# ============================
# Episode stats (RAW frame)
# ============================
def compute_episode_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Episode-level stats for clustering (RAW team-centric frame).
    - build-up only (exclude last event)
    """
    df = df.sort_values(["game_episode", "time_seconds"]).reset_index(drop=True).copy()

    if ("idx_in_ep" not in df.columns) or ("len_ep" not in df.columns):
        df["idx_in_ep"] = df.groupby("game_episode").cumcount()
        df["len_ep"] = df.groupby("game_episode")["idx_in_ep"].transform("max") + 1

    non_last_mask = df["idx_in_ep"] < (df["len_ep"] - 1)
    df_core = df[non_last_mask].copy()

    all_eps = df["game_episode"].unique()
    ep_stats = pd.DataFrame({"game_episode": all_eps})

    if len(df_core) > 0:
        grp_core = df_core.groupby("game_episode")
        ep_stats_core = grp_core.agg(
            mean_speed=("speed", "mean"),
            angle_std=("angle_diff", "std"),
            n_events=("action_id", "count"),
        ).reset_index()
        ep_stats_core["angle_std"] = ep_stats_core["angle_std"].fillna(0.0)
        ep_stats = ep_stats.merge(ep_stats_core, on="game_episode", how="left")
    else:
        ep_stats["mean_speed"] = np.nan
        ep_stats["angle_std"] = np.nan
        ep_stats["n_events"] = np.nan

    ep_stats["mean_speed"] = ep_stats["mean_speed"].fillna(0.0)
    ep_stats["angle_std"] = ep_stats["angle_std"].fillna(0.0)
    ep_stats["n_events"] = ep_stats["n_events"].fillna(0.0)
    return ep_stats


def fit_episode_cluster(ep_stats: pd.DataFrame, n_clusters: int = 4, seed: int = 42):
    stats_cols = ["mean_speed", "angle_std", "n_events"]
    X_df = ep_stats[stats_cols].astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(X_scaled.astype(np.float64))

    ep_stats = ep_stats.copy()
    ep_stats["cluster"] = kmeans.predict(X_scaled.astype(np.float64))
    return ep_stats, scaler, kmeans, stats_cols


def apply_episode_cluster_to_df(df: pd.DataFrame, ep_stats: pd.DataFrame) -> pd.DataFrame:
    cluster_map = ep_stats.set_index("game_episode")["cluster"].to_dict()
    df = df.copy()
    df["cluster"] = df["game_episode"].map(cluster_map).fillna(0).astype(int)
    return df


# ============================
# Categorical mappings
# ============================
def build_category_mappings(df: pd.DataFrame) -> Dict[str, Any]:
    df_local = df.copy()

    type_cats = sorted(df_local["type_name"].dropna().unique().tolist())
    type2idx = {t: i for i, t in enumerate(type_cats)}
    type_unknown_idx = len(type2idx)
    num_event_types = type_unknown_idx + 1

    df_local["result_name_filled"] = df_local["result_name"].fillna("Unknown")
    result_cats = sorted(df_local["result_name_filled"].unique().tolist())
    if "Unknown" not in result_cats:
        result_cats.append("Unknown")
        result_cats = sorted(result_cats)
    result2idx = {r: i for i, r in enumerate(result_cats)}
    result_unknown_idx = result2idx["Unknown"]
    num_result_types = len(result2idx)

    team_cats = sorted(df_local["team_id"].dropna().unique().tolist())
    team2idx = {t: i for i, t in enumerate(team_cats)}
    team_unknown_idx = len(team2idx)
    num_teams = team_unknown_idx + 1

    player_cats = sorted(df_local["player_id"].dropna().unique().tolist())
    player2idx = {p: i for i, p in enumerate(player_cats)}
    player_unknown_idx = len(player2idx)
    num_players = player_unknown_idx + 1

    return {
        "type2idx": type2idx,
        "type_unknown_idx": type_unknown_idx,
        "num_event_types": num_event_types,
        "result2idx": result2idx,
        "result_unknown_idx": result_unknown_idx,
        "num_result_types": num_result_types,
        "team2idx": team2idx,
        "team_unknown_idx": team_unknown_idx,
        "num_teams": num_teams,
        "player2idx": player2idx,
        "player_unknown_idx": player_unknown_idx,
        "num_players": num_players,
    }


def apply_category_mappings(df: pd.DataFrame, mappings: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    type2idx = mappings["type2idx"]
    type_unknown_idx = mappings["type_unknown_idx"]
    result2idx = mappings["result2idx"]
    result_unknown_idx = mappings["result_unknown_idx"]

    df["type_id"] = df["type_name"].map(type2idx).fillna(type_unknown_idx).astype(int)
    df["result_name_filled"] = df["result_name"].fillna("Unknown")
    df["result_id"] = df["result_name_filled"].map(result2idx).fillna(result_unknown_idx).astype(int)
    return df


# ============================
# Global seq feature normalization (mixed orientation)
# ============================
def compute_seq_feature_norm_stats_mixed(train_raw_df: pd.DataFrame):
    """
    Compute mean/std for SEQ_BASE_COLS using a mixed-orientation dataset:
      - original features
      - fully 180° rotated features
    """
    df_orig = add_event_features_all(train_raw_df)

    df_rot = train_raw_df.copy()
    df_rot["start_x"], df_rot["start_y"] = rotate_180_xy(
        df_rot["start_x"].astype(float).values,
        df_rot["start_y"].astype(float).values,
    )
    ex = df_rot["end_x"].astype(float).values
    ey = df_rot["end_y"].astype(float).values
    ex2, ey2 = rotate_180_xy(np.nan_to_num(ex, nan=0.0), np.nan_to_num(ey, nan=0.0))
    nan_mask = np.isnan(ex) | np.isnan(ey)
    ex2[nan_mask] = np.nan
    ey2[nan_mask] = np.nan
    df_rot["end_x"] = ex2
    df_rot["end_y"] = ey2

    df_rot = add_event_features_all(df_rot)

    df_mix = pd.concat([df_orig, df_rot], axis=0, ignore_index=True)
    means = df_mix[SEQ_BASE_COLS].mean()
    stds = df_mix[SEQ_BASE_COLS].std().replace(0.0, 1.0)
    return means, stds


def normalize_seq_block(g: pd.DataFrame, means, stds) -> np.ndarray:
    X = g[SEQ_BASE_COLS].astype(np.float32).values
    mu = means[SEQ_BASE_COLS].astype(np.float32).values
    sd = stds[SEQ_BASE_COLS].astype(np.float32).values
    Xn = (X - mu) / sd
    return Xn.astype(np.float32)


# ============================
# Player role clusters (fold-wise)
# ============================
def compute_player_role_clusters(df: pd.DataFrame, n_clusters: int = 5, seed: int = 42):
    df_local = df.dropna(subset=["player_id"]).copy()
    grp = df_local.groupby("player_id")
    player_stats = grp.agg(
        mean_start_x=("start_x", "mean"),
        mean_start_y=("start_y", "mean"),
        mean_dist_goal_start=("dist_goal_start", "mean"),
        mean_pass_dist=("pass_dist", "mean"),
        box_ratio=("in_box_start", "mean"),
        mean_speed=("speed", "mean"),
        forward_ratio=("pass_dx", lambda x: (x > 0).mean()),
        n_events=("action_id", "count"),
    ).reset_index()

    role_cols = [
        "mean_start_x",
        "mean_start_y",
        "mean_dist_goal_start",
        "mean_pass_dist",
        "box_ratio",
        "mean_speed",
        "forward_ratio",
    ]

    X = player_stats[role_cols].astype(np.float64).fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(X_scaled.astype(np.float64))

    player_stats["role_cluster"] = kmeans.predict(X_scaled.astype(np.float64))
    player_role_map = player_stats.set_index("player_id")["role_cluster"].to_dict()

    print(f"[INFO] Player role clusters: {n_clusters} (players={len(player_stats)})")
    return player_stats, scaler, kmeans, role_cols, player_role_map


# ============================
# Team style clusters (fold-wise)
# ============================
def compute_team_style_clusters(df: pd.DataFrame, n_clusters: int = 4, seed: int = 42):
    df_local = df.dropna(subset=["team_id"]).copy()
    grp = df_local.groupby("team_id")
    team_stats = grp.agg(
        mean_start_x=("start_x", "mean"),
        mean_start_y=("start_y", "mean"),
        mean_dist_goal_start=("dist_goal_start", "mean"),
        mean_pass_dist=("pass_dist", "mean"),
        box_ratio=("in_box_start", "mean"),
        mean_speed=("speed", "mean"),
        forward_ratio=("pass_dx", lambda x: (x > 0).mean()),
        n_events=("action_id", "count"),
    ).reset_index()

    style_cols = [
        "mean_start_x",
        "mean_start_y",
        "mean_dist_goal_start",
        "mean_pass_dist",
        "box_ratio",
        "mean_speed",
        "forward_ratio",
    ]

    X = team_stats[style_cols].astype(np.float64).fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(X_scaled.astype(np.float64))

    team_stats["style_cluster"] = kmeans.predict(X_scaled.astype(np.float64))
    team_style_map = team_stats.set_index("team_id")["style_cluster"].to_dict()

    print(f"[INFO] Team style clusters: {n_clusters} (teams={len(team_stats)})")
    return team_stats, scaler, kmeans, style_cols, team_style_map


# ============================
# Possession split helper
# ============================
def last_possession_start_index(team_ids: np.ndarray, ref_team_id, t: int) -> int:
    if t <= 0:
        return 0
    prev = team_ids[: t + 1]
    idx_opp = np.where(prev != ref_team_id)[0]
    if len(idx_opp) == 0:
        return 0
    return int(idx_opp[-1] + 1)


# ============================
# Heatmap builder
# ============================
def build_episode_heatmap_prefix(
    g_feat: pd.DataFrame,
    ref_team_id,
    target_idx: int,
    cfg: CFG,
    pos_start_idx: Optional[int] = None,
) -> np.ndarray:
    gx = cfg.heatmap_grid_x
    gy = cfg.heatmap_grid_y
    cell_w = PITCH_X / gx
    cell_h = PITCH_Y / gy

    hm = np.zeros((cfg.heatmap_channels, gy, gx), dtype=np.float32)

    def _cell(x: float, y: float) -> Tuple[int, int]:
        ix = int(x // cell_w)
        iy = int(y // cell_h)
        ix = max(0, min(gx - 1, ix))
        iy = max(0, min(gy - 1, iy))
        return ix, iy

    if target_idx > 0:
        for i, row in g_feat.iloc[:target_idx].iterrows():
            w = cfg.heatmap_weight_possession
            if pos_start_idx is not None and i < pos_start_idx:
                w = cfg.heatmap_weight_pre_possession

            sx, sy = float(row["start_x"]), float(row["start_y"])
            ex, ey = float(row["end_x"]), float(row["end_y"])
            dx = float(row["pass_dx"])
            dy = float(row["pass_dy"])

            is_att = (row["team_id"] == ref_team_id)
            ch_start = 0 if is_att else 2
            ch_end = 1 if is_att else 3
            ch_dx = 4 if is_att else 6
            ch_dy = 5 if is_att else 7

            if not np.isnan(sx) and not np.isnan(sy):
                ix, iy = _cell(sx, sy)
                hm[ch_start, iy, ix] += w
                hm[ch_dx, iy, ix] += w * (dx / PITCH_X)
                hm[ch_dy, iy, ix] += w * (dy / PITCH_Y)

            if not np.isnan(ex) and not np.isnan(ey):
                ix, iy = _cell(ex, ey)
                hm[ch_end, iy, ix] += w

    row_t = g_feat.iloc[target_idx]
    sx, sy = float(row_t["start_x"]), float(row_t["start_y"])
    is_att = (row_t["team_id"] == ref_team_id)
    ch_start = 0 if is_att else 2
    if not np.isnan(sx) and not np.isnan(sy):
        ix, iy = _cell(sx, sy)
        hm[ch_start, iy, ix] += cfg.heatmap_weight_possession

    n_events = max(1, target_idx + 1)
    hm /= float(n_events)
    return hm.astype(np.float32)


# ============================
# Dense train samples
# ============================
def build_dense_train_samples(
    df: pd.DataFrame,
    mappings: Dict[str, Any],
    cfg: CFG,
    ep_stats: pd.DataFrame,
    stats_scaler: StandardScaler,
    stats_cols: List[str],
    player_role_map: Dict,
    team_style_map: Dict,
    means,
    stds,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    team2idx = mappings["team2idx"]
    team_unknown_idx = mappings["team_unknown_idx"]
    player2idx = mappings["player2idx"]
    player_unknown_idx = mappings["player_unknown_idx"]

    ep_stats_sorted = ep_stats.sort_values("game_episode").reset_index(drop=True)
    ep_vals = ep_stats_sorted[stats_cols].astype(np.float64)
    ep_scaled = stats_scaler.transform(ep_vals)
    ep_feat_map = {str(row["game_episode"]): ep_scaled[i].astype("float32") for i, row in ep_stats_sorted.iterrows()}

    default_role = 0
    default_team_style = 0

    masked_cols = ["pass_dx", "pass_dy", "pass_dist", "speed", "angle", "angle_diff"]
    col_to_j = {col: j for j, col in enumerate(SEQ_BASE_COLS)}
    masked_js = [col_to_j[c] for c in masked_cols]

    for game_episode, g_raw in df.groupby("game_episode"):
        g_raw = g_raw.sort_values("time_seconds").reset_index(drop=True)
        ep_len = len(g_raw)
        if ep_len < cfg.min_episode_len:
            continue

        team_ids_in_ep = g_raw["team_id"].dropna().unique().tolist()
        aligned_versions: Dict[int, pd.DataFrame] = {}
        for ref_team in team_ids_in_ep:
            g_aligned = align_episode_to_ref_team(g_raw, int(ref_team), cfg)
            g_feat = compute_event_features_single_episode(g_aligned)
            aligned_versions[int(ref_team)] = g_feat

        game_episode_key = str(game_episode)
        ep_vec = ep_feat_map.get(game_episode_key, np.zeros(len(stats_cols), dtype="float32"))

        for t in range(ep_len):
            row_t_raw = g_raw.iloc[t]
            ref_team = row_t_raw["team_id"]
            if pd.isna(ref_team):
                continue
            ref_team = int(ref_team)

            g_feat = aligned_versions.get(ref_team)
            if g_feat is None:
                g_feat = compute_event_features_single_episode(align_episode_to_ref_team(g_raw, ref_team, cfg))

            row_t = g_feat.iloc[t]
            start_x = float(row_t["start_x"])
            start_y = float(row_t["start_y"])
            end_x = float(row_t["end_x"])
            end_y = float(row_t["end_y"])
            if np.isnan(end_x) or np.isnan(end_y):
                continue

            dx = end_x - start_x
            dy = end_y - start_y
            target_offset = np.array([dx / PITCH_X, dy / PITCH_Y], dtype="float32")
            target_end = np.array([end_x, end_y], dtype="float32")
            last_start_xy = np.array([start_x, start_y], dtype="float32")
            zone_id = compute_zone_id(end_x, end_y, cfg)
            fine_id, residual_tgt = compute_fine_id_and_residual(end_x, end_y, cfg)

            team_seq = g_feat["team_id"].values
            pos_start = last_possession_start_index(team_seq, ref_team, t)

            g_prefix = g_feat.iloc[: t + 1].copy()
            if cfg.use_last_possession_only:
                g_pos = g_feat.iloc[pos_start : t + 1].copy()
                if (len(g_pos) < cfg.min_episode_len) and cfg.possession_fallback_full_prefix:
                    g_seq = g_prefix
                else:
                    g_seq = g_pos
            else:
                g_seq = g_prefix

            if cfg.max_seq_len is not None and len(g_seq) > cfg.max_seq_len:
                g_seq = g_seq.iloc[-cfg.max_seq_len :].copy()

            seq_feats = normalize_seq_block(g_seq, means, stds)  # [T,F]

            # mask last timestep movement features
            last_idx = seq_feats.shape[0] - 1
            for j, col in zip(masked_js, masked_cols):
                mu = float(means[col])
                sigma = float(stds[col]) if float(stds[col]) != 0.0 else 1.0
                seq_feats[last_idx, j] = (0.0 - mu) / sigma

            type_ids = g_seq["type_id"].values.astype("int64")
            result_ids = g_seq["result_id"].values.astype("int64")
            att_side_ids = (g_seq["team_id"].values.astype("int64") != ref_team).astype("int64")

            # opponent team
            episode_team_ids = g_raw["team_id"].dropna().unique().tolist()
            opp_team_id = None
            for tid in episode_team_ids:
                if int(tid) != ref_team:
                    opp_team_id = int(tid)
                    break

            team_id_enc = team2idx.get(ref_team, team_unknown_idx)
            opp_team_id_enc = team2idx.get(opp_team_id, team_unknown_idx) if opp_team_id is not None else team_unknown_idx

            team_style_id = int(team_style_map.get(ref_team, default_team_style))
            opp_team_style_id = int(team_style_map.get(opp_team_id, default_team_style)) if opp_team_id is not None else default_team_style

            last_player_id = row_t_raw["player_id"]
            player_id_enc = player2idx.get(last_player_id, player_unknown_idx)
            player_role_id = int(player_role_map.get(last_player_id, default_role))

            heatmap = build_episode_heatmap_prefix(g_feat, ref_team, t, cfg, pos_start_idx=pos_start)

            is_last = int(t == ep_len - 1)
            type_name = str(row_t_raw.get("type_name", ""))
            is_pass = int(type_name.lower().startswith("pass"))

            if cfg.train_only_pass_targets and (is_pass == 0):
                continue

            w = cfg.pass_event_weight if is_pass else cfg.non_pass_event_weight
            if is_last:
                w *= cfg.last_event_weight_multiplier

            samples.append(
                {
                    "game_id": int(row_t_raw["game_id"]),
                    "game_episode": game_episode_key,
                    "seq_feats": seq_feats,
                    "type_ids": type_ids,
                    "result_ids": result_ids,
                    "att_side_ids": att_side_ids,
                    "cluster_id": int(g_raw["cluster"].iloc[0]),
                    "team_id_enc": int(team_id_enc),
                    "opp_team_id_enc": int(opp_team_id_enc),
                    "team_style_id": int(team_style_id),
                    "opp_team_style_id": int(opp_team_style_id),
                    "player_id_enc": int(player_id_enc),
                    "player_role_id": int(player_role_id),
                    "ep_stats": ep_vec,
                    "last_start_xy": last_start_xy,
                    "target_offset": target_offset,
                    "target_end": target_end,
                    "zone_id": int(zone_id),
                    "fine_id": int(fine_id),
                    "residual_tgt": residual_tgt,
                    "heatmap": heatmap,
                    "is_last": is_last,
                    "is_pass": is_pass,
                    "weight": float(w),
                }
            )

    print(f"[INFO] Built {len(samples)} dense samples.")
    return samples


# ============================
# Mirror augmentation (Y-axis)
# ============================
def augment_samples_mirror_y(samples: List[Dict[str, Any]], means, stds, cfg: CFG) -> List[Dict[str, Any]]:
    col_to_idx = {col: i for i, col in enumerate(SEQ_BASE_COLS)}
    aug_samples: List[Dict[str, Any]] = []

    HM_ATT_DY = 5
    HM_DEF_DY = 7

    for s in samples:
        seq_feats = np.array(s["seq_feats"], dtype=np.float32).copy()

        # unnormalize
        raw: Dict[str, np.ndarray] = {}
        for col in SEQ_BASE_COLS:
            j = col_to_idx[col]
            mu = float(means[col])
            sd = float(stds[col]) if float(stds[col]) != 0.0 else 1.0
            raw[col] = seq_feats[:, j] * sd + mu

        # mirror y-related raw features
        raw["start_y"] = PITCH_Y - raw["start_y"]
        raw["pass_dy"] = -raw["pass_dy"]
        raw["angle"] = -raw["angle"]
        raw["angle_diff"] = -raw["angle_diff"]
        raw["angle_to_goal"] = -raw["angle_to_goal"]

        # renormalize
        new_seq = np.zeros_like(seq_feats, dtype=np.float32)
        for col in SEQ_BASE_COLS:
            j = col_to_idx[col]
            mu = float(means[col])
            sd = float(stds[col]) if float(stds[col]) != 0.0 else 1.0
            new_seq[:, j] = ((raw[col] - mu) / sd).astype(np.float32)

        last_start_xy = np.array(s["last_start_xy"], dtype=np.float32).copy()
        last_start_xy[1] = PITCH_Y - last_start_xy[1]

        target_end = np.array(s["target_end"], dtype=np.float32).copy()
        target_end[1] = PITCH_Y - target_end[1]

        target_offset = np.array(s["target_offset"], dtype=np.float32).copy()
        target_offset[1] = -target_offset[1]

        zone_id = compute_zone_id(float(target_end[0]), float(target_end[1]), cfg)
        fine_id, residual_tgt = compute_fine_id_and_residual(float(target_end[0]), float(target_end[1]), cfg)

        heatmap = np.array(s["heatmap"], dtype=np.float32)
        heatmap_flipped = heatmap[:, ::-1, :].copy()
        heatmap_flipped[HM_ATT_DY] *= -1.0
        heatmap_flipped[HM_DEF_DY] *= -1.0

        new_s = {k: v for k, v in s.items() if k not in [
            "seq_feats","last_start_xy","target_end","target_offset","zone_id","fine_id","residual_tgt","heatmap"
        ]}
        new_s.update({
            "seq_feats": new_seq,
            "last_start_xy": last_start_xy,
            "target_offset": target_offset,
            "target_end": target_end,
            "zone_id": int(zone_id),
            "fine_id": int(fine_id),
            "residual_tgt": residual_tgt,
            "heatmap": heatmap_flipped,
        })
        aug_samples.append(new_s)

    print(f"[INFO] Mirror augmentation: {len(aug_samples)} new samples created.")
    return aug_samples


# ============================
# Mirror inference pack (Y-axis) for TTA
# ============================
def mirror_inference_pack(
    seq_feats: np.ndarray,
    last_start_xy: np.ndarray,
    heatmap: np.ndarray,
    means,
    stds,
    cfg: CFG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    col_to_idx = {col: i for i, col in enumerate(SEQ_BASE_COLS)}
    seq_feats = np.array(seq_feats, dtype=np.float32).copy()

    # unnormalize
    raw: Dict[str, np.ndarray] = {}
    for col in SEQ_BASE_COLS:
        j = col_to_idx[col]
        mu = float(means[col])
        sd = float(stds[col]) if float(stds[col]) != 0.0 else 1.0
        raw[col] = seq_feats[:, j] * sd + mu

    # mirror y-related raw features
    raw["start_y"] = PITCH_Y - raw["start_y"]
    raw["pass_dy"] = -raw["pass_dy"]
    raw["angle"] = -raw["angle"]
    raw["angle_diff"] = -raw["angle_diff"]
    raw["angle_to_goal"] = -raw["angle_to_goal"]

    # renormalize
    new_seq = np.zeros_like(seq_feats, dtype=np.float32)
    for col in SEQ_BASE_COLS:
        j = col_to_idx[col]
        mu = float(means[col])
        sd = float(stds[col]) if float(stds[col]) != 0.0 else 1.0
        new_seq[:, j] = ((raw[col] - mu) / sd).astype(np.float32)

    last_start_xy = np.array(last_start_xy, dtype=np.float32).copy()
    last_start_xy[1] = PITCH_Y - last_start_xy[1]

    heatmap = np.array(heatmap, dtype=np.float32)
    heatmap_flipped = heatmap[:, ::-1, :].copy()

    HM_ATT_DY = 5
    HM_DEF_DY = 7
    heatmap_flipped[HM_ATT_DY] *= -1.0
    heatmap_flipped[HM_DEF_DY] *= -1.0

    return new_seq, last_start_xy, heatmap_flipped


# ============================
# Dataset & DataLoader
# ============================
class EpisodeDenseDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.tensor(s["seq_feats"], dtype=torch.float32),
            torch.tensor(s["type_ids"], dtype=torch.long),
            torch.tensor(s["result_ids"], dtype=torch.long),
            torch.tensor(s["att_side_ids"], dtype=torch.long),
            torch.tensor(s["cluster_id"], dtype=torch.long),
            torch.tensor(s["team_id_enc"], dtype=torch.long),
            torch.tensor(s["opp_team_id_enc"], dtype=torch.long),
            torch.tensor(s["team_style_id"], dtype=torch.long),
            torch.tensor(s["opp_team_style_id"], dtype=torch.long),
            torch.tensor(s["player_id_enc"], dtype=torch.long),
            torch.tensor(s["player_role_id"], dtype=torch.long),
            torch.tensor(s["ep_stats"], dtype=torch.float32),
            torch.tensor(s["last_start_xy"], dtype=torch.float32),
            torch.tensor(s["heatmap"], dtype=torch.float32),
            torch.tensor(bool(s["is_last"]), dtype=torch.bool),
            torch.tensor(s["zone_id"], dtype=torch.long),
            torch.tensor(s["fine_id"], dtype=torch.long),
            torch.tensor(s["residual_tgt"], dtype=torch.float32),
            torch.tensor(s["target_offset"], dtype=torch.float32),
            torch.tensor(s["target_end"], dtype=torch.float32),
            torch.tensor(float(s["weight"]), dtype=torch.float32),
            torch.tensor(int(s["is_pass"]), dtype=torch.long),
        )


def collate_dense_fn(batch):
    (
        seqs,
        type_ids,
        result_ids,
        att_side_ids,
        cluster_ids,
        team_ids,
        opp_team_ids,
        team_style_ids,
        opp_team_style_ids,
        player_ids,
        player_role_ids,
        ep_stats,
        last_start_xy,
        heatmaps,
        last_flags,
        zone_ids,
        fine_ids,
        residual_tgts,
        target_offsets,
        target_ends,
        weights,
        is_pass_flags,
    ) = zip(*batch)

    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)

    seqs_padded = pad_sequence(seqs, batch_first=True)          # [B,T,F]
    type_ids_padded = pad_sequence(type_ids, batch_first=True)  # [B,T]
    result_ids_padded = pad_sequence(result_ids, batch_first=True)
    att_side_ids_padded = pad_sequence(att_side_ids, batch_first=True)

    return (
        seqs_padded,
        type_ids_padded,
        result_ids_padded,
        att_side_ids_padded,
        torch.stack(cluster_ids, dim=0),
        torch.stack(team_ids, dim=0),
        torch.stack(opp_team_ids, dim=0),
        torch.stack(team_style_ids, dim=0),
        torch.stack(opp_team_style_ids, dim=0),
        torch.stack(player_ids, dim=0),
        torch.stack(player_role_ids, dim=0),
        torch.stack(ep_stats, dim=0),
        torch.stack(last_start_xy, dim=0),
        torch.stack(heatmaps, dim=0),
        torch.stack(last_flags, dim=0),
        torch.stack(zone_ids, dim=0),
        torch.stack(fine_ids, dim=0),
        torch.stack(residual_tgts, dim=0),
        lengths,
        torch.stack(target_offsets, dim=0),
        torch.stack(target_ends, dim=0),
        torch.stack(weights, dim=0),
        torch.stack(is_pass_flags, dim=0),
    )


# ============================
# Label smoothing CE helper
# ============================
def cross_entropy_with_smoothing(logits: torch.Tensor, targets: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0.0:
        return F.cross_entropy(logits, targets, reduction="none")
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    smooth = -log_probs.mean(dim=-1)
    return (1.0 - eps) * nll + eps * smooth


# ============================
# Spatial soft-label CE for fine-grid
# ============================
def spatial_soft_ce_loss_vec(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gx: int,
    gy: int,
    kernel: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    if kernel <= 1:
        return F.cross_entropy(logits, targets, reduction="none")
    if kernel % 2 == 0:
        kernel += 1
    r = kernel // 2
    device = logits.device
    B = targets.size(0)

    log_probs = F.log_softmax(logits.float(), dim=-1)  # [B,C]

    ix0 = (targets % gx).to(torch.long)
    iy0 = (targets // gx).to(torch.long)

    dx = torch.arange(-r, r + 1, device=device, dtype=torch.long)
    dy = torch.arange(-r, r + 1, device=device, dtype=torch.long)
    dyy, dxx = torch.meshgrid(dy, dx, indexing="ij")
    dxx = dxx.reshape(-1)
    dyy = dyy.reshape(-1)
    K2 = dxx.numel()

    nix = ix0.unsqueeze(1) + dxx.unsqueeze(0)
    niy = iy0.unsqueeze(1) + dyy.unsqueeze(0)
    valid = (nix >= 0) & (nix < gx) & (niy >= 0) & (niy < gy)

    nix = nix.clamp(0, gx - 1)
    niy = niy.clamp(0, gy - 1)
    nids = niy * gx + nix

    d2 = (dxx.to(torch.float32) ** 2 + dyy.to(torch.float32) ** 2)
    w_base = torch.exp(-0.5 * d2 / max(float(sigma) ** 2, 1e-6))
    w = w_base.unsqueeze(0).expand(B, K2) * valid.to(torch.float32)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)

    lp = log_probs.gather(dim=1, index=nids)
    loss = -(lp * w).sum(dim=1)
    return loss


# ============================
# EMA helper
# ============================
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.995, warmup_steps: int = 200):
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.updates = 0

        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _get_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        return self.decay * (1.0 - math.exp(-self.updates / float(self.warmup_steps)))

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = self._get_decay()
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            v = esd[k]
            mv = msd[k]
            if torch.is_floating_point(v):
                v.mul_(d).add_(mv.detach(), alpha=1.0 - d)
            else:
                v.copy_(mv)


# ============================
# Model
# ============================
class PassLSTMDenseHeatmapFineResidualMTL(nn.Module):
    def __init__(
        self,
        num_event_types: int,
        num_result_types: int,
        num_clusters: int,
        num_teams: int,
        num_players: int,
        num_player_roles: int,
        num_team_styles: int,
        num_zones: int,
        num_fine_cells: int,
        in_features: int,
        ep_stats_dim: int,
        heatmap_channels: int,
        cfg: CFG,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_fine_cells = num_fine_cells

        self.bidirectional = cfg.bidirectional
        self.num_directions = 2 if cfg.bidirectional else 1
        self.final_window_size = cfg.final_window_size

        self.type_emb = nn.Embedding(num_event_types, cfg.type_emb_dim)
        self.result_emb = nn.Embedding(num_result_types, cfg.result_emb_dim)
        self.att_side_emb = nn.Embedding(2, cfg.att_side_emb_dim)
        self.cluster_emb = nn.Embedding(num_clusters, cfg.cluster_emb_dim)
        self.team_emb = nn.Embedding(num_teams, cfg.team_emb_dim)
        self.player_emb = nn.Embedding(num_players, cfg.player_emb_dim)
        self.player_role_emb = nn.Embedding(num_player_roles, cfg.player_role_emb_dim)
        self.team_style_emb = nn.Embedding(num_team_styles, cfg.team_style_emb_dim)

        lstm_input_dim = in_features + cfg.type_emb_dim + cfg.result_emb_dim + cfg.att_side_emb_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.lstm_out_dim = cfg.hidden_dim * self.num_directions

        self.attn = nn.Linear(self.lstm_out_dim, cfg.attn_dim)
        self.attn_v = nn.Linear(cfg.attn_dim, 1)

        ep_hidden_dim = 32
        self.ep_mlp = nn.Sequential(
            nn.LayerNorm(ep_stats_dim),
            nn.Linear(ep_stats_dim, ep_hidden_dim),
            nn.ReLU(),
        )

        last_start_hidden_dim = 16
        self.last_start_mlp = nn.Sequential(
            nn.Linear(2, last_start_hidden_dim),
            nn.ReLU(),
        )

        ctx_dim = (
            cfg.cluster_emb_dim
            + cfg.team_emb_dim * 2
            + cfg.team_style_emb_dim * 2
            + cfg.player_emb_dim
            + cfg.player_role_emb_dim
        )

        gate_in_dim = (
            cfg.cluster_emb_dim
            + cfg.team_emb_dim * 2
            + cfg.team_style_emb_dim * 2
            + cfg.player_role_emb_dim
        )
        self.ctx_gate = nn.Linear(gate_in_dim, self.lstm_out_dim)

        self.hm_cnn = nn.Sequential(
            nn.Conv2d(heatmap_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        heatmap_vec_dim = 32

        head_in_dim = (
            self.lstm_out_dim  # context_global
            + self.lstm_out_dim  # context_fw
            + self.lstm_out_dim  # last_hidden
            + ctx_dim
            + ep_hidden_dim
            + last_start_hidden_dim
            + heatmap_vec_dim
        )
        self.head_in_dim = head_in_dim

        self.fc_offset = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Linear(head_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 2),
        )

        self.fc_zone = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Linear(head_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, num_zones),
        )

        self.fc_fine = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Linear(head_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, num_fine_cells),
        )

        self.fine_cell_emb = nn.Embedding(num_fine_cells, cfg.fine_cell_emb_dim)
        self.fc_residual = nn.Sequential(
            nn.LayerNorm(head_in_dim + cfg.fine_cell_emb_dim),
            nn.Linear(head_in_dim + cfg.fine_cell_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 2),
        )

        self.fc_mix = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Linear(head_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 1),
        )

    def _encode(
        self,
        seq_feats,
        type_ids,
        result_ids,
        att_side_ids,
        cluster_ids,
        team_ids,
        opp_team_ids,
        team_style_ids,
        opp_team_style_ids,
        player_ids,
        player_role_ids,
        ep_stats,
        last_start_xy,
        heatmaps,
        lengths,
    ) -> torch.Tensor:
        B, T, _ = seq_feats.size()
        device = seq_feats.device

        if seq_feats.is_cuda:
            try:
                self.lstm.flatten_parameters()
            except Exception:
                pass

        type_emb = self.type_emb(type_ids)
        result_emb = self.result_emb(result_ids)
        att_emb = self.att_side_emb(att_side_ids)
        lstm_input = torch.cat([seq_feats, type_emb, result_emb, att_emb], dim=-1)

        packed = pack_padded_sequence(
            lstm_input,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)

        mask_pad = torch.arange(T, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

        cluster_emb = self.cluster_emb(cluster_ids)
        team_emb_att = self.team_emb(team_ids)
        team_emb_opp = self.team_emb(opp_team_ids)
        team_style_emb_att = self.team_style_emb(team_style_ids)
        team_style_emb_opp = self.team_style_emb(opp_team_style_ids)
        player_emb = self.player_emb(player_ids)
        player_role_emb = self.player_role_emb(player_role_ids)

        gate_input = torch.cat(
            [cluster_emb, team_emb_att, team_emb_opp, team_style_emb_att, team_style_emb_opp, player_role_emb],
            dim=-1,
        )
        gate = torch.sigmoid(self.ctx_gate(gate_input))
        lstm_out = lstm_out * gate.unsqueeze(1)

        attn_h = torch.tanh(self.attn(lstm_out))
        attn_logits = self.attn_v(attn_h).squeeze(-1)
        attn_logits = attn_logits.float().masked_fill(mask_pad, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=-1).to(lstm_out.dtype)

        context_global = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)

        fw_size = max(1, min(self.final_window_size, T))
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        valid_mask = ~mask_pad
        fw_start = (lengths - fw_size).clamp(min=0).unsqueeze(1).expand(B, T)
        mask_fw = valid_mask & (positions >= fw_start)
        fw_len = mask_fw.sum(dim=1).clamp(min=1).unsqueeze(-1).to(lstm_out.dtype)
        fw_sum = torch.sum(lstm_out * mask_fw.unsqueeze(-1), dim=1)
        context_fw = fw_sum / fw_len

        last_indices = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(B, device=device)
        last_hidden = lstm_out[batch_idx, last_indices]

        ctx = torch.cat(
            [cluster_emb, team_emb_att, team_emb_opp, team_style_emb_att, team_style_emb_opp, player_emb, player_role_emb],
            dim=-1,
        )

        ep_vec = self.ep_mlp(ep_stats)

        last_start_norm = torch.empty_like(last_start_xy)
        last_start_norm[:, 0] = last_start_xy[:, 0] / PITCH_X
        last_start_norm[:, 1] = last_start_xy[:, 1] / PITCH_Y
        last_start_vec = self.last_start_mlp(last_start_norm)

        hm_feat = self.hm_cnn(heatmaps).view(B, -1)

        h = torch.cat([context_global, context_fw, last_hidden, ctx, ep_vec, last_start_vec, hm_feat], dim=-1)
        return h

    def residual_from_h(self, h: torch.Tensor, fine_cell_ids: torch.Tensor) -> torch.Tensor:
        fine_emb = self.fine_cell_emb(fine_cell_ids)
        res_in = torch.cat([h, fine_emb], dim=-1)
        residual_raw = self.fc_residual(res_in)
        residual = torch.tanh(residual_raw) * self.cfg.residual_max_norm
        return residual

    def forward(
        self,
        seq_feats,
        type_ids,
        result_ids,
        att_side_ids,
        cluster_ids,
        team_ids,
        opp_team_ids,
        team_style_ids,
        opp_team_style_ids,
        player_ids,
        player_role_ids,
        ep_stats,
        last_start_xy,
        heatmaps,
        lengths,
    ):
        h = self._encode(
            seq_feats, type_ids, result_ids, att_side_ids, cluster_ids, team_ids, opp_team_ids,
            team_style_ids, opp_team_style_ids, player_ids, player_role_ids,
            ep_stats, last_start_xy, heatmaps, lengths,
        )
        offsets = self.fc_offset(h)
        zone_logits = self.fc_zone(h)
        fine_logits = self.fc_fine(h)
        mix_gate = torch.sigmoid(self.fc_mix(h)).squeeze(-1)
        return offsets, zone_logits, fine_logits, mix_gate, h


# ============================
# End-coordinate reconstruction
# ============================
def end_from_coarse_offsets(offsets_norm: torch.Tensor, last_start_xy: torch.Tensor) -> torch.Tensor:
    dx_m = offsets_norm[:, 0] * PITCH_X
    dy_m = offsets_norm[:, 1] * PITCH_Y
    pred_x = last_start_xy[:, 0] + dx_m
    pred_y = last_start_xy[:, 1] + dy_m
    return torch.stack([pred_x, pred_y], dim=-1)


def end_from_fine_and_residual(fine_ids: torch.Tensor, residual: torch.Tensor, cfg: CFG) -> torch.Tensor:
    cx, cy, cell_w, cell_h = batch_grid_centers(fine_ids, cfg.fine_grid_x, cfg.fine_grid_y)
    pred_x = cx.to(residual.device) + residual[:, 0] * cell_w
    pred_y = cy.to(residual.device) + residual[:, 1] * cell_h
    return torch.stack([pred_x, pred_y], dim=-1)


def mix_end_pred(end_coarse: torch.Tensor, end_fine: torch.Tensor, mix_gate: torch.Tensor, cfg: CFG) -> torch.Tensor:
    if not cfg.use_mix_gate:
        return end_fine
    a = mix_gate.unsqueeze(-1)
    return a * end_coarse + (1.0 - a) * end_fine


def euclidean_distance(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    dx = pred_xy[:, 0] - target_xy[:, 0]
    dy = pred_xy[:, 1] - target_xy[:, 1]
    return torch.sqrt(dx * dx + dy * dy + 1e-9)


def weighted_mean(loss_vec: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.clamp(min=1e-6)
    return (loss_vec * weights).sum() / weights.sum()


# ============================
# Inference-style end prediction (TOP-K expected)
# ============================
@torch.no_grad()
def predict_end_topk_expected_eval(
    model: PassLSTMDenseHeatmapFineResidualMTL,
    offsets: torch.Tensor,
    fine_logits: torch.Tensor,
    mix_gate: torch.Tensor,
    h: torch.Tensor,
    last_start_xy: torch.Tensor,
    cfg: CFG,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax((fine_logits / max(cfg.fine_temperature, 1e-6)).float(), dim=-1)
    pmax, top1 = probs.max(dim=-1)

    K = min(cfg.fine_topk, probs.size(-1))
    topk_probs, topk_ids = probs.topk(K, dim=-1)
    w = topk_probs / (topk_probs.sum(dim=1, keepdim=True).clamp(min=1e-6))

    B = fine_logits.size(0)
    h_rep = h.unsqueeze(1).expand(B, K, h.size(-1)).reshape(B * K, -1)
    ids_flat = topk_ids.reshape(-1)
    residual_flat = model.residual_from_h(h_rep, ids_flat)
    residual = residual_flat.view(B, K, 2)

    cx, cy, cell_w, cell_h = batch_grid_centers(ids_flat, cfg.fine_grid_x, cfg.fine_grid_y)
    cx = cx.to(h.device).view(B, K)
    cy = cy.to(h.device).view(B, K)
    end_fine_k = torch.stack(
        [cx + residual[:, :, 0] * cell_w, cy + residual[:, :, 1] * cell_h],
        dim=-1,
    )

    end_coarse = end_from_coarse_offsets(offsets, last_start_xy)
    end_mix_k = mix_end_pred(end_coarse.unsqueeze(1), end_fine_k, mix_gate.unsqueeze(1), cfg)
    pred_end = (w.unsqueeze(-1) * end_mix_k).sum(dim=1)
    return pred_end, pmax, top1


def predict_end_topk_expected_train(
    model: PassLSTMDenseHeatmapFineResidualMTL,
    offsets: torch.Tensor,
    fine_logits: torch.Tensor,
    mix_gate: torch.Tensor,
    h: torch.Tensor,
    last_start_xy: torch.Tensor,
    cfg: CFG,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax((fine_logits / max(cfg.fine_temperature, 1e-6)).float(), dim=-1)
    pmax, top1 = probs.max(dim=-1)

    K = min(cfg.fine_topk, probs.size(-1))
    topk_probs, topk_ids = probs.topk(K, dim=-1)
    w = topk_probs / (topk_probs.sum(dim=1, keepdim=True).clamp(min=1e-6))

    B = fine_logits.size(0)
    h_rep = h.unsqueeze(1).expand(B, K, h.size(-1)).reshape(B * K, -1)
    ids_flat = topk_ids.reshape(-1)

    residual_flat = model.residual_from_h(h_rep, ids_flat)
    residual = residual_flat.view(B, K, 2)

    cx, cy, cell_w, cell_h = batch_grid_centers(ids_flat, cfg.fine_grid_x, cfg.fine_grid_y)
    cx = cx.to(h.device).view(B, K)
    cy = cy.to(h.device).view(B, K)
    end_fine_k = torch.stack(
        [cx + residual[:, :, 0] * cell_w, cy + residual[:, :, 1] * cell_h],
        dim=-1,
    )

    end_coarse = end_from_coarse_offsets(offsets, last_start_xy)
    end_mix_k = mix_end_pred(end_coarse.unsqueeze(1), end_fine_k, mix_gate.unsqueeze(1), cfg)
    pred_end = (w.unsqueeze(-1) * end_mix_k).sum(dim=1)
    return pred_end, pmax, top1


# ============================
# AMP helpers
# ============================
def _get_grad_scaler(use_amp: bool, device: str):
    # torch.amp.GradScaler (torch>=2.0) supports device arg.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device='cuda', enabled=bool(use_amp and device == "cuda"))
        except TypeError:
            return torch.amp.GradScaler(enabled=bool(use_amp and device == "cuda"))
    # fallback
    return torch.cuda.amp.GradScaler(enabled=bool(use_amp and device == "cuda"))


# ============================
# Train / Eval
# ============================
def train_one_epoch(
    model,
    loader,
    optimizer,
    device: str,
    cfg: CFG,
    scaler=None,
    stage2: bool = False,
    ema: Optional[ModelEMA] = None,
):
    model.train()

    if stage2:
        mse_w = cfg.finetune_mse_weight
        dist_w = cfg.finetune_dist_weight
        zone_w = cfg.finetune_zone_loss_weight
        fine_w = cfg.finetune_fine_loss_weight
        res_w = cfg.finetune_residual_loss_weight
        gate_w = cfg.finetune_gate_loss_weight
        coarse_w = cfg.finetune_coarse_dist_weight
    else:
        mse_w = cfg.mse_weight
        dist_w = cfg.dist_weight
        zone_w = cfg.zone_loss_weight
        fine_w = cfg.fine_loss_weight
        res_w = cfg.residual_loss_weight
        gate_w = cfg.gate_loss_weight
        coarse_w = cfg.coarse_dist_weight

    total_loss = 0.0
    total_dist_all = 0.0
    total_dist_last = 0.0
    n_all = 0
    n_last = 0

    total_zone_correct = 0
    total_fine_correct = 0
    total_gate_mean = 0.0
    total_pmax_mean = 0.0

    use_amp = bool(cfg.use_amp and (device == "cuda"))
    if scaler is None:
        scaler = _get_grad_scaler(use_amp, device)

    autocast_ctx = torch.amp.autocast if hasattr(torch, "amp") else torch.cuda.amp.autocast  # type: ignore

    for (
        seqs,
        type_ids,
        result_ids,
        att_side_ids,
        cluster_ids,
        team_ids,
        opp_team_ids,
        team_style_ids,
        opp_team_style_ids,
        player_ids,
        player_role_ids,
        ep_stats,
        last_start_xy,
        heatmaps,
        last_flags,
        zone_ids,
        fine_ids,
        residual_tgts,
        lengths,
        target_offsets,
        target_ends,
        weights,
        _is_pass_flags,
    ) in loader:
        seqs = seqs.to(device, non_blocking=True)
        type_ids = type_ids.to(device, non_blocking=True)
        result_ids = result_ids.to(device, non_blocking=True)
        att_side_ids = att_side_ids.to(device, non_blocking=True)
        cluster_ids = cluster_ids.to(device, non_blocking=True)
        team_ids = team_ids.to(device, non_blocking=True)
        opp_team_ids = opp_team_ids.to(device, non_blocking=True)
        team_style_ids = team_style_ids.to(device, non_blocking=True)
        opp_team_style_ids = opp_team_style_ids.to(device, non_blocking=True)
        player_ids = player_ids.to(device, non_blocking=True)
        player_role_ids = player_role_ids.to(device, non_blocking=True)
        ep_stats = ep_stats.to(device, non_blocking=True)
        last_start_xy = last_start_xy.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        last_flags = last_flags.to(device, non_blocking=True)
        zone_ids = zone_ids.to(device, non_blocking=True)
        fine_ids = fine_ids.to(device, non_blocking=True)
        residual_tgts = residual_tgts.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        target_offsets = target_offsets.to(device, non_blocking=True)
        target_ends = target_ends.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(device_type='cuda', enabled=use_amp):
            offsets, zone_logits, fine_logits, mix_gate, h = model(
                seqs, type_ids, result_ids, att_side_ids, cluster_ids, team_ids, opp_team_ids,
                team_style_ids, opp_team_style_ids, player_ids, player_role_ids,
                ep_stats, last_start_xy, heatmaps, lengths,
            )

            mse_vec = ((offsets - target_offsets) ** 2).mean(dim=1)
            mse = weighted_mean(mse_vec, weights)

            zone_loss_vec = cross_entropy_with_smoothing(zone_logits, zone_ids, cfg.zone_label_smoothing)
            zone_loss = weighted_mean(zone_loss_vec, weights)

            if cfg.fine_use_spatial_soft:
                fine_loss_vec = spatial_soft_ce_loss_vec(
                    fine_logits, fine_ids, cfg.fine_grid_x, cfg.fine_grid_y,
                    kernel=cfg.fine_soft_kernel, sigma=cfg.fine_soft_sigma,
                )
            else:
                fine_loss_vec = cross_entropy_with_smoothing(fine_logits, fine_ids, cfg.fine_label_smoothing)
            fine_loss = weighted_mean(fine_loss_vec, weights)

            residual_gt = model.residual_from_h(h, fine_ids)
            res_loss_vec = F.smooth_l1_loss(residual_gt, residual_tgts, reduction="none").mean(dim=1)
            res_loss = weighted_mean(res_loss_vec, weights)

            pred_end, pmax, _top1 = predict_end_topk_expected_train(
                model=model,
                offsets=offsets,
                fine_logits=fine_logits,
                mix_gate=mix_gate,
                h=h,
                last_start_xy=last_start_xy,
                cfg=cfg,
            )
            dist_vec = euclidean_distance(pred_end, target_ends)
            dist_m = weighted_mean(dist_vec, weights)
            dist_loss = dist_m / cfg.dist_divisor

            if coarse_w > 0.0:
                end_coarse = end_from_coarse_offsets(offsets, last_start_xy)
                dist_coarse_vec = euclidean_distance(end_coarse, target_ends)
                dist_coarse_m = weighted_mean(dist_coarse_vec, weights)
                coarse_dist_loss = dist_coarse_m / cfg.dist_divisor
            else:
                coarse_dist_loss = torch.tensor(0.0, device=device, dtype=dist_loss.dtype)

            if gate_w > 0.0:
                p_for_gate = pmax.detach() if getattr(cfg, "gate_detach_pmax", True) else pmax
                gate_target = (1.0 - p_for_gate).clamp(0.0, 1.0) ** max(cfg.gate_target_power, 1e-6)
                gate_loss_vec = (mix_gate - gate_target).pow(2)
                gate_loss = weighted_mean(gate_loss_vec, weights)
            else:
                gate_loss = torch.tensor(0.0, device=device, dtype=dist_loss.dtype)

            loss = (
                mse_w * mse
                + dist_w * dist_loss
                + zone_w * zone_loss
                + fine_w * fine_loss
                + res_w * res_loss
                + coarse_w * coarse_dist_loss
                + gate_w * gate_loss
            )

        if use_amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        bs = seqs.size(0)
        total_loss += float(loss.detach().cpu().item()) * bs
        total_dist_all += float(dist_vec.detach().cpu().sum().item())
        n_all += bs

        mask_last = last_flags.bool()
        if mask_last.any():
            total_dist_last += float(dist_vec[mask_last].detach().sum().item())
            n_last += int(mask_last.sum().item())

        total_zone_correct += int((zone_logits.argmax(dim=-1) == zone_ids).sum().item())
        total_fine_correct += int((fine_logits.argmax(dim=-1) == fine_ids).sum().item())
        total_gate_mean += float(mix_gate.detach().mean().item()) * bs
        total_pmax_mean += float(pmax.detach().mean().item()) * bs

    avg_loss = total_loss / max(n_all, 1)
    avg_dist_all = total_dist_all / max(n_all, 1)
    avg_dist_last = total_dist_last / max(n_last, 1) if n_last > 0 else 0.0
    zone_acc = total_zone_correct / max(n_all, 1)
    fine_acc = total_fine_correct / max(n_all, 1)
    gate_mean = total_gate_mean / max(n_all, 1)
    pmax_mean = total_pmax_mean / max(n_all, 1)
    return avg_loss, avg_dist_all, avg_dist_last, zone_acc, fine_acc, gate_mean, pmax_mean


@torch.no_grad()
def eval_one_epoch(model, loader, device: str, cfg: CFG):
    model.eval()

    total_dist_all = 0.0
    total_dist_last = 0.0
    n_all = 0
    n_last = 0
    total_zone_correct = 0
    total_fine_correct = 0
    total_gate_mean = 0.0
    total_pmax_mean = 0.0

    for (
        seqs,
        type_ids,
        result_ids,
        att_side_ids,
        cluster_ids,
        team_ids,
        opp_team_ids,
        team_style_ids,
        opp_team_style_ids,
        player_ids,
        player_role_ids,
        ep_stats,
        last_start_xy,
        heatmaps,
        last_flags,
        zone_ids,
        fine_ids,
        _residual_tgts,
        lengths,
        _target_offsets,
        target_ends,
        _weights,
        _is_pass_flags,
    ) in loader:
        seqs = seqs.to(device, non_blocking=True)
        type_ids = type_ids.to(device, non_blocking=True)
        result_ids = result_ids.to(device, non_blocking=True)
        att_side_ids = att_side_ids.to(device, non_blocking=True)
        cluster_ids = cluster_ids.to(device, non_blocking=True)
        team_ids = team_ids.to(device, non_blocking=True)
        opp_team_ids = opp_team_ids.to(device, non_blocking=True)
        team_style_ids = team_style_ids.to(device, non_blocking=True)
        opp_team_style_ids = opp_team_style_ids.to(device, non_blocking=True)
        player_ids = player_ids.to(device, non_blocking=True)
        player_role_ids = player_role_ids.to(device, non_blocking=True)
        ep_stats = ep_stats.to(device, non_blocking=True)
        last_start_xy = last_start_xy.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        last_flags = last_flags.to(device, non_blocking=True)
        zone_ids = zone_ids.to(device, non_blocking=True)
        fine_ids = fine_ids.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        target_ends = target_ends.to(device, non_blocking=True)

        offsets, zone_logits, fine_logits, mix_gate, h = model(
            seqs, type_ids, result_ids, att_side_ids, cluster_ids, team_ids, opp_team_ids,
            team_style_ids, opp_team_style_ids, player_ids, player_role_ids,
            ep_stats, last_start_xy, heatmaps, lengths,
        )

        pred_end, pmax, _top1 = predict_end_topk_expected_eval(
            model=model,
            offsets=offsets,
            fine_logits=fine_logits,
            mix_gate=mix_gate,
            h=h,
            last_start_xy=last_start_xy,
            cfg=cfg,
        )
        dist_vec = euclidean_distance(pred_end, target_ends)

        bs = seqs.size(0)
        total_dist_all += float(dist_vec.sum().item())
        n_all += bs

        mask_last = last_flags.bool()
        if mask_last.any():
            total_dist_last += float(dist_vec[mask_last].sum().item())
            n_last += int(mask_last.sum().item())

        total_zone_correct += int((zone_logits.argmax(dim=-1) == zone_ids).sum().item())
        total_fine_correct += int((fine_logits.argmax(dim=-1) == fine_ids).sum().item())
        total_gate_mean += float(mix_gate.mean().item()) * bs
        total_pmax_mean += float(pmax.mean().item()) * bs

    avg_dist_all = total_dist_all / max(n_all, 1)
    avg_dist_last = total_dist_last / max(n_last, 1) if n_last > 0 else 0.0
    zone_acc = total_zone_correct / max(n_all, 1)
    fine_acc = total_fine_correct / max(n_all, 1)
    gate_mean = total_gate_mean / max(n_all, 1)
    pmax_mean = total_pmax_mean / max(n_all, 1)
    return avg_dist_all, avg_dist_last, zone_acc, fine_acc, gate_mean, pmax_mean


# ============================
# Fine-tune helper (freeze backbone)
# ============================
def set_backbone_trainable(model: nn.Module, trainable: bool):
    for p in model.parameters():
        p.requires_grad = trainable
    if not trainable:
        head_modules = [
            model.fc_offset,
            model.fc_zone,
            model.fc_fine,
            model.fc_residual,
            model.fc_mix,
            model.fine_cell_emb,
        ]
        for m in head_modules:
            for p in m.parameters():
                p.requires_grad = True


# ============================
# Train 1 fold
# ============================
def train_one_fold(
    fold: int,
    train_samples: List[Dict[str, Any]],
    valid_samples: List[Dict[str, Any]],
    meta: Dict[str, Any],
    mappings: Dict[str, Any],
    means,
    stds,
    device: str,
    cfg: CFG,
):
    print(f"\n[INFO] ===== Fold {fold} =====")
    print(f"Train samples: {len(train_samples)}, Valid samples: {len(valid_samples)}")

    valid_ds = EpisodeDenseDataset(valid_samples)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_dense_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = PassLSTMDenseHeatmapFineResidualMTL(
        num_event_types=mappings["num_event_types"],
        num_result_types=mappings["num_result_types"],
        num_clusters=cfg.ep_n_clusters,
        num_teams=mappings["num_teams"],
        num_players=mappings["num_players"],
        num_player_roles=cfg.player_role_n_clusters,
        num_team_styles=cfg.team_style_n_clusters,
        num_zones=cfg.zone_grid_x * cfg.zone_grid_y,
        num_fine_cells=cfg.fine_grid_x * cfg.fine_grid_y,
        in_features=len(SEQ_BASE_COLS),
        ep_stats_dim=len(meta["stats_cols"]),
        heatmap_channels=cfg.heatmap_channels,
        cfg=cfg,
    ).to(device)

    ema = ModelEMA(model, decay=cfg.ema_decay, warmup_steps=getattr(cfg, "ema_warmup_steps", 0)) if cfg.use_ema else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_last = float("inf")
    best_state = None
    patience_counter = 0

    base_train_samples = train_samples
    scaler = _get_grad_scaler(bool(cfg.use_amp and device == "cuda"), device)

    if cfg.use_mirror_aug:
        aug_train_samples = augment_samples_mirror_y(base_train_samples, means, stds, cfg)
        train_samples_epoch_base = base_train_samples + aug_train_samples
        print(f"[Fold {fold}] Train samples (base+mirror): {len(base_train_samples)} + {len(aug_train_samples)} = {len(train_samples_epoch_base)}")
    else:
        train_samples_epoch_base = base_train_samples

    for epoch in range(1, cfg.num_epochs + 1):
        train_ds = EpisodeDenseDataset(train_samples_epoch_base)
        # Deterministic per-epoch shuffling (decouples DataLoader order from model RNG state)
        _gen = torch.Generator()
        _gen.manual_seed(int(cfg.seed) + 1000 * int(fold) + int(epoch))
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=_gen,
            collate_fn=collate_dense_fn,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        train_loss, train_dist_all, train_dist_last, train_zone_acc, train_fine_acc, train_gate, train_pmax = train_one_epoch(
            model, train_loader, optimizer, device, cfg, scaler=scaler, stage2=False, ema=ema
        )

        val_dist_all_m, val_dist_last_m, val_zone_acc_m, val_fine_acc_m, val_gate_m, val_pmax_m = eval_one_epoch(
            model, valid_loader, device, cfg
        )

        if ema is not None:
            val_dist_all_e, val_dist_last_e, val_zone_acc_e, val_fine_acc_e, val_gate_e, val_pmax_e = eval_one_epoch(
                ema.ema, valid_loader, device, cfg
            )
        else:
            val_dist_last_e = None

        if (ema is not None) and bool(getattr(cfg, "eval_use_ema", False)):
            val_dist_last_sel = val_dist_last_e
        else:
            val_dist_last_sel = val_dist_last_m

        scheduler.step(val_dist_last_sel)

        msg = (
            f"[Fold {fold}] Epoch {epoch}/{cfg.num_epochs} "
            f"| TrainLoss: {train_loss:.5f} "
            f"| TrainDistAll(m): {train_dist_all:.4f} "
            f"| TrainLastDist(m): {train_dist_last:.4f} "
            f"| TrainZoneAcc: {train_zone_acc*100:.2f}% "
            f"| TrainFineAcc: {train_fine_acc*100:.2f}% "
            f"| TrainGate(mean): {train_gate:.3f} "
            f"| TrainPmax(mean): {train_pmax:.3f} "
            f"| Val(model) LastDist(m): {val_dist_last_m:.4f}"
        )
        if ema is not None:
            msg += f" | Val(EMA) LastDist(m): {val_dist_last_e:.4f}"
        print(msg)

        if val_dist_last_sel < best_val_last - 1e-4:
            best_val_last = float(val_dist_last_sel)
            best_state = copy.deepcopy((ema.ema if ema is not None else model).state_dict())
            patience_counter = 0
            print(f"  -> New best model! (ValLastDist={best_val_last:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"  -> Early stopping at epoch {epoch}")
                break

    # Stage-2: last-pass-only fine-tune
    if cfg.do_finetune_last_pass_only and best_state is not None:
        print(f"[Fold {fold}] Starting LAST-PASS-only fine-tuning...")
        model.load_state_dict(best_state)
        ema = ModelEMA(model, decay=cfg.ema_decay, warmup_steps=getattr(cfg, "ema_warmup_steps", 0)) if cfg.use_ema else None

        train_lp = [s for s in base_train_samples if int(s.get("is_last", 0)) == 1 and int(s.get("is_pass", 0)) == 1]
        valid_lp = [s for s in valid_samples if int(s.get("is_last", 0)) == 1 and int(s.get("is_pass", 0)) == 1]
        print(f"[Fold {fold}] Last-pass subset: train={len(train_lp)} valid={len(valid_lp)}")

        if len(train_lp) >= 2000 and len(valid_lp) >= 400:
            if cfg.finetune_use_mirror_aug and cfg.use_mirror_aug:
                train_lp_aug = augment_samples_mirror_y(train_lp, means, stds, cfg)
                train_lp_epoch_base = train_lp + train_lp_aug
                print(f"[Fold {fold}] FT samples (base+mirror): {len(train_lp)} + {len(train_lp_aug)} = {len(train_lp_epoch_base)}")
            else:
                train_lp_epoch_base = train_lp

            ft_train_ds = EpisodeDenseDataset(train_lp_epoch_base)
            # Deterministic per-epoch shuffling for fine-tune stage
            _gen_ft = torch.Generator()
            _gen_ft.manual_seed(int(cfg.seed) + 200000 + 1000 * int(fold))
            ft_train_loader = DataLoader(
                ft_train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                generator=_gen_ft,
                collate_fn=collate_dense_fn,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )

            ft_valid_ds = EpisodeDenseDataset(valid_lp)
            ft_valid_loader = DataLoader(
                ft_valid_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=collate_dense_fn,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )

            if cfg.finetune_freeze_backbone:
                set_backbone_trainable(model, trainable=False)
                print(f"[Fold {fold}] FT mode: heads-only (backbone frozen)")

            ft_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.finetune_lr, weight_decay=cfg.weight_decay)
            ft_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ft_optimizer, mode="min", factor=0.5, patience=1)

            ft_best = best_val_last
            ft_best_state = copy.deepcopy(model.state_dict())
            ft_pat = 0

            for e in range(1, cfg.finetune_epochs + 1):
                _ = train_one_epoch(model, ft_train_loader, ft_optimizer, device, cfg, scaler=scaler, stage2=True, ema=ema)
                _val_all_m, val_last_m, *_ = eval_one_epoch(model, ft_valid_loader, device, cfg)

                if ema is not None:
                    _val_all_e, val_last_e, *_ = eval_one_epoch(ema.ema, ft_valid_loader, device, cfg)
                else:
                    val_last_e = None

                if (ema is not None) and bool(getattr(cfg, "eval_use_ema", False)):
                    val_last_sel = val_last_e
                else:
                    val_last_sel = val_last_m

                ft_scheduler.step(val_last_sel)

                msg_ft = f"[Fold {fold}] Finetune {e}/{cfg.finetune_epochs} | ValLastDist(last-pass, model): {val_last_m:.4f}"
                if ema is not None:
                    msg_ft += f" | ValLastDist(last-pass, EMA): {val_last_e:.4f}"
                print(msg_ft)

                if val_last_sel < ft_best - 1e-4:
                    ft_best = float(val_last_sel)
                    ft_best_state = copy.deepcopy((ema.ema if ema is not None else model).state_dict())
                    ft_pat = 0
                    print("  -> New finetune best!")
                else:
                    ft_pat += 1
                    if ft_pat >= cfg.finetune_patience:
                        print("  -> Finetune early stop")
                        break

            best_val_last = ft_best
            best_state = ft_best_state
        else:
            print(f"[Fold {fold}] Last-pass subset too small -> skip finetune.")

    print(f"[Fold {fold}] Best ValLastDist(m): {best_val_last:.4f}")
    return best_state, float(best_val_last)


# ============================
# Test processing helpers
# ============================
def estimate_player_role_from_ep(
    g_feat: pd.DataFrame,
    player_id,
    player_role_cols: List[str],
    role_scaler: StandardScaler,
    role_kmeans: KMeans,
) -> int:
    g_p = g_feat[g_feat["player_id"] == player_id]
    if len(g_p) == 0:
        return 0
    stats = {
        "mean_start_x": g_p["start_x"].mean(),
        "mean_start_y": g_p["start_y"].mean(),
        "mean_dist_goal_start": g_p["dist_goal_start"].mean(),
        "mean_pass_dist": g_p["pass_dist"].mean(),
        "box_ratio": g_p["in_box_start"].mean(),
        "mean_speed": g_p["speed"].mean(),
        "forward_ratio": (g_p["pass_dx"] > 0).mean(),
    }
    X = pd.DataFrame([stats], columns=player_role_cols).astype(np.float64).fillna(0.0)
    X_scaled = role_scaler.transform(X)
    return int(role_kmeans.predict(X_scaled.astype(np.float64))[0])


def estimate_team_style_from_ep(
    g_feat: pd.DataFrame,
    team_id,
    team_style_cols: List[str],
    style_scaler: StandardScaler,
    style_kmeans: KMeans,
) -> int:
    g_t = g_feat[g_feat["team_id"] == team_id]
    if len(g_t) == 0:
        return 0
    stats = {
        "mean_start_x": g_t["start_x"].mean(),
        "mean_start_y": g_t["start_y"].mean(),
        "mean_dist_goal_start": g_t["dist_goal_start"].mean(),
        "mean_pass_dist": g_t["pass_dist"].mean(),
        "box_ratio": g_t["in_box_start"].mean(),
        "mean_speed": g_t["speed"].mean(),
        "forward_ratio": (g_t["pass_dx"] > 0).mean(),
    }
    X = pd.DataFrame([stats], columns=team_style_cols).astype(np.float64).fillna(0.0)
    X_scaled = style_scaler.transform(X)
    return int(style_kmeans.predict(X_scaled.astype(np.float64))[0])


def process_single_episode_for_test(
    ep_df: pd.DataFrame,
    stats_scaler: StandardScaler,
    kmeans_ep: KMeans,
    stats_cols: List[str],
    mappings: Dict[str, Any],
    means,
    stds,
    cfg: CFG,
    player_role_map: Dict,
    player_role_cols: List[str],
    player_role_scaler: StandardScaler,
    player_role_kmeans: KMeans,
    team_style_map: Dict,
    team_style_cols: List[str],
    team_style_scaler: StandardScaler,
    team_style_kmeans: KMeans,
):
    g_raw = ep_df.sort_values("time_seconds").reset_index(drop=True).copy()
    if len(g_raw) == 0:
        return None

    ref_team = g_raw["team_id"].iloc[-1]
    ref_team = int(ref_team) if not pd.isna(ref_team) else None

    g_raw_feat = compute_event_features_single_episode(g_raw)

    if len(g_raw_feat) >= 2:
        g_core = g_raw_feat.iloc[:-1].copy()
        mean_speed = float(g_core["speed"].mean())
        angle_std = float(g_core["angle_diff"].std()) if len(g_core) > 1 else 0.0
        n_events = float(len(g_core))
    else:
        mean_speed, angle_std, n_events = 0.0, 0.0, 0.0

    ep_feat_df = pd.DataFrame(
        [{"mean_speed": mean_speed, "angle_std": 0.0 if np.isnan(angle_std) else angle_std, "n_events": n_events}],
        columns=stats_cols,
    ).astype(np.float64)

    ep_feat_scaled = stats_scaler.transform(ep_feat_df)
    ep_vec = ep_feat_scaled[0].astype("float32")
    cluster_id = int(kmeans_ep.predict(ep_feat_scaled.astype(np.float64))[0])

    g_aligned = align_episode_to_ref_team(g_raw, ref_team, cfg)
    g_feat = compute_event_features_single_episode(g_aligned)

    # categorical ids in the aligned frame
    type2idx = mappings["type2idx"]
    type_unknown_idx = mappings["type_unknown_idx"]
    result2idx = mappings["result2idx"]
    result_unknown_idx = mappings["result_unknown_idx"]
    team2idx = mappings["team2idx"]
    team_unknown_idx = mappings["team_unknown_idx"]
    player2idx = mappings["player2idx"]
    player_unknown_idx = mappings["player_unknown_idx"]

    g_feat["type_id"] = g_feat["type_name"].map(type2idx).fillna(type_unknown_idx).astype(int)
    g_feat["result_name_filled"] = g_feat["result_name"].fillna("Unknown")
    g_feat["result_id"] = g_feat["result_name_filled"].map(result2idx).fillna(result_unknown_idx).astype(int)

    t = len(g_feat) - 1
    team_seq = g_feat["team_id"].values
    pos_start = last_possession_start_index(team_seq, ref_team, t)

    g_prefix = g_feat.iloc[: t + 1].copy()
    if cfg.use_last_possession_only:
        g_pos = g_feat.iloc[pos_start : t + 1].copy()
        if (len(g_pos) < cfg.min_episode_len) and cfg.possession_fallback_full_prefix:
            g_seq = g_prefix
        else:
            g_seq = g_pos
    else:
        g_seq = g_prefix

    if cfg.max_seq_len is not None and len(g_seq) > cfg.max_seq_len:
        g_seq = g_seq.iloc[-cfg.max_seq_len :].copy()

    seq_feats = normalize_seq_block(g_seq, means, stds)

    masked_cols = ["pass_dx", "pass_dy", "pass_dist", "speed", "angle", "angle_diff"]
    col_to_j = {col: j for j, col in enumerate(SEQ_BASE_COLS)}
    last_idx = seq_feats.shape[0] - 1
    for col in masked_cols:
        j = col_to_j[col]
        mu = float(means[col])
        sd = float(stds[col]) if float(stds[col]) != 0.0 else 1.0
        seq_feats[last_idx, j] = (0.0 - mu) / sd

    type_ids = g_seq["type_id"].values.astype("int64")
    result_ids = g_seq["result_id"].values.astype("int64")
    if ref_team is None:
        att_side_ids = np.zeros(len(g_seq), dtype="int64")
    else:
        att_side_ids = (g_seq["team_id"].values.astype("int64") != int(ref_team)).astype("int64")

    # team / opponent
    last_team_id = ref_team
    episode_team_ids = g_raw["team_id"].dropna().unique().tolist()
    opp_team_id = None
    for tid in episode_team_ids:
        if int(tid) != int(last_team_id):
            opp_team_id = int(tid)
            break

    team_id_enc = team2idx.get(last_team_id, team_unknown_idx)
    opp_team_id_enc = team2idx.get(opp_team_id, team_unknown_idx) if opp_team_id is not None else team_unknown_idx

    # team style id (fallback uses RAW)
    if last_team_id in team_style_map:
        team_style_id = int(team_style_map[last_team_id])
    else:
        team_style_id = estimate_team_style_from_ep(
            g_raw_feat, last_team_id, team_style_cols, team_style_scaler, team_style_kmeans
        )

    if opp_team_id is not None:
        if opp_team_id in team_style_map:
            opp_team_style_id = int(team_style_map[opp_team_id])
        else:
            opp_team_style_id = estimate_team_style_from_ep(
                g_raw_feat, opp_team_id, team_style_cols, team_style_scaler, team_style_kmeans
            )
    else:
        opp_team_style_id = 0

    # player & role (fallback uses RAW)
    last_player_id = g_raw["player_id"].iloc[-1]
    player_id_enc = player2idx.get(last_player_id, player_unknown_idx)
    if last_player_id in player_role_map:
        player_role_id = int(player_role_map[last_player_id])
    else:
        player_role_id = estimate_player_role_from_ep(
            g_raw_feat, last_player_id, player_role_cols, player_role_scaler, player_role_kmeans
        )

    last_row = g_feat.iloc[-1]
    last_start_xy = np.array([float(last_row["start_x"]), float(last_row["start_y"])], dtype="float32")
    heatmap = build_episode_heatmap_prefix(g_feat, ref_team, t, cfg, pos_start_idx=pos_start)

    return (
        seq_feats,
        type_ids,
        result_ids,
        att_side_ids,
        cluster_id,
        ep_vec,
        team_id_enc,
        opp_team_id_enc,
        team_style_id,
        opp_team_style_id,
        player_id_enc,
        player_role_id,
        last_start_xy,
        heatmap,
    )


# ============================
# Inference (fold ensemble)
# ============================
def predict_test_ensemble_from_artifacts(
    cfg: CFG,
    mappings: Dict[str, Any],
    fold_weights_paths: List[str],
    fold_meta_paths: List[str],
    fold_scores: Optional[List[float]],
    device: str,
    output_csv_path: str,
):
    if (not os.path.exists(cfg.test_meta_csv)) or (not os.path.exists(cfg.sample_submission_csv)):
        raise FileNotFoundError(
            f"Missing test.csv or sample_submission.csv under data_root='{cfg.data_root}'."
        )

    print(f"[INFO] Loading test meta from {cfg.test_meta_csv}")
    test_meta = pd.read_csv(cfg.test_meta_csv)
    submission = pd.read_csv(cfg.sample_submission_csv)
    submission = submission.merge(test_meta, on="game_episode", how="left")

    def _convert_path(p: str) -> str:
        p = str(p).replace("./", "")
        return os.path.join(cfg.data_root, p)

    submission["path"] = submission["path"].apply(_convert_path)

    fold_metas = [load_pickle(p) for p in fold_meta_paths]

    # build models
    models = []
    for w_path in fold_weights_paths:
        model = PassLSTMDenseHeatmapFineResidualMTL(
            num_event_types=mappings["num_event_types"],
            num_result_types=mappings["num_result_types"],
            num_clusters=cfg.ep_n_clusters,
            num_teams=mappings["num_teams"],
            num_players=mappings["num_players"],
            num_player_roles=cfg.player_role_n_clusters,
            num_team_styles=cfg.team_style_n_clusters,
            num_zones=cfg.zone_grid_x * cfg.zone_grid_y,
            num_fine_cells=cfg.fine_grid_x * cfg.fine_grid_y,
            in_features=len(SEQ_BASE_COLS),
            ep_stats_dim=len(fold_metas[0]["stats_cols"]),
            heatmap_channels=cfg.heatmap_channels,
            cfg=cfg,
        ).to(device)

        state_dict = torch.load(w_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    # fold weights: inverse of CV last distance (if provided)
    if fold_scores is None or any(s is None for s in fold_scores):
        w_folds = np.ones(len(models), dtype=np.float32) / float(len(models))
    else:
        fs = np.array(fold_scores, dtype=np.float32)
        w_folds = 1.0 / (fs + 1e-6)
        w_folds = w_folds / w_folds.sum()

    preds_x, preds_y = [], []

    for idx, row in submission.iterrows():
        ep_path = row["path"]
        if not os.path.exists(ep_path):
            preds_x.append(PITCH_X / 2.0)
            preds_y.append(PITCH_Y / 2.0)
            continue

        ep_df = pd.read_csv(ep_path)
        if len(ep_df) == 0:
            preds_x.append(PITCH_X / 2.0)
            preds_y.append(PITCH_Y / 2.0)
            continue

        fold_pred_ends = []

        for model, meta in zip(models, fold_metas):
            pack = process_single_episode_for_test(
                ep_df,
                meta["stats_scaler"],
                meta["kmeans_ep"],
                meta["stats_cols"],
                mappings,
                meta["means"],
                meta["stds"],
                cfg,
                meta["player_role_map"],
                meta["player_role_cols"],
                meta["player_role_scaler"],
                meta["player_role_kmeans"],
                meta["team_style_map"],
                meta["team_style_cols"],
                meta["team_style_scaler"],
                meta["team_style_kmeans"],
            )
            if pack is None:
                fold_pred_ends.append(np.array([PITCH_X / 2.0, PITCH_Y / 2.0], dtype=np.float32))
                continue

            (
                seq_feats,
                type_ids,
                result_ids,
                att_side_ids,
                cluster_id,
                ep_vec,
                team_id_enc,
                opp_team_id_enc,
                team_style_id,
                opp_team_style_id,
                player_id_enc,
                player_role_id,
                last_start_xy,
                heatmap,
            ) = pack

            x = torch.tensor(seq_feats, dtype=torch.float32).unsqueeze(0).to(device)
            t_ids = torch.tensor(type_ids, dtype=torch.long).unsqueeze(0).to(device)
            r_ids = torch.tensor(result_ids, dtype=torch.long).unsqueeze(0).to(device)
            a_ids = torch.tensor(att_side_ids, dtype=torch.long).unsqueeze(0).to(device)
            c_ids = torch.tensor([cluster_id], dtype=torch.long).to(device)
            team_ids_t = torch.tensor([team_id_enc], dtype=torch.long).to(device)
            opp_team_ids_t = torch.tensor([opp_team_id_enc], dtype=torch.long).to(device)
            team_style_ids_t = torch.tensor([team_style_id], dtype=torch.long).to(device)
            opp_team_style_ids_t = torch.tensor([opp_team_style_id], dtype=torch.long).to(device)
            player_ids_t = torch.tensor([player_id_enc], dtype=torch.long).to(device)
            player_role_ids_t = torch.tensor([player_role_id], dtype=torch.long).to(device)
            ep_stats_tensor = torch.tensor(ep_vec, dtype=torch.float32).unsqueeze(0).to(device)
            last_start_xy_tensor = torch.tensor(last_start_xy, dtype=torch.float32).unsqueeze(0).to(device)
            heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([len(seq_feats)], dtype=torch.long).to(device)

            with torch.no_grad():
                offsets, _zone_logits, fine_logits, mix_gate, h = model(
                    x, t_ids, r_ids, a_ids, c_ids, team_ids_t, opp_team_ids_t,
                    team_style_ids_t, opp_team_style_ids_t, player_ids_t, player_role_ids_t,
                    ep_stats_tensor, last_start_xy_tensor, heatmap_tensor, lengths,
                )
                pred_end, _pmax, _top1 = predict_end_topk_expected_eval(
                    model=model,
                    offsets=offsets,
                    fine_logits=fine_logits,
                    mix_gate=mix_gate,
                    h=h,
                    last_start_xy=last_start_xy_tensor,
                    cfg=cfg,
                )
                end_xy = pred_end.squeeze(0).cpu().numpy().astype(np.float32)

                if cfg.use_tta_mirror:
                    seq_m, last_start_m, hm_m = mirror_inference_pack(
                        seq_feats=np.array(seq_feats, dtype=np.float32),
                        last_start_xy=np.array(last_start_xy, dtype=np.float32),
                        heatmap=np.array(heatmap, dtype=np.float32),
                        means=meta["means"],
                        stds=meta["stds"],
                        cfg=cfg,
                    )
                    x_m = torch.tensor(seq_m, dtype=torch.float32).unsqueeze(0).to(device)
                    last_start_m_t = torch.tensor(last_start_m, dtype=torch.float32).unsqueeze(0).to(device)
                    hm_m_t = torch.tensor(hm_m, dtype=torch.float32).unsqueeze(0).to(device)
                    lengths_m = torch.tensor([len(seq_m)], dtype=torch.long).to(device)

                    offsets_m, _z_m, fine_logits_m, mix_gate_m, h_m = model(
                        x_m, t_ids, r_ids, a_ids, c_ids, team_ids_t, opp_team_ids_t,
                        team_style_ids_t, opp_team_style_ids_t, player_ids_t, player_role_ids_t,
                        ep_stats_tensor, last_start_m_t, hm_m_t, lengths_m,
                    )
                    pred_end_m, _pmax_m, _top1_m = predict_end_topk_expected_eval(
                        model=model,
                        offsets=offsets_m,
                        fine_logits=fine_logits_m,
                        mix_gate=mix_gate_m,
                        h=h_m,
                        last_start_xy=last_start_m_t,
                        cfg=cfg,
                    )
                    end_xy_m = pred_end_m.squeeze(0).cpu().numpy().astype(np.float32)
                    end_xy_m[1] = PITCH_Y - end_xy_m[1]
                    end_xy = 0.5 * (end_xy + end_xy_m)

            end_xy[0] = float(np.clip(end_xy[0], 0.0, PITCH_X))
            end_xy[1] = float(np.clip(end_xy[1], 0.0, PITCH_Y))
            fold_pred_ends.append(end_xy)

        pred_stack = np.stack(fold_pred_ends, axis=0)
        pred_end = (pred_stack * w_folds.reshape(-1, 1)).sum(axis=0)
        preds_x.append(float(pred_end[0]))
        preds_y.append(float(pred_end[1]))

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(submission)} episodes...")

    out_df = submission[["game_episode"]].copy()
    out_df["end_x"] = preds_x
    out_df["end_y"] = preds_y

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    print(f"[INFO] Submission saved to: {output_csv_path}")


# ============================
# Artifact IO
# ============================
def _to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if joblib is not None:
        joblib.dump(obj, path, compress=3)
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    if joblib is not None:
        return joblib.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_cfg_json(cfg_path: str) -> CFG:
    d = load_json(cfg_path)
    cfg = CFG(**{k: v for k, v in d.items() if hasattr(CFG, k)})
    # Some cfg.json may contain absolute paths; do NOT trust them for execution.
    return cfg


def write_cfg_json(cfg: CFG, path: str):
    save_json(asdict(cfg), path)


def write_mappings(mappings: Dict[str, Any], weights_dir: str):
    save_pickle(mappings, os.path.join(weights_dir, "mappings.pkl"))
    save_json(mappings, os.path.join(weights_dir, "mappings.json"))


def read_mappings(weights_dir: str) -> Dict[str, Any]:
    pkl_path = os.path.join(weights_dir, "mappings.pkl")
    json_path = os.path.join(weights_dir, "mappings.json")
    if os.path.exists(pkl_path):
        return load_pickle(pkl_path)
    if os.path.exists(json_path):
        return load_json(json_path)
    raise FileNotFoundError(f"mappings.pkl/json not found under: {weights_dir}")


def read_fold_scores(weights_dir: str, n_folds: int) -> Optional[List[float]]:
    fs_path = os.path.join(weights_dir, "fold_scores.json")
    if not os.path.exists(fs_path):
        return None
    d = load_json(fs_path)
    scores = d.get("fold_scores")
    if not scores or len(scores) != n_folds:
        return None
    return [float(s) for s in scores]


# ============================
# High-level pipelines (used by train.py / inference.py)
# ============================
def train_cv_and_save(cfg: CFG, device: str = "cuda") -> Dict[str, Any]:
    """
    Full CV training + save artifacts into cfg.weights_dir:
      - fold{i}.pt
      - cfg.json
      - mappings.pkl/json
      - meta_fold{i}.pkl
      - fold_scores.json

    Returns a dict with fold_scores and basic summary.
    """
    set_seed(cfg.seed)
    os.makedirs(cfg.weights_dir, exist_ok=True)

    if not os.path.exists(cfg.train_csv):
        raise FileNotFoundError(f"train.csv not found at: {cfg.train_csv}")

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading train from {cfg.train_csv}")
    train_raw = pd.read_csv(cfg.train_csv)

    train_df = add_event_features_all(train_raw)
    mappings = build_category_mappings(train_df)
    train_df = apply_category_mappings(train_df, mappings)

    print(
        "[INFO] Category sizes -> "
        f"event_types: {mappings['num_event_types']}, result_types: {mappings['num_result_types']}, "
        f"teams: {mappings['num_teams']}, players: {mappings['num_players']}, "
        f"ep_clusters: {cfg.ep_n_clusters}, zones: {cfg.zone_grid_x*cfg.zone_grid_y}, "
        f"fine_cells: {cfg.fine_grid_x*cfg.fine_grid_y}"
    )

    episodes = train_df[["game_episode", "game_id"]].drop_duplicates().reset_index(drop=True)
    ep_indices = np.arange(len(episodes))
    gkf = GroupKFold(n_splits=cfg.n_folds)

    fold_scores: List[float] = []

    for fold, (train_ep_idx, valid_ep_idx) in enumerate(gkf.split(ep_indices, groups=episodes["game_id"].values)):
        train_eps = episodes.loc[train_ep_idx, "game_episode"].values
        valid_eps = episodes.loc[valid_ep_idx, "game_episode"].values

        train_df_fold = train_df[train_df["game_episode"].isin(train_eps)].copy()
        valid_df_fold = train_df[train_df["game_episode"].isin(valid_eps)].copy()

        train_raw_fold = train_raw[train_raw["game_episode"].isin(train_eps)].copy()
        fold_means, fold_stds = compute_seq_feature_norm_stats_mixed(train_raw_fold)

        ep_stats_train = compute_episode_stats(train_df_fold)
        ep_stats_train, stats_scaler_fold, kmeans_ep_fold, stats_cols = fit_episode_cluster(
            ep_stats_train, n_clusters=cfg.ep_n_clusters, seed=cfg.seed
        )
        train_df_fold = apply_episode_cluster_to_df(train_df_fold, ep_stats_train)

        ep_stats_valid = compute_episode_stats(valid_df_fold)
        X_val = ep_stats_valid[stats_cols].astype(np.float64)
        X_val_scaled = stats_scaler_fold.transform(X_val)
        clusters_val = kmeans_ep_fold.predict(X_val_scaled.astype(np.float64))
        ep_stats_valid = ep_stats_valid.copy()
        ep_stats_valid["cluster"] = clusters_val
        valid_df_fold = apply_episode_cluster_to_df(valid_df_fold, ep_stats_valid)

        (
            _player_role_stats_fold,
            player_role_scaler_fold,
            player_role_kmeans_fold,
            player_role_cols,
            player_role_map_fold,
        ) = compute_player_role_clusters(train_df_fold, n_clusters=cfg.player_role_n_clusters, seed=cfg.seed)

        (
            _team_style_stats_fold,
            team_style_scaler_fold,
            team_style_kmeans_fold,
            team_style_cols,
            team_style_map_fold,
        ) = compute_team_style_clusters(train_df_fold, n_clusters=cfg.team_style_n_clusters, seed=cfg.seed)

        train_samples = build_dense_train_samples(
            train_df_fold,
            mappings,
            cfg,
            ep_stats_train,
            stats_scaler_fold,
            stats_cols,
            player_role_map_fold,
            team_style_map_fold,
            fold_means,
            fold_stds,
        )
        valid_samples = build_dense_train_samples(
            valid_df_fold,
            mappings,
            cfg,
            ep_stats_valid,
            stats_scaler_fold,
            stats_cols,
            player_role_map_fold,
            team_style_map_fold,
            fold_means,
            fold_stds,
        )

        meta = {
            "means": fold_means,
            "stds": fold_stds,
            "stats_scaler": stats_scaler_fold,
            "kmeans_ep": kmeans_ep_fold,
            "stats_cols": stats_cols,
            "player_role_map": player_role_map_fold,
            "player_role_cols": player_role_cols,
            "player_role_scaler": player_role_scaler_fold,
            "player_role_kmeans": player_role_kmeans_fold,
            "team_style_map": team_style_map_fold,
            "team_style_cols": team_style_cols,
            "team_style_scaler": team_style_scaler_fold,
            "team_style_kmeans": team_style_kmeans_fold,
        }

        best_state, best_val_last = train_one_fold(
            fold=fold,
            train_samples=train_samples,
            valid_samples=valid_samples,
            meta=meta,
            mappings=mappings,
            means=fold_means,
            stds=fold_stds,
            device=device,
            cfg=cfg,
        )

        fold_scores.append(best_val_last)

        # Save per-fold artifacts
        torch.save(best_state, os.path.join(cfg.weights_dir, f"fold{fold}.pt"))
        save_pickle(meta, os.path.join(cfg.weights_dir, f"meta_fold{fold}.pkl"))
        print(f"[Fold {fold}] Saved fold{fold}.pt and meta_fold{fold}.pkl")

    # Save common artifacts
    write_cfg_json(cfg, os.path.join(cfg.weights_dir, "cfg.json"))
    write_mappings(mappings, cfg.weights_dir)
    save_json({"fold_scores": fold_scores}, os.path.join(cfg.weights_dir, "fold_scores.json"))

    print("[INFO] CV Fold scores (ValLastDist in meters):")
    for i, s in enumerate(fold_scores):
        print(f"  Fold {i}: {s:.4f}")
    print(f"  -> CV Mean: {np.mean(fold_scores):.4f} | Std: {np.std(fold_scores):.4f}")

    return {"fold_scores": fold_scores, "cv_mean": float(np.mean(fold_scores)), "cv_std": float(np.std(fold_scores))}


def inference_and_save(cfg: CFG, device: str = "cuda") -> str:
    """
    Load artifacts from cfg.weights_dir and run inference on test set.
    Saves submission CSV under cfg.output_dir.
    Returns path to saved CSV.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load cfg.json if present (ensures hyperparams match weights), but allow CLI override for paths.
    cfg_path = os.path.join(cfg.weights_dir, "cfg.json")
    if os.path.exists(cfg_path):
        cfg_loaded = load_cfg_json(cfg_path)
        # keep hyperparams from cfg.json, but override paths from current cfg (relative paths)
        cfg_loaded.data_root = cfg.data_root
        cfg_loaded.train_csv = cfg.train_csv
        cfg_loaded.test_meta_csv = cfg.test_meta_csv
        cfg_loaded.sample_submission_csv = cfg.sample_submission_csv
        cfg_loaded.weights_dir = cfg.weights_dir
        cfg_loaded.output_dir = cfg.output_dir
        if cfg.submission_filename:
            cfg_loaded.submission_filename = cfg.submission_filename
        cfg = cfg_loaded

    mappings = read_mappings(cfg.weights_dir)
    fold_scores = read_fold_scores(cfg.weights_dir, cfg.n_folds)

    fold_weights_paths = [os.path.join(cfg.weights_dir, f"fold{i}.pt") for i in range(cfg.n_folds)]
    fold_meta_paths = [os.path.join(cfg.weights_dir, f"meta_fold{i}.pkl") for i in range(cfg.n_folds)]
    for p in fold_weights_paths + fold_meta_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required artifact: {p}")

    out_path = os.path.join(cfg.output_dir, cfg.submission_filename)
    predict_test_ensemble_from_artifacts(
        cfg=cfg,
        mappings=mappings,
        fold_weights_paths=fold_weights_paths,
        fold_meta_paths=fold_meta_paths,
        fold_scores=fold_scores,
        device=device,
        output_csv_path=out_path,
    )
    return out_path
