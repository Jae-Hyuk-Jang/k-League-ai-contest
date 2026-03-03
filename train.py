# -*- coding: utf-8 -*-
"""train.py

Reproducible training entrypoint (CV 5 folds) for KLeagueAI Open Track 1 (v7).

This script is intentionally *thin* and delegates the actual ML logic to
`kleague_v7_core.py` to avoid divergence between training/inference code.

Key points for reproducibility
- Uses relative paths by default (./data, ./weights, ./output).
- Forces working directory = folder containing this file, so relative paths are stable.
- Sets seeds + common deterministic toggles (cuDNN, TF32 off).
- Optionally enforces PyTorch deterministic algorithms.
- Writes environment snapshot into weights/env_info.json for audit.

Artifacts saved under weights_dir:
- fold{0..4}.pt
- meta_fold{0..4}.pkl
- mappings.pkl (+ mappings.json)
- cfg.json
- fold_scores.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from kleague_v7_core import build_cfg, load_cfg_json, set_seed, train_cv_and_save


def _chdir_to_project_root() -> Path:
    """Make relative paths stable no matter where the user runs `python train.py` from."""
    root = Path(__file__).resolve().parent
    os.chdir(str(root))
    return root


def _norm_path(p: str) -> str:
    # Keep paths readable, but avoid surprises with trailing slashes.
    return os.path.normpath(p)


def _safe_set_env_for_repro(seed: int, strict_deterministic: bool, warn_only: bool) -> Dict[str, Any]:
    """Best-effort reproducibility settings.

    Notes
    - True bitwise reproducibility across *different* GPUs/CUDA/cuDNN versions is not guaranteed.
    - These toggles significantly reduce run-to-run variance on the same environment.
    """
    info: Dict[str, Any] = {}

    # Helpful for determinism in Python hashing-based ops.
    # NOTE: Full effect requires setting before interpreter start, but we still record it.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    # cuBLAS determinism for some GEMM paths (CUDA).
    # NOTE: Must be set before CUDA context init for full effect.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Disable TF32 to reduce numerical drift across GPUs.
    try:
        torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
    except Exception:
        pass

    # Enforce deterministic kernels when possible.
    # warn_only=True: no crash, but prints warnings when nondeterministic ops are used.
    # strict_deterministic=True and warn_only=False: may raise RuntimeError for some CUDA ops.
    try:
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(strict_deterministic, warn_only=warn_only)  # type: ignore[arg-type]
            info["torch_use_deterministic_algorithms"] = bool(strict_deterministic)
            info["torch_deterministic_warn_only"] = bool(warn_only)
    except Exception as e:
        # If strict was requested, bubble up; otherwise just record.
        info["torch_use_deterministic_algorithms_error"] = repr(e)
        if strict_deterministic and (not warn_only):
            raise

    return info


def _collect_env_info(device: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_cudnn": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }

    if torch.cuda.is_available():
        try:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            info["cuda_device_name"] = None
        try:
            info["cuda_device_capability"] = list(torch.cuda.get_device_capability(0))
        except Exception:
            info["cuda_device_capability"] = None

    if extra:
        info.update(extra)

    return info


def _write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # I/O
    p.add_argument("--data_root", type=str, default="./data", help="Path to data directory (contains train.csv etc.)")
    p.add_argument("--weights_dir", type=str, default="./weights", help="Where to save fold weights / meta artifacts")
    p.add_argument("--output_dir", type=str, default="./output", help="Kept for consistency (not used in training)")

    # Runtime
    p.add_argument("--device", type=str, default=None, help="cuda / cpu. Default: auto")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Repro
    p.add_argument(
        "--strict_deterministic",
        action="store_true",
        help="Enable torch.use_deterministic_algorithms(True). May error on some CUDA ops.",
    )
    p.add_argument(
        "--deterministic_warn_only",
        action="store_true",
        help="If strict deterministic is enabled, convert errors to warnings when possible.",
    )

    p.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable AMP even when CUDA is available (more stable reproducibility).",
    )

    # Config
    p.add_argument(
        "--cfg_json",
        type=str,
        default=None,
        help="Optional cfg.json to load hyperparameters from (paths are still taken from CLI).",
    )

    # Safety
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into weights_dir even if fold*.pt already exist.",
    )

    return p.parse_args()


def main() -> None:
    _chdir_to_project_root()
    args = parse_args()

    # Normalize (but keep them relative-friendly)
    data_root = _norm_path(args.data_root)
    weights_dir = _norm_path(args.weights_dir)
    output_dir = _norm_path(args.output_dir)

    # Device selection
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prevent accidental mixing of old artifacts
    if os.path.isdir(weights_dir) and (not args.overwrite):
        existing = []
        for i in range(5):
            if os.path.exists(os.path.join(weights_dir, f"fold{i}.pt")):
                existing.append(f"fold{i}.pt")
        if existing:
            raise RuntimeError(
                "Refusing to overwrite existing weights in weights_dir without --overwrite. "
                f"Found: {existing} under {weights_dir}"
            )

    # Build cfg (paths)
    cfg = build_cfg(data_root=data_root, weights_dir=weights_dir, output_dir=output_dir)

    # Optionally load hyperparams from cfg.json (for audit/replay)
    if args.cfg_json is not None:
        cfg_loaded = load_cfg_json(args.cfg_json)
        # Keep CLI paths
        cfg_loaded.data_root = cfg.data_root
        cfg_loaded.train_csv = cfg.train_csv
        cfg_loaded.test_meta_csv = cfg.test_meta_csv
        cfg_loaded.sample_submission_csv = cfg.sample_submission_csv
        cfg_loaded.weights_dir = cfg.weights_dir
        cfg_loaded.output_dir = cfg.output_dir
        cfg = cfg_loaded

    # Apply seed from CLI (even if cfg_json had a different one)
    cfg.seed = int(args.seed)

    if getattr(args, "no_amp", False):
        cfg.use_amp = False

    # Repro toggles (best effort)
    repro_info = _safe_set_env_for_repro(
        seed=cfg.seed,
        strict_deterministic=bool(args.strict_deterministic),
        warn_only=bool(args.deterministic_warn_only),
    )

    # Seed
    set_seed(cfg.seed)

    # Basic I/O sanity
    if not os.path.exists(cfg.train_csv):
        raise FileNotFoundError(
            f"train.csv not found at: {cfg.train_csv}. "
            "Check --data_root (should contain train.csv)."
        )

    os.makedirs(cfg.weights_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Write env snapshot *before* training
    env_info = _collect_env_info(device=device, extra={"repro": repro_info, "cfg_seed": cfg.seed})
    _write_json(os.path.join(cfg.weights_dir, "env_info.json"), env_info)

    print("[INFO] device:", device)
    print("[INFO] data_root:", cfg.data_root)
    print("[INFO] weights_dir:", cfg.weights_dir)
    print("[INFO] output_dir:", cfg.output_dir)
    print("[INFO] seed:", cfg.seed)

    summary = train_cv_and_save(cfg=cfg, device=device)

    # Post-check: ensure expected artifacts exist
    expected = [
        "cfg.json",
        "mappings.json",
        "fold_scores.json",
    ]
    for i in range(cfg.n_folds):
        expected.append(f"fold{i}.pt")
        expected.append(f"meta_fold{i}.pkl")

    missing = [name for name in expected if not os.path.exists(os.path.join(cfg.weights_dir, name))]
    if missing:
        raise RuntimeError(f"Training finished but some expected artifacts are missing under {cfg.weights_dir}: {missing}")

    print("[DONE] Training finished.")
    print(summary)


if __name__ == "__main__":
    main()
