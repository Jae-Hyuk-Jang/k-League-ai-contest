# -*- coding: utf-8 -*-
"""
inference.py

Inference script for KLeagueAI Open Track 1 (v7).

- Requires the following under ./weights (or --weights_dir):
    cfg.json
    mappings.pkl (or mappings.json)
    meta_fold{0..4}.pkl
    fold{0..4}.pt
    fold_scores.json (optional, used for weighted ensemble)

- Reads test episodes via:
    ./data/test.csv + ./data/sample_submission.csv
    where sample_submission.csv contains relative file paths under ./data/test/

- Writes:
    ./output/<submission_filename>.csv
"""

from __future__ import annotations

import argparse
import os
import torch

from kleague_v7_core import build_cfg, inference_and_save


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data", help="Path to data directory (contains test.csv etc.)")
    p.add_argument("--weights_dir", type=str, default="./weights", help="Weights/meta directory")
    p.add_argument("--output_dir", type=str, default="./output", help="Where to write submission CSV")
    p.add_argument("--submission_filename", type=str, default=None, help="Override output csv name")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu. Default: auto")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = build_cfg(
        data_root=args.data_root,
        weights_dir=args.weights_dir,
        output_dir=args.output_dir,
        submission_filename=args.submission_filename,
    )

    cfg.data_root = os.path.normpath(cfg.data_root)
    cfg.weights_dir = os.path.normpath(cfg.weights_dir)
    cfg.output_dir = os.path.normpath(cfg.output_dir)

    print("[INFO] device:", device)
    print("[INFO] data_root:", cfg.data_root)
    print("[INFO] weights_dir:", cfg.weights_dir)
    print("[INFO] output_dir:", cfg.output_dir)

    out_path = inference_and_save(cfg=cfg, device=device)
    print("[DONE] Inference finished.")
    print("[INFO] Saved:", out_path)


if __name__ == "__main__":
    main()
