"""Run inference on labeled train_soundscapes and report F1 against ground truth.

Mirrors the kaggle_inference.py pipeline but targets train_soundscapes/ so
predictions can be compared against train_soundscapes_labels.csv.

Usage:
    python scripts/infer_train_soundscapes.py --checkpoint outputs/checkpoints/best_fold0.pt
    python scripts/infer_train_soundscapes.py --checkpoint outputs/checkpoints/best_fold0.pt --limit 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import load_config
from src.evaluation.metrics import macro_roc_auc, segment_f1
from src.inference.predictor import Predictor
from src.utils.label_encoder import LabelEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    p.add_argument('--config',     default='configs/base.yaml')
    p.add_argument('--limit',      type=int, default=None,
                   help='Only process this many soundscapes (for quick runs)')
    return p.parse_args()


def parse_start_sec(time_str: str) -> int:
    """Convert HH:MM:SS to integer seconds."""
    h, m, s = time_str.strip().split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def build_ground_truth(labels_df: pd.DataFrame, enc: LabelEncoder, window_sec: int = 5) -> tuple[list[str], np.ndarray]:
    """Build a (N_windows, num_classes) ground truth matrix from the labels CSV.

    Returns row_ids and the label matrix, aligned to the same row_id format
    as make_submission (stem_endsec).
    """
    rows = {}
    for _, row in labels_df.iterrows():
        stem = Path(row['filename']).stem
        start_sec = parse_start_sec(row['start'])
        end_sec = start_sec + window_sec
        row_id = f'{stem}_{end_sec}'

        vec = np.zeros(enc.num_classes, dtype=np.float32)
        for lbl in str(row['primary_label']).split(';'):
            lbl = lbl.strip()
            try:
                vec[enc.encode(lbl)] = 1.0
            except KeyError:
                pass
        rows[row_id] = vec

    row_ids = sorted(rows.keys())
    matrix = np.stack([rows[r] for r in row_ids])
    return row_ids, matrix


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    enc = LabelEncoder(f'{cfg.paths.data_root}/taxonomy.csv')
    soundscape_dir = Path(cfg.paths.data_root) / 'train_soundscapes'
    labels_csv = Path(cfg.paths.data_root) / 'train_soundscapes_labels.csv'

    labels_df = pd.read_csv(labels_csv)
    labeled_files = sorted(labels_df['filename'].unique())
    if args.limit:
        labeled_files = labeled_files[:args.limit]
    print(f'Labeled soundscapes: {len(labeled_files)}')

    predictor = Predictor(cfg, args.checkpoint, device)

    # --- Run inference ---
    soundscape_probs: dict[str, np.ndarray] = {}
    for filename in tqdm(labeled_files, desc='Inference'):
        path = soundscape_dir / filename
        if not path.exists():
            print(f'  Missing: {filename}')
            continue
        stem = path.stem
        soundscape_probs[stem] = predictor.predict(str(path))  # (N, 234)

    # --- Build aligned pred/GT matrices ---
    # Only keep windows that have ground truth labels
    gt_df = labels_df[labels_df['filename'].isin(labeled_files)]
    gt_row_ids, gt_matrix = build_ground_truth(gt_df, enc)

    pred_rows = {}
    for stem, probs in soundscape_probs.items():
        for i, prob_vec in enumerate(probs):
            end_sec = (i + 1) * 5
            pred_rows[f'{stem}_{end_sec}'] = prob_vec

    # Align to GT row order — skip windows with no GT entry
    aligned_preds = []
    aligned_gt = []
    missing = 0
    for row_id, gt_vec in zip(gt_row_ids, gt_matrix):
        if row_id in pred_rows:
            aligned_preds.append(pred_rows[row_id])
            aligned_gt.append(gt_vec)
        else:
            missing += 1

    if missing:
        print(f'  Warning: {missing} GT windows had no matching prediction')

    preds_np = np.stack(aligned_preds)   # (N, 234)
    gt_np    = np.stack(aligned_gt)      # (N, 234)

    # --- Metrics ---
    f1  = segment_f1(preds_np, gt_np)
    auc = macro_roc_auc(preds_np, gt_np)
    pos_windows = int(gt_np.max(axis=1).sum())
    print(f'\nResults over {len(preds_np)} windows ({pos_windows} with ≥1 positive label):')
    print(f'  Segment-F1 @ 0.5 : {f1:.4f}')
    print(f'  Macro ROC-AUC    : {auc:.4f}')


if __name__ == '__main__':
    main()
