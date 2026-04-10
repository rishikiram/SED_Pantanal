"""Evaluate a saved checkpoint on the validation split and browse predictions.

Usage:
    python scripts/evaluate_val.py --checkpoint outputs/checkpoints/best_fold0.pt
    python scripts/evaluate_val.py --checkpoint best_fold0.pt --val_split 0.1 --max_clips 500
    python scripts/evaluate_val.py --checkpoint best_fold0.pt --no_interactive

After showing the validation metrics, enter a clip index to:
  - See top-5 predicted species and their probabilities
  - Play the audio clip (requires 'sounddevice' — install with: pip install sounddevice)
  - Type 'q' or press Enter on an empty line to quit
"""
import argparse
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.clip_dataset import ClipDataset
from src.evaluation.metrics import segment_f1
from src.models.rcnn_sed import Rcnnsed
from src.training.losses import FocalBCELoss
from src.utils.label_encoder import LabelEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',   required=True,      help='Path to .pt checkpoint')
    p.add_argument('--config',       default='configs/base.yaml')
    p.add_argument('--val_split',    type=float, default=0.1)
    p.add_argument('--max_clips',    type=int,   default=None)
    p.add_argument('--num_workers',  type=int,   default=0)
    p.add_argument('--threshold',    type=float, default=0.5, help='Prediction threshold for F1')
    p.add_argument('--no_interactive', action='store_true', help='Skip interactive browser')
    return p.parse_args()


def make_val_indices(n: int, val_frac: float, seed: int) -> list[int]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(n * val_frac))
    return indices[:n_val]


def run_eval(
    model: Rcnnsed,
    loader: DataLoader,
    loss_fn: FocalBCELoss,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Returns (val_loss, val_f1, all_probs (N,234), all_labels (N,234))."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for mels, labels in tqdm(loader, desc='Evaluating', leave=True):
            mels   = mels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                logits = model(mels)                        # (B, T', 234)
                loss   = loss_fn(logits, labels, None)

            total_loss += loss.item()
            probs = torch.sigmoid(logits).mean(dim=1)       # (B, 234)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs_np  = np.concatenate(all_probs,  axis=0)          # (N, 234)
    labels_np = np.concatenate(all_labels, axis=0)          # (N, 234)
    f1        = segment_f1(probs_np, labels_np)
    return total_loss / len(loader), f1, probs_np, labels_np


def play_audio(path: Path):
    """Play audio via sounddevice + soundfile (supports OGG/WAV/FLAC)."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print('  sounddevice not installed — run: pip install sounddevice')
        return
    try:
        print(f'  Playing {path.name} ...')
        data, sr = sf.read(str(path), dtype='float32', always_2d=True)
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print(f'  Playback error: {e}')


def interactive_browser(
    val_df: pd.DataFrame,
    probs_np: np.ndarray,
    labels_np: np.ndarray,
    enc: LabelEncoder,
    audio_root: Path,
    threshold: float,
):
    n = len(val_df)
    print(f'\n--- Interactive browser ({n} validation clips) ---')
    print('Enter a clip index (0 to {}) to inspect, or "q" to quit.'.format(n - 1))

    while True:
        try:
            raw = input('\nClip index > ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if raw.lower() in ('q', 'quit', ''):
            break

        try:
            idx = int(raw)
        except ValueError:
            print('  Enter an integer index or "q".')
            continue

        if not (0 <= idx < n):
            print(f'  Index out of range (0–{n - 1}).')
            continue

        row       = val_df.iloc[idx]
        probs     = probs_np[idx]       # (234,)
        targets   = labels_np[idx]      # (234,)

        # Ground truth
        gt_primary   = str(row['primary_label']).strip()
        gt_secondary = []
        try:
            import ast
            gt_secondary = ast.literal_eval(str(row.get('secondary_labels', '[]')))
        except (ValueError, SyntaxError):
            pass

        # Top-5 predictions
        top5_idx = np.argsort(probs)[::-1][:5]

        print(f'\n  File       : {row["filename"]}')
        print(f'  True label : {gt_primary}' +
              (f'  (secondary: {", ".join(gt_secondary)})' if gt_secondary else ''))
        print(f'  Top-5 predictions (threshold={threshold}):')
        for rank, ci in enumerate(top5_idx, 1):
            species = enc.decode(ci)
            marker  = '*' if probs[ci] >= threshold else ' '
            is_gt   = ' [GT]' if (targets[ci] > 0) else ''
            print(f'    {rank}. {marker} {species:<30s}  p={probs[ci]:.4f}{is_gt}')

        audio_path = audio_root / row['filename']
        if audio_path.exists():
            try:
                play = input('  Play audio? [y/N] ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if play == 'y':
                play_audio(audio_path)
        else:
            print(f'  Audio file not found: {audio_path}')


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    enc = LabelEncoder(f'{cfg.paths.data_root}/taxonomy.csv')

    # --- Reproduce the same val split as training ---
    train_csv = f'{cfg.paths.data_root}/train.csv'
    full_df   = pd.read_csv(train_csv)
    n_total   = len(full_df)
    if args.max_clips:
        n_total = min(n_total, args.max_clips)

    val_idx = make_val_indices(n_total, args.val_split, cfg.training.seed)
    print(f'Validation clips: {len(val_idx)}')

    audio_root = Path(f'{cfg.paths.data_root}/train_audio')
    val_ds     = ClipDataset(train_csv, str(audio_root), enc, cfg.audio, indices=val_idx)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    # --- Load model ---
    model = Rcnnsed(cfg.model)
    ckpt_path = Path(args.checkpoint)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    print(f'Loaded checkpoint: {ckpt_path}')

    # --- Evaluate ---
    loss_fn = FocalBCELoss(gamma=cfg.training.focal_gamma)
    val_loss, val_f1, probs_np, labels_np = run_eval(model, val_loader, loss_fn, device)

    print(f'\n{"="*50}')
    print(f'  val_loss : {val_loss:.4f}')
    print(f'  val_f1   : {val_f1:.4f}  (threshold={args.threshold})')
    print(f'{"="*50}')

    if args.no_interactive:
        return

    # Build val_df in the same order as the loader
    val_df = full_df.iloc[val_idx].reset_index(drop=True)
    interactive_browser(val_df, probs_np, labels_np, enc, audio_root, args.threshold)


if __name__ == '__main__':
    main()
