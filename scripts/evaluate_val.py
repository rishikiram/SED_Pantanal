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
from src.evaluation.metrics import macro_roc_auc, segment_f1
from src.models.rcnn_sed import Rcnnsed
from src.training.losses import FocalBCELoss
from src.utils.label_encoder import LabelEncoder


DEFAULT_WINDOWS_CSV = 'cache/train_clip_windows.csv'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',   required=True,      help='Path to .pt checkpoint')
    p.add_argument('--config',       default='configs/base.yaml')
    p.add_argument('--windows_csv',  default=DEFAULT_WINDOWS_CSV,
                   help='Windows CSV from generate_clip_windows.py (must match training run)')
    p.add_argument('--val_split',    type=float, default=0.1)
    p.add_argument('--max_clips',    type=int,   default=None)
    p.add_argument('--n_eval_clips', type=int,   default=None, help='Cap number of validation windows (for quick runs)')
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
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Returns (val_loss, val_f1, val_auc, all_probs (N,234), all_labels (N,234))."""
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
    auc       = macro_roc_auc(probs_np, labels_np)
    return total_loss / len(loader), f1, auc, probs_np, labels_np


def play_audio(path: Path, offset_sec: float = 0.0, duration_sec: float = 5.0):
    """Play a 5s window from the audio file via sounddevice + soundfile."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print('  sounddevice not installed — run: pip install sounddevice')
        return
    try:
        info = sf.info(str(path))
        start = int(offset_sec * info.samplerate)
        stop  = int((offset_sec + duration_sec) * info.samplerate)
        print(f'  Playing {path.name} [{offset_sec:.1f}s – {offset_sec + duration_sec:.1f}s] ...')
        data, sr = sf.read(str(path), start=start, stop=stop, dtype='float32', always_2d=True)
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
        top_idx = np.argsort(probs)[::-1]
        # top_idx = np.argsort(probs)[::-1][:100]

        time_start = float(row.get('time_start', 0.0))
        print(f'\n  File       : {row["filename"]}  @ {time_start:.1f}s')
        print(f'  True label : {gt_primary}' +
              (f'  (secondary: {", ".join(gt_secondary)})' if gt_secondary else ''))
        # print(f'  Top-5 predictions (threshold={threshold}):')
        # for rank, ci in enumerate(top_idx, 1):
        #     species = enc.decode(ci)
        #     marker  = '*' if probs[ci] >= threshold else ' '
        #     is_gt   = ' [GT]' if (targets[ci] > 0) else ''
        #     print(f'    {rank}. {marker} {species:<30s}  p={probs[ci]:.4f}{is_gt}')

        # Bar-graph view of top-5 predictions
        BAR_WIDTH  = 50
        col_width  = max(len(enc.decode(ci)) for ci in top_idx)
        print(f'  Top-5 predictions (threshold={threshold}):')
        for rank, ci in enumerate(top_idx, 1):
            species    = enc.decode(ci)
            p          = probs[ci]
            filled     = int(round(p * BAR_WIDTH))
            bar        = '█' * filled + '░' * (BAR_WIDTH - filled)
            thresh_pos = int(round(threshold * BAR_WIDTH))
            bar        = bar[:thresh_pos] + '|' + bar[thresh_pos + 1:]
            is_gt      = ' [GT]' if (targets[ci] > 0) else ''
            print(f'    {rank:3}. {species:<{col_width}s}  [{bar}]  {p:.3f}{is_gt}')

        audio_path = audio_root / row['filename']
        if audio_path.exists():
            try:
                play = input('  Play audio? [y/N] ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if play == 'y':
                play_audio(audio_path, offset_sec=time_start)
        else:
            print(f'  Audio file not found: {audio_path}')


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    enc = LabelEncoder(f'{cfg.paths.data_root}/taxonomy.csv')

    # --- Reproduce the same val split as training ---
    windows_csv = Path(args.windows_csv)
    if not windows_csv.exists():
        print(f'ERROR: {windows_csv} not found.')
        print('Run:  python scripts/generate_clip_windows.py')
        sys.exit(1)

    full_df = pd.read_csv(windows_csv)
    n_total = len(full_df)
    if args.max_clips:
        n_total = min(n_total, args.max_clips)

    val_idx = make_val_indices(n_total, args.val_split, cfg.training.seed)
    if args.n_eval_clips:
        if args.n_eval_clips >= len(val_idx):
            print(f'  n_eval_clips ({args.n_eval_clips}) >= total val windows ({len(val_idx)}) — using all.')
        else:
            val_idx = val_idx[:args.n_eval_clips]
    print(f'Validation windows: {len(val_idx)}')

    audio_root = Path(f'{cfg.paths.data_root}/train_audio')
    val_ds     = ClipDataset(str(windows_csv), str(audio_root), enc, cfg.audio, indices=val_idx)
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
    from src.training.trainer import Trainer as _Trainer
    _Trainer.load_checkpoint(str(ckpt_path), model, device)
    model.to(device)
    print(f'Loaded checkpoint: {ckpt_path}')

    # --- Evaluate ---
    loss_fn = FocalBCELoss(gamma=cfg.training.focal_gamma)
    val_loss, val_f1, val_auc, probs_np, labels_np = run_eval(model, val_loader, loss_fn, device)

    print(f'\n{"="*50}')
    print(f'  val_loss : {val_loss:.4f}')
    print(f'  val_f1   : {val_f1:.4f}  (threshold={args.threshold})')
    print(f'  val_auc  : {val_auc:.4f}  (macro ROC-AUC)')
    print(f'{"="*50}')

    try:
        import sounddevice as sd
        sr = 44100
        t  = np.linspace(0, 0.15, int(sr * 0.15), endpoint=False)
        chime = (
            0.4 * np.sin(2 * np.pi * 880 * t) * np.exp(-8 * t) +
            0.3 * np.sin(2 * np.pi * 1320 * t) * np.exp(-8 * t)
        ).astype(np.float32)
        sd.play(chime, sr)
        sd.wait()
    except Exception:
        pass

    if args.no_interactive:
        return

    # Build val_df in the same order as the loader
    val_df = full_df.iloc[val_idx].reset_index(drop=True)
    interactive_browser(val_df, probs_np, labels_np, enc, audio_root, args.threshold)


if __name__ == '__main__':
    main()
