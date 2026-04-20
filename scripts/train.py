"""Phase A training: train on clips, report validation results.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/base.yaml --val_split 0.1 --epochs 5
    python scripts/train.py --max_clips 2000   # quick smoke-run on a subset

Requires cache/train_clip_windows.csv — run scripts/generate_clip_windows.py first.
"""
import argparse
import dataclasses
import datetime
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.clip_dataset import ClipDataset
from src.models.rcnn_sed import Rcnnsed
from src.training.losses import compute_class_weights
from src.training.trainer import Trainer
from src.utils.label_encoder import LabelEncoder

DEFAULT_WINDOWS_CSV = 'cache/train_clip_windows.csv'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     default='configs/base.yaml')
    p.add_argument('--windows_csv', default=DEFAULT_WINDOWS_CSV,
                   help='Windows CSV from generate_clip_windows.py')
    p.add_argument('--val_split',  type=float, default=0.1,  help='Fraction of windows held out for validation')
    p.add_argument('--epochs',     type=int,   default=None, help='Override num_epochs from config')
    p.add_argument('--max_clips',  type=int,   default=None, help='Cap dataset size (for quick runs)')
    p.add_argument('--num_workers',   type=int,   default=0)
    p.add_argument('--checkpoint_dir', default=None, help='Where to save checkpoints (default: outputs/checkpoints)')
    p.add_argument('--resume', default=None, metavar='CHECKPOINT',
                   help='Path to a checkpoint to resume training from')
    return p.parse_args()


def make_splits(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(n * val_frac))
    return indices[n_val:], indices[:n_val]


def compute_label_counts(windows_csv: str, indices: list[int], enc: LabelEncoder) -> torch.Tensor:
    """Count positive labels per class by reading the CSV directly — no audio loading."""
    import ast
    df = pd.read_csv(windows_csv).iloc[indices]
    counts = torch.zeros(enc.num_classes)
    for _, row in df.iterrows():
        try:
            counts[enc.encode(str(row['primary_label']).strip())] += 1
        except KeyError:
            pass
        try:
            for lbl in ast.literal_eval(str(row.get('secondary_labels', '[]'))):
                try:
                    counts[enc.encode(str(lbl).strip())] += 1
                except KeyError:
                    pass
        except (ValueError, SyntaxError):
            pass
    return counts


def main():
    args = parse_args()
    cfg = load_config(args.config)

    windows_csv = Path(args.windows_csv)
    if not windows_csv.exists():
        print(f'ERROR: {windows_csv} not found.')
        print('Run:  python scripts/generate_clip_windows.py')
        sys.exit(1)

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    enc = LabelEncoder(f'{cfg.paths.data_root}/taxonomy.csv')

    # --- Build full index list ---
    n_total = len(pd.read_csv(windows_csv))
    if args.max_clips:
        n_total = min(n_total, args.max_clips)
    train_idx, val_idx = make_splits(n_total, args.val_split, cfg.training.seed)
    print(f'Windows — train: {len(train_idx)}  val: {len(val_idx)}')

    # --- Datasets ---
    audio_root = f'{cfg.paths.data_root}/train_audio'
    train_ds = ClipDataset(str(windows_csv), audio_root, enc, cfg.audio, indices=train_idx)
    val_ds   = ClipDataset(str(windows_csv), audio_root, enc, cfg.audio, indices=val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
    )

    # --- Save run config snapshot ---
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'windows_csv': str(windows_csv),
        'num_train': len(train_idx),
        'num_val': len(val_idx),
        'device': str(device),
        'config': dataclasses.asdict(cfg),
    }
    run_config_path = output_dir / 'run_config.json'
    run_config_path.write_text(json.dumps(run_config, indent=2))
    print(f'Run config saved to {run_config_path}')

    # --- Class weights from training set ---
    print('Computing class weights...')
    label_counts = compute_label_counts(str(windows_csv), train_idx, enc)
    print(f'  Classes with ≥1 sample: {(label_counts > 0).sum().item()} / {enc.num_classes}')

    # --- Model ---
    model = Rcnnsed(cfg.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')

    # --- Train ---
    trainer = Trainer(cfg, model, device)
    trainer.set_class_weights(label_counts)

    num_epochs = args.epochs or cfg.training.pretrain_clip_epochs
    checkpoint_dir = args.checkpoint_dir or f'{cfg.paths.output_dir}/checkpoints'

    print(f'\nPhase A — clip pre-training for {num_epochs} epochs')
    print('-' * 60)
    log_path = str(output_dir / 'train_log.jsonl')
    best_f1 = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=cfg.training.lr,
        backbone_lr_multiplier=cfg.training.backbone_lr_multiplier,
        checkpoint_dir=checkpoint_dir,
        fold=0,
        log_path=log_path,
        resume_checkpoint=args.resume,
    )

    # --- Final validation pass ---
    print('-' * 60)
    val_loss, val_f1 = trainer.eval_epoch(val_loader)
    print(f'Final val_loss={val_loss:.4f}  val_f1={val_f1:.4f}  best_val_f1={best_f1:.4f}')

    # --- Save results ---
    results = {
        'best_val_f1': best_f1,
        'final_val_loss': val_loss,
        'final_val_f1': val_f1,
        'num_train': len(train_idx),
        'num_val': len(val_idx),
        'num_epochs': num_epochs,
        'checkpoint': str(Path(checkpoint_dir) / 'best_fold0.pt'),
    }
    results_path = Path(cfg.paths.output_dir) / 'train_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f'Results saved to {results_path}')


if __name__ == '__main__':
    main()
