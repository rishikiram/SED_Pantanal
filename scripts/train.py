"""Phase A training: train on clips, report validation results.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/base.yaml --val_split 0.1 --epochs 5
    python scripts/train.py --max_clips 2000   # quick smoke-run on a subset
"""
import argparse
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     default='configs/base.yaml')
    p.add_argument('--val_split',  type=float, default=0.1,  help='Fraction of clips held out for validation')
    p.add_argument('--epochs',     type=int,   default=None, help='Override num_epochs from config')
    p.add_argument('--max_clips',  type=int,   default=None, help='Cap dataset size (for quick runs)')
    p.add_argument('--num_workers',type=int,   default=0)
    return p.parse_args()


def make_splits(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(n * val_frac))
    return indices[n_val:], indices[:n_val]


def compute_label_counts(dataset: ClipDataset) -> torch.Tensor:
    counts = torch.zeros(dataset.num_classes)
    for _, label in dataset:
        counts += (label > 0).float()
    return counts


def main():
    args = parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    enc = LabelEncoder(f'{cfg.paths.data_root}/taxonomy.csv')

    # --- Build full index list ---
    train_csv = f'{cfg.paths.data_root}/train.csv'
    n_total = len(pd.read_csv(train_csv))
    if args.max_clips:
        n_total = min(n_total, args.max_clips)
    train_idx, val_idx = make_splits(n_total, args.val_split, cfg.training.seed)
    print(f'Clips — train: {len(train_idx)}  val: {len(val_idx)}')

    # --- Datasets ---
    audio_root = f'{cfg.paths.data_root}/train_audio'
    train_ds = ClipDataset(train_csv, audio_root, enc, cfg.audio, indices=train_idx)
    val_ds   = ClipDataset(train_csv, audio_root, enc, cfg.audio, indices=val_idx)

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

    # --- Class weights from training set ---
    print('Computing class weights...')
    label_counts = compute_label_counts(train_ds)
    print(f'  Classes with ≥1 sample: {(label_counts > 0).sum().item()} / {enc.num_classes}')

    # --- Model ---
    model = Rcnnsed(cfg.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')

    # --- Train ---
    trainer = Trainer(cfg, model, device)
    trainer.set_class_weights(label_counts)

    num_epochs = args.epochs or cfg.training.pretrain_clip_epochs
    checkpoint_dir = f'{cfg.paths.output_dir}/checkpoints'

    print(f'\nPhase A — clip pre-training for {num_epochs} epochs')
    print('-' * 60)
    best_f1 = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=cfg.training.lr,
        backbone_lr_multiplier=cfg.training.backbone_lr_multiplier,
        checkpoint_dir=checkpoint_dir,
        fold=0,
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
