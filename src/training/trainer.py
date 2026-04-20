import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.evaluation.metrics import segment_f1
from src.models.rcnn_sed import Rcnnsed
from src.training.losses import FocalBCELoss, compute_class_weights


class Trainer:
    def __init__(self, cfg: Config, model: Rcnnsed, device: torch.device, progress: bool = True):
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.loss_fn = FocalBCELoss(gamma=cfg.training.focal_gamma)
        self.class_weights: torch.Tensor | None = None
        self.progress = progress   # set False to silence all tqdm bars
        self.scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')

    def set_class_weights(self, label_counts: torch.Tensor):
        self.class_weights = compute_class_weights(label_counts).to(self.device)

    def _make_optimizer(self, lr: float, backbone_lr: float) -> torch.optim.Optimizer:
        backbone_params = list(self.model.backbone.parameters())
        head_params = list(self.model.rnn_head.parameters())
        return torch.optim.AdamW(
            [
                {'params': backbone_params, 'lr': backbone_lr},
                {'params': head_params, 'lr': lr},
            ],
            weight_decay=self.cfg.training.weight_decay,
        )

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer, epoch_desc: str = 'Train') -> float:
        self.model.train()
        total_loss = 0.0
        bar = tqdm(loader, desc=epoch_desc, leave=False, disable=not self.progress)
        for mels, labels in bar:
            mels = mels.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
                logits = self.model(mels)           # (B, T', C)
                loss = self.loss_fn(logits, labels, self.class_weights)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += loss.item()
            bar.set_postfix(loss=f'{loss.item():.4f}')

        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        bar = tqdm(loader, desc='Val  ', leave=False, disable=not self.progress)
        for mels, labels in bar:
            mels = mels.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
                logits = self.model(mels)           # (B, T', C)
                loss = self.loss_fn(logits, labels, self.class_weights)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).mean(dim=1)  # mean over T' → (B, C)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            bar.set_postfix(loss=f'{loss.item():.4f}')

        probs_np = np.concatenate(all_probs, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        f1 = segment_f1(probs_np, labels_np)
        return total_loss / len(loader), f1

    @staticmethod
    def load_checkpoint(path: str, model: 'Rcnnsed', device: torch.device) -> dict:
        """Load a checkpoint and restore model weights. Returns the full checkpoint dict."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        return ckpt

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        best_f1: float,
    ):
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_f1': best_f1,
        }, path)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        num_epochs: int,
        lr: float,
        backbone_lr_multiplier: float = 0.1,
        checkpoint_dir: str | None = None,
        fold: int = 0,
        log_path: str | None = None,
        resume_checkpoint: str | None = None,
    ):
        backbone_lr = lr * backbone_lr_multiplier
        optimizer = self._make_optimizer(lr, backbone_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

        best_f1 = 0.0
        start_epoch = 0

        if resume_checkpoint:
            ckpt = torch.load(resume_checkpoint, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            self.scaler.load_state_dict(ckpt['scaler'])
            start_epoch = ckpt['epoch'] + 1
            best_f1 = ckpt['best_f1']
            print(f'Resumed from {resume_checkpoint}  (epoch {ckpt["epoch"]+1}, best_f1={best_f1:.4f})')

        ckpt_dir = None
        if checkpoint_dir:
            ckpt_dir = Path(checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        n_train = len(train_loader.dataset)

        for epoch in range(start_epoch, num_epochs):
            t0 = time.perf_counter()
            train_loss = self.train_epoch(train_loader, optimizer, epoch_desc=f'Train {epoch+1}/{num_epochs}')
            epoch_sec = time.perf_counter() - t0
            scheduler.step()

            lr_head = optimizer.param_groups[1]['lr']
            lr_backbone = optimizer.param_groups[0]['lr']
            samples_per_sec = n_train / epoch_sec

            log_entry: dict = {
                'epoch': epoch + 1,
                'train_loss': round(train_loss, 6),
                'epoch_sec': round(epoch_sec, 2),
                'samples_per_sec': round(samples_per_sec, 1),
                'lr_head': lr_head,
                'lr_backbone': lr_backbone,
            }

            if val_loader is not None:
                val_loss, val_f1 = self.eval_epoch(val_loader)
                log_entry['val_loss'] = round(val_loss, 6)
                log_entry['val_f1'] = round(val_f1, 6)
                print(
                    f'Epoch {epoch+1}/{num_epochs}'
                    f'  train={train_loss:.4f}  val={val_loss:.4f}  f1={val_f1:.4f}'
                    f'  {samples_per_sec:.0f} samp/s  {epoch_sec:.1f}s'
                )
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    if ckpt_dir:
                        self._save_checkpoint(
                            ckpt_dir / f'best_fold{fold}.pt', epoch, optimizer, scheduler, best_f1
                        )
            else:
                print(
                    f'Epoch {epoch+1}/{num_epochs}'
                    f'  train={train_loss:.4f}'
                    f'  {samples_per_sec:.0f} samp/s  {epoch_sec:.1f}s'
                )

            if ckpt_dir:
                self._save_checkpoint(
                    ckpt_dir / f'epoch{epoch+1}_fold{fold}.pt', epoch, optimizer, scheduler, best_f1
                )

            if log_path:
                with open(log_path, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')

        return best_f1
