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
            mels = mels.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            logits = self.model(mels)           # (B, T', C)
            loss = self.loss_fn(logits, labels, self.class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip_norm)
            optimizer.step()
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
            mels = mels.to(self.device)
            labels = labels.to(self.device)

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

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        num_epochs: int,
        lr: float,
        backbone_lr_multiplier: float = 0.1,
        checkpoint_dir: str | None = None,
        fold: int = 0,
    ):
        backbone_lr = lr * backbone_lr_multiplier
        optimizer = self._make_optimizer(lr, backbone_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

        best_f1 = 0.0
        ckpt_dir = None
        if checkpoint_dir:
            ckpt_dir = Path(checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer, epoch_desc=f'Train {epoch+1}/{num_epochs}')
            scheduler.step()

            if val_loader is not None:
                val_loss, val_f1 = self.eval_epoch(val_loader)
                print(f'Epoch {epoch+1}/{num_epochs}  train={train_loss:.4f}  val={val_loss:.4f}  f1={val_f1:.4f}')
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    if ckpt_dir:
                        torch.save(self.model.state_dict(), ckpt_dir / f'best_fold{fold}.pt')
            else:
                print(f'Epoch {epoch+1}/{num_epochs}  train={train_loss:.4f}')

            if ckpt_dir:
                torch.save(self.model.state_dict(), ckpt_dir / f'epoch{epoch+1}_fold{fold}.pt')

        return best_f1
