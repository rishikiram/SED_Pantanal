import torch
import torch.nn as nn

from src.models.cnn_backbone import CNNBackbone
from src.models.rnn_head import RNNHead
from src.config import ModelConfig


class Rcnnsed(nn.Module):
    """Full RCNN-SED model.

    Forward pass:
      (B, 1, 128, 500)
      → CNN → (B, C, H', T')
      → freq-pool → (B, C, T')
      → GRU → (B, T', num_classes)   # frame-level logits

    Sigmoid is NOT applied inside forward — use BCEWithLogitsLoss during training
    and torch.sigmoid() at inference.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.backbone = CNNBackbone(
            model_name=cfg.backbone,
            pretrained=cfg.backbone_pretrained,
            in_chans=cfg.in_chans,
        )
        self.rnn_head = RNNHead(
            input_size=self.backbone.out_channels,
            hidden_dim=cfg.rnn_hidden_dim,
            num_layers=cfg.rnn_num_layers,
            bidirectional=cfg.rnn_bidirectional,
            dropout=cfg.rnn_dropout,
            classifier_dropout=cfg.classifier_dropout,
            num_classes=cfg.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 128, 500)
        Returns: (B, T', num_classes) frame-level logits
        """
        feat = self.backbone(x)           # (B, C, H', T')
        feat = feat.mean(dim=2)           # freq-pool → (B, C, T')
        logits = self.rnn_head(feat)      # (B, T', num_classes)
        return logits
