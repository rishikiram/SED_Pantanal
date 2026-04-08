from pathlib import Path

import numpy as np
import torch

from src.config import Config
from src.inference.sliding_window import sliding_window
from src.models.rcnn_sed import Rcnnsed


class Predictor:
    """Runs inference on a single soundscape file.

    Supports ensembling over multiple model checkpoints.
    """

    def __init__(self, cfg: Config, checkpoint_paths: list[str], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.models = []
        for ckpt in checkpoint_paths:
            model = Rcnnsed(cfg.model)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.to(device).eval()
            self.models.append(model)

    @torch.no_grad()
    def predict(self, audio_path: str) -> np.ndarray:
        """Run inference on a 60s soundscape.

        Returns: (12, num_classes) probability matrix, averaged over ensemble.
        """
        windows = sliding_window(audio_path, self.cfg.audio)   # (12, 1, 128, 500)
        batch = windows.to(self.device)

        ensemble_probs = []
        for model in self.models:
            logits = model(batch)                               # (12, T', num_classes)
            probs = torch.sigmoid(logits).mean(dim=1)          # (12, num_classes)
            ensemble_probs.append(probs.cpu().numpy())

        return np.mean(ensemble_probs, axis=0)                 # (12, num_classes)
