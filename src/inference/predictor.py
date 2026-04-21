import numpy as np
import torch

from src.config import Config
from src.inference.transform_and_slide_window import transform_and_slide_window
from src.models.rcnn_sed import Rcnnsed
from src.utils.checkpoint import load_checkpoint


class Predictor:
    """Runs inference on a single soundscape file."""

    def __init__(self, cfg: Config, checkpoint_path: str, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = Rcnnsed(cfg.model)
        load_checkpoint(checkpoint_path, self.model, device)
        self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, audio_path: str) -> np.ndarray:
        """Run inference on a 60s soundscape.

        Returns: (12, num_classes) probability matrix.
        """
        windows = transform_and_slide_window(audio_path, self.cfg.audio)   # (12, 1, 128, 500)
        batch = windows.to(self.device)
        logits = self.model(batch)                              # (12, T', num_classes)
        probs = torch.sigmoid(logits).mean(dim=1)              # (12, num_classes)
        return probs.cpu().numpy()
