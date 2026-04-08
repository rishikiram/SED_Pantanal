import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import AudioConfig
from src.data.audio_io import load_audio, pad_or_trim
from src.data.mel_transform import MelTransform
from src.utils.label_encoder import LabelEncoder


class ClipDataset(Dataset):
    """train_audio clips → mel (1, 128, 500), label (234,).

    Primary label gets weight 1.0, secondary labels get weight 0.5.
    """

    def __init__(
        self,
        train_csv: str,
        audio_root: str,
        encoder: LabelEncoder,
        audio_cfg: AudioConfig,
        indices: list[int] | None = None,
    ):
        self.df = pd.read_csv(train_csv)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

        self.audio_root = Path(audio_root)
        self.encoder = encoder
        self.audio_cfg = audio_cfg
        self.transform = MelTransform(audio_cfg)
        self.num_classes = encoder.num_classes

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load and preprocess audio
        path = self.audio_root / row['filename']
        wav = load_audio(str(path), self.audio_cfg.sample_rate)
        wav = pad_or_trim(wav, self.audio_cfg.samples_per_window)
        mel = self.transform(wav)  # (1, 128, 500)

        # Build label vector
        label = torch.zeros(self.num_classes, dtype=torch.float32)

        primary = str(row['primary_label']).strip()
        try:
            label[self.encoder.encode(primary)] = 1.0
        except KeyError:
            pass  # unknown label — skip silently

        secondary_raw = row.get('secondary_labels', '[]')
        try:
            secondary = ast.literal_eval(str(secondary_raw))
        except (ValueError, SyntaxError):
            secondary = []
        for lbl in secondary:
            try:
                label[self.encoder.encode(str(lbl).strip())] = 0.5
            except KeyError:
                pass

        return mel, label
