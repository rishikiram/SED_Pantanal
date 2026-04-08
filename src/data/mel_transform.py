import torch
import torchaudio
from src.config import AudioConfig


class MelTransform:
    """Converts a (1, N) waveform to a normalised (1, n_mels, T) mel spectrogram.

    torchaudio center-pads by n_fft//2 on each side, producing one extra frame.
    We trim to exactly frames_per_window after the transform.
    """

    def __init__(self, cfg: AudioConfig):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=cfg.top_db)
        self.frames_per_window = cfg.frames_per_window

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (1, N)
        Returns: (1, n_mels, T)
        """
        spec = self.mel(waveform)                       # (1, n_mels, T)
        spec = self.to_db(spec)                         # log scale
        spec = spec[..., :self.frames_per_window]       # trim to exactly 500 frames
        # Per-instance normalisation
        mean = spec.mean()
        std = spec.std() + 1e-6
        spec = (spec - mean) / std
        return spec
