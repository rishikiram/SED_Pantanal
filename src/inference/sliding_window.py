import torch
from src.config import AudioConfig
from src.data.audio_io import load_audio
from src.data.mel_transform import MelTransform


def sliding_window(audio_path: str, cfg: AudioConfig) -> torch.Tensor:
    """Slice a 60s soundscape into 12 non-overlapping 5s mel windows.

    Returns: (12, 1, n_mels, frames_per_window)
    """
    transform = MelTransform(cfg)
    samples = cfg.samples_per_window                     # 160000
    waveform = load_audio(audio_path, cfg.sample_rate)   # (1, N)

    total_samples = waveform.shape[-1]
    num_windows = int(cfg.sample_rate * 60) // samples   # 12 for 60s audio

    windows = []
    for i in range(num_windows):
        start = i * samples
        end = start + samples
        if end <= total_samples:
            chunk = waveform[:, start:end]
        else:
            # Pad last window if audio is shorter than expected
            chunk = waveform[:, start:]
            pad = samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))
        windows.append(transform(chunk))                 # (1, n_mels, T)

    return torch.stack(windows)                          # (12, 1, n_mels, T)
