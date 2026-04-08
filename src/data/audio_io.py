import torch
import torchaudio


def load_audio(path: str, target_sr: int = 32000) -> torch.Tensor:
    """Load audio file, resample to target_sr, downmix to mono.

    Returns: (1, N) float32 waveform tensor.
    """
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    return waveform


def pad_or_trim(waveform: torch.Tensor, target_samples: int, deterministic: bool = False) -> torch.Tensor:
    """Pad or crop waveform to exactly target_samples.

    waveform: (1, N)
    Returns: (1, target_samples)
    """
    n = waveform.shape[-1]
    if n == target_samples:
        return waveform
    if n < target_samples:
        pad = target_samples - n
        return torch.nn.functional.pad(waveform, (0, pad))
    # Crop
    if deterministic:
        start = 0
    else:
        start = torch.randint(0, n - target_samples + 1, (1,)).item()
    return waveform[:, start: start + target_samples]
