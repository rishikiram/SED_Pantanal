import pytest
import torch

from src.data.audio_io import pad_or_trim
from src.data.mel_transform import MelTransform
from tests.conftest import SOUNDSCAPE_PATH


# ---------------------------------------------------------------------------
# pad_or_trim — fast, no I/O
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_pad_short_waveform(audio_cfg):
    short = torch.randn(1, 80000)   # half of target
    out = pad_or_trim(short, audio_cfg.samples_per_window)
    assert out.shape == (1, audio_cfg.samples_per_window)
    # Original content preserved, rest is zeros
    assert torch.allclose(out[:, :80000], short)
    assert out[:, 80000:].sum() == 0.0


@pytest.mark.fast
def test_trim_long_waveform(audio_cfg):
    long = torch.randn(1, 320000)   # double the target
    out = pad_or_trim(long, audio_cfg.samples_per_window, deterministic=True)
    assert out.shape == (1, audio_cfg.samples_per_window)
    # Deterministic crop always starts at 0
    assert torch.allclose(out, long[:, :audio_cfg.samples_per_window])


@pytest.mark.fast
def test_exact_length_unchanged(audio_cfg):
    exact = torch.randn(1, audio_cfg.samples_per_window)
    out = pad_or_trim(exact, audio_cfg.samples_per_window)
    assert torch.allclose(out, exact)


@pytest.mark.fast
def test_deterministic_crop_reproducible(audio_cfg):
    wav = torch.randn(1, 320000)
    out1 = pad_or_trim(wav, audio_cfg.samples_per_window, deterministic=True)
    out2 = pad_or_trim(wav, audio_cfg.samples_per_window, deterministic=True)
    assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# MelTransform — fast, uses dummy waveform
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_mel_output_shape(audio_cfg):
    transform = MelTransform(audio_cfg)
    wav = torch.randn(1, audio_cfg.samples_per_window)
    mel = transform(wav)
    assert mel.shape == (1, audio_cfg.n_mels, audio_cfg.frames_per_window)


@pytest.mark.fast
def test_mel_is_normalised(audio_cfg):
    transform = MelTransform(audio_cfg)
    wav = torch.randn(1, audio_cfg.samples_per_window)
    mel = transform(wav)
    # Per-instance normalisation: mean ≈ 0, std ≈ 1
    assert abs(mel.mean().item()) < 0.1
    assert abs(mel.std().item() - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Slow: load a real .ogg file
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_load_real_soundscape(audio_cfg):
    from src.data.audio_io import load_audio
    wav = load_audio(SOUNDSCAPE_PATH, audio_cfg.sample_rate)
    assert wav.shape[0] == 1        # mono
    assert wav.shape[1] > 0


@pytest.mark.slow
def test_mel_shape_on_real_audio(audio_cfg):
    from src.data.audio_io import load_audio
    transform = MelTransform(audio_cfg)
    wav = load_audio(SOUNDSCAPE_PATH, audio_cfg.sample_rate)
    wav = pad_or_trim(wav, audio_cfg.samples_per_window, deterministic=True)
    mel = transform(wav)
    assert mel.shape == (1, 128, 500)
