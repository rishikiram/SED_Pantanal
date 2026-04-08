import pytest
from src.config import load_config, AudioConfig


@pytest.mark.fast
def test_config_loads(cfg):
    assert cfg.audio.sample_rate == 32000
    assert cfg.audio.n_mels == 128
    assert cfg.model.num_classes == 234
    assert cfg.training.seed == 42


@pytest.mark.fast
def test_samples_per_window(audio_cfg):
    # 32000 Hz * 5s = 160000 samples
    assert audio_cfg.samples_per_window == 160000


@pytest.mark.fast
def test_frames_per_window(audio_cfg):
    # 160000 / 320 = 500 frames
    assert audio_cfg.frames_per_window == 500


@pytest.mark.fast
def test_audio_config_defaults():
    cfg = AudioConfig()
    assert cfg.fmin == 40
    assert cfg.fmax == 15000
    assert cfg.top_db == 80.0
