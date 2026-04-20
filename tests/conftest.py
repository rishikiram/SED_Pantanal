"""Shared fixtures for all tests.

Fast fixtures use dummy tensors — no disk I/O.
Slow fixtures (marked) load real audio or pretrained weights.
"""
import numpy as np
import pytest
import torch

from src.config import AudioConfig, Config, ModelConfig, PathConfig, TrainingConfig, load_config
from src.models.rcnn_sed import Rcnnsed
from src.utils.label_encoder import LabelEncoder

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that load real audio or pretrained weights")
    config.addinivalue_line("markers", "fast: tests that use only dummy data")


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config("configs/base.yaml")


@pytest.fixture(scope="session")
def audio_cfg(cfg) -> AudioConfig:
    return cfg.audio


@pytest.fixture(scope="session")
def model_cfg(cfg) -> ModelConfig:
    return cfg.model


# ---------------------------------------------------------------------------
# Label encoder
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def encoder(cfg) -> LabelEncoder:
    return LabelEncoder(f"{cfg.paths.data_root}/taxonomy.csv")


# ---------------------------------------------------------------------------
# Dummy tensor fixtures (no I/O)
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_mel() -> torch.Tensor:
    """Single mel window: (1, 128, 500)."""
    return torch.randn(1, 128, 500)


@pytest.fixture
def dummy_mel_batch() -> torch.Tensor:
    """Batch of mel windows: (4, 1, 128, 500)."""
    return torch.randn(4, 1, 128, 500)


@pytest.fixture
def dummy_labels() -> torch.Tensor:
    """Batch of multi-hot label vectors: (4, 234)."""
    labels = torch.zeros(4, 234)
    labels[0, 5] = 1.0
    labels[1, 10] = 1.0
    labels[1, 20] = 0.5
    return labels


# ---------------------------------------------------------------------------
# Model fixture (untrained, no pretrained weights — fast)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fast_model_cfg() -> ModelConfig:
    """ModelConfig with pretrained=False to avoid network download."""
    return ModelConfig(backbone_pretrained=False)


@pytest.fixture(scope="session")
def model(fast_model_cfg) -> Rcnnsed:
    return Rcnnsed(fast_model_cfg)


# ---------------------------------------------------------------------------
# Real-data paths (used only by slow tests)
# ---------------------------------------------------------------------------

TAXONOMY_PATH = "data/birdclef-2026/taxonomy.csv"
TRAIN_CSV = "data/birdclef-2026/train.csv"
TRAIN_AUDIO_ROOT = "data/birdclef-2026/train_audio"
WINDOWS_CSV = "cache/train_clip_windows.csv"
SOUNDSCAPE_PATH = "data/birdclef-2026/train_soundscapes/BC2026_Train_0001_S08_20250606_030007.ogg"
