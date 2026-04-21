import numpy as np
import pytest
import torch

from src.inference.postprocess import make_submission
from tests.conftest import SOUNDSCAPE_PATH


# ---------------------------------------------------------------------------
# transform_and_slide_window — slow (loads real audio)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_transform_and_slide_window_shape(cfg):
    from src.inference.transform_and_slide_window import transform_and_slide_window
    windows = transform_and_slide_window(SOUNDSCAPE_PATH, cfg.audio)
    assert windows.shape == (12, 1, 128, 500)


@pytest.mark.slow
def test_transform_and_slide_window_dtype(cfg):
    from src.inference.transform_and_slide_window import transform_and_slide_window
    windows = transform_and_slide_window(SOUNDSCAPE_PATH, cfg.audio)
    assert windows.dtype == torch.float32


@pytest.mark.slow
def test_transform_and_slide_windows_are_normalised(cfg):
    from src.inference.transform_and_slide_window import transform_and_slide_window
    windows = transform_and_slide_window(SOUNDSCAPE_PATH, cfg.audio)
    # Each window is per-instance normalised — mean should be near 0
    for i in range(windows.shape[0]):
        mean = windows[i].mean().item()
        assert abs(mean) < 0.5, f"Window {i} mean={mean:.3f} — normalisation may have failed"


# ---------------------------------------------------------------------------
# make_submission — fast
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_submission_row_count(encoder):
    probs = {
        "BC2026_Test_0001_S05_20250227_010002": np.random.rand(12, 234),
        "BC2026_Test_0002_S05_20250227_010002": np.random.rand(12, 234),
    }
    df = make_submission(probs, encoder.species, "/tmp/test_submission.csv")
    assert len(df) == 24    # 2 soundscapes × 12 windows


@pytest.mark.fast
def test_submission_column_count(encoder):
    probs = {"stem": np.random.rand(12, 234)}
    df = make_submission(probs, encoder.species, "/tmp/test_submission.csv")
    assert len(df.columns) == 235   # row_id + 234 species


@pytest.mark.fast
def test_submission_row_id_format(encoder):
    stem = "BC2026_Test_0001_S05_20250227_010002"
    probs = {stem: np.random.rand(12, 234)}
    df = make_submission(probs, encoder.species, "/tmp/test_submission.csv")
    expected_ids = [f"{stem}_{(i+1)*5}" for i in range(12)]
    assert list(df["row_id"]) == expected_ids


@pytest.mark.fast
def test_submission_column_order_matches_encoder(encoder):
    probs = {"stem": np.random.rand(12, 234)}
    df = make_submission(probs, encoder.species, "/tmp/test_submission.csv")
    assert list(df.columns[1:]) == encoder.species


@pytest.mark.fast
def test_submission_probs_in_range(encoder):
    probs = {"stem": np.random.rand(12, 234)}
    df = make_submission(probs, encoder.species, "/tmp/test_submission.csv")
    values = df[encoder.species].values
    assert values.min() >= 0.0
    assert values.max() <= 1.0
