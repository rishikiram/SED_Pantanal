import numpy as np
import pytest
import torch

from src.inference.postprocess import make_submission
from tests.conftest import SOUNDSCAPE_PATH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_checkpoint(tmp_path, fast_model_cfg):
    """Save an untrained model in the current checkpoint format to a temp file."""
    from src.models.rcnn_sed import Rcnnsed
    model = Rcnnsed(fast_model_cfg)
    path = str(tmp_path / 'test_checkpoint.pt')
    torch.save({'epoch': 0, 'model': model.state_dict(),
                'optimizer': {}, 'scheduler': {}, 'scaler': {}, 'best_f1': 0.0}, path)
    return path


# ---------------------------------------------------------------------------
# Predictor — fast (temp checkpoint, no audio I/O)
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_predictor_init(cfg, temp_checkpoint):
    from src.inference.predictor import Predictor
    device = torch.device('cpu')
    p = Predictor(cfg, temp_checkpoint, device)
    assert not p.model.training, "Model should be in eval mode after Predictor.__init__"


@pytest.mark.slow
def test_predictor_output_shape(cfg, temp_checkpoint):
    from src.inference.predictor import Predictor
    device = torch.device('cpu')
    p = Predictor(cfg, temp_checkpoint, device)
    probs = p.predict(SOUNDSCAPE_PATH)
    assert isinstance(probs, np.ndarray)
    assert probs.ndim == 2
    assert probs.shape[1] == cfg.model.num_classes
    assert probs.shape[0] >= 1   # at least one window


@pytest.mark.slow
def test_predictor_output_is_probabilities(cfg, temp_checkpoint):
    from src.inference.predictor import Predictor
    p = Predictor(cfg, temp_checkpoint, torch.device('cpu'))
    probs = p.predict(SOUNDSCAPE_PATH)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


@pytest.mark.slow
def test_predictor_is_deterministic(cfg, temp_checkpoint):
    from src.inference.predictor import Predictor
    p = Predictor(cfg, temp_checkpoint, torch.device('cpu'))
    probs1 = p.predict(SOUNDSCAPE_PATH)
    probs2 = p.predict(SOUNDSCAPE_PATH)
    np.testing.assert_array_equal(probs1, probs2)


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
