import numpy as np
import pytest

from src.evaluation.metrics import segment_f1


@pytest.mark.fast
def test_perfect_predictions():
    targets = np.zeros((10, 234))
    targets[:, 5] = 1.0
    targets[:, 42] = 1.0
    probs = targets.copy()      # perfect confidence
    assert segment_f1(probs, targets) == pytest.approx(1.0)


@pytest.mark.fast
def test_all_wrong_predictions():
    targets = np.zeros((10, 234))
    targets[:, 5] = 1.0
    probs = np.zeros((10, 234))
    probs[:, 99] = 1.0          # always predicts wrong class
    score = segment_f1(probs, targets)
    assert score == pytest.approx(0.0)


@pytest.mark.fast
def test_no_active_classes_returns_zero():
    targets = np.zeros((10, 234))   # no positives at all
    probs = np.random.rand(10, 234)
    assert segment_f1(probs, targets) == 0.0


@pytest.mark.fast
def test_output_in_range():
    targets = (np.random.rand(20, 234) > 0.9).astype(float)
    probs = np.random.rand(20, 234)
    score = segment_f1(probs, targets)
    assert 0.0 <= score <= 1.0


@pytest.mark.fast
def test_threshold_matters():
    targets = np.zeros((5, 234))
    targets[:, 0] = 1.0
    probs = np.zeros((5, 234))
    probs[:, 0] = 0.4           # below default 0.5 threshold

    score_default = segment_f1(probs, targets, threshold=0.5)
    score_lower = segment_f1(probs, targets, threshold=0.3)
    assert score_default == pytest.approx(0.0)
    assert score_lower == pytest.approx(1.0)
