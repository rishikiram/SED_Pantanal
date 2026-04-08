import pytest
import torch

from src.training.losses import FocalBCELoss, compute_class_weights


@pytest.mark.fast
def test_loss_is_scalar(dummy_mel_batch, dummy_labels, model):
    loss_fn = FocalBCELoss()
    logits = model(dummy_mel_batch)         # (B, T', 234)
    loss = loss_fn(logits, dummy_labels)
    assert loss.shape == ()                 # scalar
    assert loss.item() > 0


@pytest.mark.fast
def test_loss_decreases_toward_target():
    """Loss on correct logits should be lower than on random logits."""
    loss_fn = FocalBCELoss()
    targets = torch.zeros(2, 234)
    targets[:, 0] = 1.0

    # Logits strongly predicting the correct class
    good_logits = torch.full((2, 16, 234), -5.0)
    good_logits[:, :, 0] = 5.0

    # Random logits
    random_logits = torch.randn(2, 16, 234)

    good_loss = loss_fn(good_logits, targets).item()
    random_loss = loss_fn(random_logits, targets).item()
    assert good_loss < random_loss


@pytest.mark.fast
def test_loss_broadcasts_targets_across_time():
    """(B, C) targets must be correctly broadcast to (B, T', C) logits."""
    loss_fn = FocalBCELoss()
    B, T, C = 3, 16, 234
    logits = torch.randn(B, T, C)
    targets = torch.zeros(B, C)
    targets[0, 5] = 1.0
    # Should not raise
    loss = loss_fn(logits, targets)
    assert loss.item() > 0


@pytest.mark.fast
def test_class_weights_applied(dummy_labels, model, dummy_mel_batch):
    loss_fn = FocalBCELoss()
    logits = model(dummy_mel_batch)
    weights = torch.ones(234)
    loss_no_weights = loss_fn(logits, dummy_labels).item()
    loss_with_weights = loss_fn(logits, dummy_labels, class_weights=weights).item()
    # Uniform weights of 1.0 should give same result as no weights
    assert abs(loss_no_weights - loss_with_weights) < 1e-5


@pytest.mark.fast
def test_compute_class_weights_shape():
    counts = torch.tensor([100.0] * 234)
    weights = compute_class_weights(counts)
    assert weights.shape == (234,)


@pytest.mark.fast
def test_compute_class_weights_rare_class_gets_higher_weight():
    counts = torch.ones(234) * 100.0
    counts[0] = 1.0     # very rare class
    weights = compute_class_weights(counts)
    assert weights[0] > weights[1]


@pytest.mark.fast
def test_compute_class_weights_zero_count_no_nan():
    counts = torch.zeros(234)
    counts[5] = 10.0
    weights = compute_class_weights(counts)
    assert not torch.isnan(weights).any()
    assert not torch.isinf(weights).any()
