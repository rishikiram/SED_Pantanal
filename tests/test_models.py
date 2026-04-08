import pytest
import torch

from src.models.cnn_backbone import CNNBackbone
from src.models.rnn_head import RNNHead
from src.models.rcnn_sed import Rcnnsed
from src.config import ModelConfig


# ---------------------------------------------------------------------------
# CNNBackbone
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_backbone_output_shape(model, dummy_mel_batch):
    feat = model.backbone(dummy_mel_batch)
    B = dummy_mel_batch.shape[0]
    assert feat.shape[0] == B
    assert feat.shape[1] == model.backbone.out_channels
    # Both spatial dims should be > 1 (global pool disabled)
    assert feat.shape[2] > 1, "Freq dimension collapsed — global_pool may not be disabled"
    assert feat.shape[3] > 1, "Time dimension collapsed — global_pool may not be disabled"
    # assert feat.shape[3] < 1, f"shape is {feat.shape}"


@pytest.mark.fast
def test_backbone_out_channels_consistent(model, dummy_mel_batch):
    feat = model.backbone(dummy_mel_batch)
    assert feat.shape[1] == model.backbone.out_channels


# ---------------------------------------------------------------------------
# RNNHead
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_rnn_head_output_shape():
    C, T_prime, B, num_classes = 1280, 16, 4, 234
    head = RNNHead(input_size=C, num_classes=num_classes)
    x = torch.randn(B, C, T_prime)
    out = head(x)
    assert out.shape == (B, T_prime, num_classes)


@pytest.mark.fast
def test_rnn_head_single_sample():
    head = RNNHead(input_size=64, hidden_dim=32, num_layers=1, num_classes=10)
    x = torch.randn(1, 64, 8)
    out = head(x)
    assert out.shape == (1, 8, 10)


# ---------------------------------------------------------------------------
# Rcnnsed end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_rcnn_output_shape(model, dummy_mel_batch):
    out = model(dummy_mel_batch)
    B = dummy_mel_batch.shape[0]
    assert out.shape[0] == B
    assert out.shape[2] == 234      # num_classes
    assert out.shape[1] > 1        # T' > 1 (multiple time frames)


@pytest.mark.fast
def test_rcnn_output_is_logits(model, dummy_mel_batch):
    # Raw output should NOT be bounded to [0,1] — it's logits
    out = model(dummy_mel_batch)
    assert out.min().item() < 0 or out.max().item() > 1, \
        "Output looks like probabilities — sigmoid should not be inside forward()"


@pytest.mark.fast
def test_rcnn_single_sample(model):
    x = torch.randn(1, 1, 128, 500)
    out = model(x)
    assert out.shape[0] == 1
    assert out.shape[2] == 234


@pytest.mark.fast
def test_rcnn_no_pretrained_weights():
    # Confirm fast_model_cfg uses pretrained=False (no network call needed)
    cfg = ModelConfig(backbone_pretrained=False)
    model = Rcnnsed(cfg)
    x = torch.randn(2, 1, 128, 500)
    out = model(x)
    assert out.shape == (2, out.shape[1], 234)
