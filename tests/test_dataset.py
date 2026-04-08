import pytest
import torch
from torch.utils.data import DataLoader

from tests.conftest import TRAIN_CSV, TRAIN_AUDIO_ROOT


@pytest.mark.slow
class TestClipDataset:
    @pytest.fixture(scope="class")
    def dataset(self, cfg, encoder):
        from src.data.clip_dataset import ClipDataset
        return ClipDataset(TRAIN_CSV, TRAIN_AUDIO_ROOT, encoder, cfg.audio, indices=list(range(8)))

    def test_length(self, dataset):
        assert len(dataset) == 8

    def test_mel_shape(self, dataset):
        mel, _ = dataset[0]
        assert mel.shape == (1, 128, 500)

    def test_label_shape(self, dataset, encoder):
        _, label = dataset[0]
        assert label.shape == (encoder.num_classes,)

    def test_label_dtype(self, dataset):
        _, label = dataset[0]
        assert label.dtype == torch.float32

    def test_primary_label_is_one(self, dataset):
        _, label = dataset[0]
        # At least one class must be 1.0 (primary label)
        assert (label == 1.0).any()

    def test_label_values_valid(self, dataset):
        # Only 0.0, 0.5, or 1.0 are valid label values
        for i in range(len(dataset)):
            _, label = dataset[i]
            unique = label.unique()
            for v in unique:
                assert v.item() in {0.0, 0.5, 1.0}, f"Unexpected label value: {v.item()}"

    def test_dataloader_batch_shape(self, dataset, encoder):
        loader = DataLoader(dataset, batch_size=4)
        mels, labels = next(iter(loader))
        assert mels.shape == (4, 1, 128, 500)
        assert labels.shape == (4, encoder.num_classes)
