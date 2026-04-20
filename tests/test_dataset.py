import pytest
import torch
import pandas as pd
from torch.utils.data import DataLoader

from tests.conftest import WINDOWS_CSV, TRAIN_AUDIO_ROOT

windows_csv_exists = pytest.mark.skipif(
    not __import__('pathlib').Path(WINDOWS_CSV).exists(),
    reason=f'{WINDOWS_CSV} not found — run scripts/generate_clip_windows.py first',
)


@pytest.mark.slow
@windows_csv_exists
class TestClipDataset:
    @pytest.fixture(scope="class")
    def dataset(self, cfg, encoder):
        from src.data.clip_dataset import ClipDataset
        return ClipDataset(WINDOWS_CSV, TRAIN_AUDIO_ROOT, encoder, cfg.audio, indices=list(range(8)))

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

    def test_different_time_starts_produce_different_mels(self, cfg, encoder):
        """Windows at different offsets within the same file must produce different mels."""
        df = pd.read_csv(WINDOWS_CSV)
        # Find a file that has at least 2 windows
        counts = df.groupby('filename').size()
        multi = counts[counts >= 2].index[0]
        rows = df[df['filename'] == multi].sort_values('time_start')
        idx0 = rows.index[0]
        idx1 = rows.index[1]

        from src.data.clip_dataset import ClipDataset
        ds = ClipDataset(WINDOWS_CSV, TRAIN_AUDIO_ROOT, encoder, cfg.audio,
                         indices=[idx0, idx1])
        mel0, _ = ds[0]
        mel1, _ = ds[1]
        assert not torch.allclose(mel0, mel1), \
            "Two windows at different time_start offsets produced identical mels"


# ---------------------------------------------------------------------------
# train_clip_windows.csv structure
# ---------------------------------------------------------------------------

@pytest.mark.fast
@windows_csv_exists
class TestWindowsCSV:
    @pytest.fixture(scope="class")
    def df(self):
        return pd.read_csv(WINDOWS_CSV)

    def test_required_columns(self, df):
        for col in ('filename', 'primary_label', 'secondary_labels', 'time_start'):
            assert col in df.columns, f"Missing column: {col}"

    def test_time_start_non_negative(self, df):
        assert (df['time_start'] >= 0).all()

    def test_time_start_multiples_of_window(self, df):
        window_sec = 5.0
        remainder = df['time_start'] % window_sec
        assert (remainder.abs() < 1e-6).all(), \
            "time_start values should be multiples of 5s"

    def test_each_source_file_has_at_least_one_window(self, df):
        # Every filename should appear at least once
        assert (df.groupby('filename').size() >= 1).all()

    def test_no_duplicate_windows(self, df):
        dupes = df.duplicated(subset=['filename', 'time_start'])
        assert not dupes.any(), f"{dupes.sum()} duplicate (filename, time_start) pairs"
