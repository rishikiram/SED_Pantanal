import pandas as pd
import pytest

from src.utils.label_encoder import LabelEncoder
from tests.conftest import TAXONOMY_PATH


@pytest.mark.fast
def test_num_classes(encoder):
    assert encoder.num_classes == 234
    assert len(encoder) == 234


@pytest.mark.fast
def test_encode_decode_roundtrip(encoder):
    for i, label in enumerate(encoder.species):
        assert encoder.encode(label) == i
        assert encoder.decode(i) == label


@pytest.mark.fast
def test_unknown_label_raises(encoder):
    with pytest.raises(KeyError):
        encoder.encode("not_a_real_species")


@pytest.mark.fast
def test_encode_many(encoder):
    labels = encoder.species[:3]
    indices = encoder.encode_many(labels)
    assert indices == [0, 1, 2]


@pytest.mark.slow
def test_order_matches_submission(cfg, encoder):
    sub = pd.read_csv(f"{cfg.paths.data_root}/sample_submission.csv", nrows=0)
    sub_cols = list(sub.columns[1:])
    assert encoder.species == sub_cols, "Label encoder order does not match submission columns"


@pytest.mark.slow
def test_all_train_labels_known(cfg, encoder):
    train = pd.read_csv(f"{cfg.paths.data_root}/train.csv")
    unknown = set(train["primary_label"].astype(str)) - set(encoder.species)
    assert len(unknown) == 0, f"Unknown labels in train.csv: {unknown}"
