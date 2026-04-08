"""
Maps species identifiers to column indices and back.

taxonomy.csv primary_label IS the submission column name:
  - eBird code for birds (e.g. 'brnowl')
  - iNaturalist taxon ID for non-birds (e.g. '116570')
  - Sonotype code for unidentified insects (e.g. '47158son01')

The 234 rows of taxonomy.csv correspond 1:1 to the 234 submission columns in order.
train.csv and train_soundscapes_labels.csv primary_label uses the same codes.
"""
import pandas as pd


class LabelEncoder:
    def __init__(self, taxonomy_path: str):
        df = pd.read_csv(taxonomy_path)
        # primary_label is the submission column name and the canonical species identifier
        self.species = [str(x) for x in df['primary_label'].tolist()]
        self.num_classes = len(self.species)
        self._label_to_idx = {label: i for i, label in enumerate(self.species)}

    def encode(self, label: str) -> int:
        """Return column index for a species identifier."""
        key = str(label).strip()
        if key not in self._label_to_idx:
            raise KeyError(f"Unknown species label: {key!r}")
        return self._label_to_idx[key]

    def decode(self, idx: int) -> str:
        return self.species[idx]

    def encode_many(self, labels: list[str]) -> list[int]:
        return [self.encode(lbl) for lbl in labels]

    def __len__(self) -> int:
        return self.num_classes
