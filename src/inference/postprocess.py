from pathlib import Path

import numpy as np
import pandas as pd


def make_submission(
    soundscape_probs: dict[str, np.ndarray],
    species: list[str],
    output_path: str,
):
    """Build submission CSV from per-soundscape probability matrices.

    soundscape_probs: {soundscape_stem: (12, num_classes) array}
    species: ordered list of species column names (from LabelEncoder.species)
    output_path: path to write submission CSV
    """
    rows = []
    for stem, probs in soundscape_probs.items():
        # probs: (12, num_classes)
        for window_idx in range(probs.shape[0]):
            end_sec = (window_idx + 1) * 5
            row_id = f'{stem}_{end_sec}'
            row = {'row_id': row_id}
            row.update(dict(zip(species, probs[window_idx].tolist())))
            rows.append(row)

    df = pd.DataFrame(rows, columns=['row_id'] + species)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
