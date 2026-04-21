"""BirdCLEF 2026 — Kaggle notebook submission script.

Mirrors the flow of a Kaggle notebook. Run locally to verify the full
pipeline before pasting into the notebook.

Kaggle dataset setup (three datasets to attach to the notebook):
  1. Competition data  : birdclef-2026           → /kaggle/input/birdclef-2026/
  2. Model checkpoint  : birdclef2026-checkpoints → /kaggle/input/birdclef2026-checkpoints/
  3. Source code       : birdclef2026-source      → /kaggle/input/birdclef2026-source/

The script auto-detects whether it is running inside Kaggle or locally and
sets paths accordingly.  Set CHECKPOINT_NAME below to select which checkpoint to use.
"""

# =============================================================================
# Cell 1 — Imports & path setup
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

IN_KAGGLE = Path('/kaggle/input').exists()

if IN_KAGGLE:
    KAGGLE_SOURCE   = Path('/kaggle/input/models/rishiyang/sed-lemonjar-v1/pytorch/default/1')
    sys.path.insert(0, str(KAGGLE_SOURCE))
    DATA_ROOT       = Path('/kaggle/input/birdclef-2026')
    CHECKPOINT_DIR  = KAGGLE_SOURCE / 'weights'
    OUTPUT_PATH     = Path('/kaggle/working/submission.csv')
else:
    # Local paths — adjust CHECKPOINT_DIR if needed
    _repo = Path(__file__).parent.parent
    sys.path.insert(0, str(_repo))
    DATA_ROOT       = _repo / 'data/birdclef-2026'
    CHECKPOINT_DIR  = _repo / 'outputs/checkpoints'
    OUTPUT_PATH     = _repo / 'outputs/submissions/submission.csv'

CHECKPOINT_NAME     = 'best_fold0.pt' # filename within CHECKPOINT_DIR
CONFIG_PATH         = 'configs/base.yaml' if not IN_KAGGLE else None
SOUNDSCAPE_DIR      = DATA_ROOT / 'test_soundscapes'
SAMPLE_SUBMISSION   = DATA_ROOT / 'sample_submission.csv'
TAXONOMY_CSV        = DATA_ROOT / 'taxonomy.csv'

print(f'Running in Kaggle : {IN_KAGGLE}')
print(f'Data root         : {DATA_ROOT}')
print(f'Checkpoint dir    : {CHECKPOINT_DIR}')
print(f'Output path       : {OUTPUT_PATH}')

# =============================================================================
# Cell 2 — Load config
# =============================================================================
from src.config import load_config, AudioConfig, ModelConfig

if CONFIG_PATH:
    cfg = load_config(CONFIG_PATH)
else:
    # Inline config for Kaggle (matches configs/base.yaml)
    import yaml
    _yaml_str = """
audio:
  sample_rate: 32000
  window_duration: 5.0
  n_fft: 1024
  hop_length: 320
  n_mels: 128
  fmin: 40
  fmax: 15000
  top_db: 80.0

model:
  backbone: efficientnet_b0
  in_chans: 1
  rnn_type: gru
  rnn_hidden_dim: 256
  rnn_num_layers: 2
  rnn_bidirectional: true
  rnn_dropout: 0.3
  classifier_dropout: 0.3
  num_classes: 234

training:
  batch_size: 32
  focal_gamma: 2.0
  seed: 42
"""
    import tempfile, os
    _tmp = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
    _tmp.write(_yaml_str)
    _tmp.close()
    cfg = load_config(_tmp.name)
    os.unlink(_tmp.name)

print('Config loaded.')

# =============================================================================
# Cell 3 — Device & model setup
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

checkpoint = CHECKPOINT_DIR / CHECKPOINT_NAME
if not checkpoint.exists():
    raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
print(f'Checkpoint: {checkpoint.name}')

from src.inference.predictor import Predictor

predictor = Predictor(cfg, str(checkpoint), device)
print('Model loaded.')

# =============================================================================
# Cell 4 — Discover test soundscapes
# =============================================================================
audio_files = sorted(SOUNDSCAPE_DIR.glob('*.ogg'))
if not audio_files:
    raise FileNotFoundError(f'No .ogg files found in {SOUNDSCAPE_DIR}')
print(f'Test soundscapes: {len(audio_files)}')

# =============================================================================
# Cell 5 — Run inference
# =============================================================================
soundscape_probs: dict[str, np.ndarray] = {}

for i, path in enumerate(tqdm(audio_files, desc='Inference')):
    soundscape_probs[path.stem] = predictor.predict(str(path))   # (12, 234)

print(f'Inference complete — {len(soundscape_probs)} soundscapes processed.')

# =============================================================================
# Cell 6 — Build submission, aligned to sample_submission column order
# =============================================================================
from src.inference.postprocess import make_submission
from src.utils.label_encoder import LabelEncoder

enc = LabelEncoder(str(TAXONOMY_CSV))

# Use sample_submission to get the canonical column order
sample_sub = pd.read_csv(SAMPLE_SUBMISSION)
species_cols = [c for c in sample_sub.columns if c != 'row_id']

# Reorder enc.species to match sample_submission (they should match, but be safe)
assert set(species_cols) == set(enc.species), (
    'Species mismatch between taxonomy.csv and sample_submission.csv'
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df = make_submission(soundscape_probs, species_cols, str(OUTPUT_PATH))

# Reorder columns to exactly match sample_submission
df = df[['row_id'] + species_cols]
df.to_csv(str(OUTPUT_PATH), index=False)

print(f'Submission saved : {OUTPUT_PATH}')
print(f'Shape            : {df.shape}')
print(df.head())
