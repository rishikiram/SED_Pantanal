# RCNN for Sound Event Detection — BirdCLEF 2026 / Pantanal Species Identification

## Task

Multi-label classification: predict probability of presence for 234 species per 5-second window
within 1-minute soundscapes collected in the Pantanal, Mato Grosso do Sul, Brazil.

- Input: audio at 32kHz, `.ogg` format
- Output: `(12, 234)` probability matrix per soundscape (12 windows × 234 species)
- Submission format: row `{filename}_{end_time_seconds}` with 234 probability columns

---

## Data (READ-ONLY at `data/birdclef-2026/`)

```
data -> birdclef-2026/           # symlink, NEVER WRITE HERE
    taxonomy.csv                 # 234 species: primary_label, inat_taxon_id, scientific_name, common_name, class_name
    train.csv                    # ~35,549 clips: primary_label, secondary_labels, filename, ...
    train_audio/                 # 206 species folders (named by inat_taxon_id), .ogg clips inside
    train_soundscapes/           # 10,657 one-minute .ogg soundscapes
    train_soundscapes_labels.csv # ~1,478 rows: filename, start, end, primary_label (;-separated taxon IDs)
    test_soundscapes/            # hidden at competition runtime
    sample_submission.csv        # row_id + 234 species columns
    recording_location.txt       # Pantanal coordinates
```

### Critical data observations

- **Only ~123 soundscapes are labeled** (1,478 label rows ÷ 12 windows = ~123); the remaining ~10,500
  soundscapes are unlabeled and available for pseudo-labeling or self-supervised pretraining.
- Species labels are `inat_taxon_id` integers (e.g. `22961`) or eBird codes (e.g. `ashgre1`),
  semicolon-separated in `primary_label` column of soundscape labels.
- `train_audio/` has 206 folders vs 234 species in taxonomy — some species have no clips at all.
- `secondary_labels` in `train.csv` is a Python-literal list string (e.g. `[]` or `['22961']`).

---

## Directory Structure (all generated files outside `data/birdclef-2026/`)

```
SEDpantanal/
├── data -> birdclef-2026/       # READ-ONLY symlink
│
├── cache/                       # generated, never commit large files
│   ├── mels/                    # precomputed mel spectrograms (.npy or .pt)
│   │   ├── clips/               # one file per clip
│   │   └── soundscapes/         # one file per soundscape (12 windows stacked)
│   └── pseudo_labels/           # generated pseudo-label CSVs for unlabeled soundscapes
│
├── outputs/                     # training artifacts
│   ├── checkpoints/             # model weights per fold
│   ├── predictions/             # val fold predictions (for threshold search)
│   ├── submissions/             # final submission CSVs
│   └── thresholds/              # per-class thresholds per fold (JSON)
│
├── configs/
│   ├── base.yaml                # all default hyperparameters
│   ├── rcnn_gru.yaml            # GRU-based RCNN
│
├── src/
│   ├── config.py                # dataclass-based config loader from YAML
│   ├── data/
│   │   ├── audio_io.py          # torchaudio load, resample, mono downmix
│   │   ├── mel_transform.py     # mel spectrogram + per-instance normalization
│   │   ├── augmentations.py     # waveform + spectrogram augmentations
│   │   ├── soundscape_dataset.py# labeled soundscapes → (12, 1, 128, T) + (12, 234)
│   │   ├── clip_dataset.py      # train_audio clips → (1, 128, T) + (234,)
│   │   ├── combined_dataset.py  # interleaves clip + soundscape loaders
│   │   └── samplers.py          # WeightedRandomSampler for class imbalance
│   ├── models/
│   │   ├── cnn_backbone.py      # timm EfficientNet-B0 window encoder
│   │   ├── rnn_head.py          # bidirectional GRU/LSTM over 12-window sequence
│   │   └── rcnn_sed.py          # top-level: CNN → RNN → Linear(234)
│   ├── training/
│   │   ├── losses.py            # focal BCE with masked window loss
│   │   ├── trainer.py           # train/val loops
│   │   ├── callbacks.py         # checkpointing, early stopping
│   │   └── mixup.py             # waveform Mixup; soundscape-aware CutMix
│   ├── evaluation/
│   │   ├── metrics.py           # ROC-AUC
│   │   └── threshold_search.py  # per-class threshold optimization on val fold
│   ├── inference/
│   │   ├── sliding_window.py    # infer over 1-2s windows, with ~50% overlap
│   │   ├── predictor.py         # batch inference, 5-fold ensemble averaging
│   │   ├── postprocess.py       # aggregate finer predictions into predections for every 5s window (per-class thresholding)
│   │   └── export_onnx.py       # ONNX export + parity validation
│   └── utils/
│       ├── seed.py
│       ├── label_encoder.py     # species ↔ column index; handles both int IDs and eBird codes
│       ├── logging_utils.py     # structured logging, W&B integration
│       └── viz.py               # spectrogram plots, prediction visualizations
│
├── scripts/
│   ├── precompute_mels.py       # cache mel spectrograms to cache/mels/
│   ├── train_folds.py           # 5-fold CV driver
│   ├── pseudo_label.py          # pseudo-label the ~10,500 unlabeled soundscapes
│   ├── evaluate.py              # full PSDS + F1 evaluation on val fold
│   └── run_inference.py         # produce submission CSV for test_soundscapes/
│
├── notebooks/
│   ├── 01_eda.ipynb             # label distribution, species counts, spectrogram inspection
│   ├── 02_augmentation_debug.ipynb
│   ├── 03_model_shapes.ipynb    # verify tensor shapes end-to-end
│   └── 04_metric_analysis.ipynb
│
└── tests/
    ├── test_dataset.py
    ├── test_model.py
    ├── test_metrics.py
    └── test_inference.py
```

---

## Architecture

The model processes an entire 1-minute soundscape as a **sequence of 12 windows**.
Each window is encoded independently by the CNN; the RNN contextualizes across the sequence.

```
Input: (B, 12, 1, 128, 500)        # B soundscapes, 12 windows, (C=1, F=128, T=500)
         │
         ├─ Flatten windows into batch dim: (B×12, 1, 128, 500)
         │
       CNN Backbone                  # EfficientNet-B0, in_chans=1
         │
         └─ (B×12, feature_dim)
              │
              ├─ Reshape to sequence: (B, 12, feature_dim)
              │
            Bidirectional GRU        # 2 layers, hidden=256
              │
              └─ (B, 12, 256)
                   │
                 Dropout + Linear(256, 234)
                   │
                   └─ (B, 12, 234)  # logits → sigmoid at inference
```

Short clips are fed as T=1 sequences through the same forward path — no architecture branching.

### `src/models/cnn_backbone.py`
- `timm.create_model('efficientnet_b0', pretrained=True, in_chans=1, num_classes=0, global_pool='')`
- Adaptive average pool → `(B, feature_dim)` per window
- Alternative: PANNs CNN10 (AudioSet pretrained) — more semantically aligned to wildlife audio

### `src/models/rnn_head.py`
- `nn.GRU(input_size=feature_dim, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)`
- Project from `2×256` → `256` via linear layer
- Bidirectionality helps: context at t=30s resolves ambiguous detections at t=25s

### `src/models/rcnn_sed.py`
- Orchestrates CNN → RNN → classifier
- Sigmoid applied at inference / in loss, not inside `forward`

---

## Data Pipeline

### Audio Parameters (`configs/base.yaml`)

| Parameter | Value | Notes |
|---|---|---|
| `sample_rate` | 32000 | all audio pre-resampled |
| `window_duration` | 2.0s | |
| `n_fft` | 1024 | |
| `hop_length` | 320 | 10ms hop → 500 frames per 5s window |
| `n_mels` | 128 | |
| `fmin` | 40 Hz | |
| `fmax` | 15000 Hz | |
| `top_db` | 80 | |

### `src/data/audio_io.py`
- `load_audio(path, target_sr=32000)`: torchaudio with soundfile backend for `.ogg`
- `pad_or_trim(waveform, target_samples)`: zero-pad or random crop; deterministic in eval mode
- Downmix stereo: `waveform.mean(0, keepdim=True)`

### `src/data/mel_transform.py`
- `torchaudio.transforms.MelSpectrogram` + `AmplitudeToDB`
- Per-instance normalization: `(spec - mean) / (std + eps)` — better than global stats for SED
- Output: `(1, 128, 500)` per window

### `src/data/soundscape_dataset.py`
- Reads `train_soundscapes_labels.csv`; parses semicolon-separated `primary_label` into multi-hot `(234,)` vector per window
- Loads the full 60s waveform; slices into 12 non-overlapping 5s windows
- Returns `mel: (12, 1, 128, 500)`, `labels: (12, 234)`, `mask: (12,)` (1 = annotated window)
- Only ~123 soundscapes have labels — these are the primary supervised training examples

### `src/data/clip_dataset.py`
- Reads `train.csv`; resolves `filename` against `data/birdclef-2026/train_audio/`
- Parses `secondary_labels` from Python-literal string with `ast.literal_eval`
- Returns `mel: (1, 128, 500)`, `labels: (234,)` (primary=1.0, secondary=0.5)

### `src/utils/label_encoder.py`
- Loads `taxonomy.csv`; builds bidirectional map between species identifiers and column indices
- Must handle both integer `inat_taxon_id` (e.g. `22961`) and eBird codes (e.g. `ashgre1`) since
  `train_soundscapes_labels.csv` uses integers and `sample_submission.csv` column headers mix both

### `src/data/augmentations.py`

**Waveform-level (before mel):**
- `BackgroundNoiseMix(snr_db_range=(5, 30), p=0.5)` — mix segments from the ~10,500 unlabeled soundscapes
- `GainJitter(min_gain_db=-6, max_gain_db=6, p=0.3)`
- `TimeShift(max_shift_sec=1.0, p=0.3)`
- `Mixup(alpha=0.4, p=0.3)` — blend waveforms + labels; for soundscapes, blend matching window indices

**Spectrogram-level (after mel):**
- `SpecAugment(time_mask_max=50, freq_mask_max=15, n_time_masks=2, n_freq_masks=2, p=0.5)`
- `CutMix(p=0.3)` — clips only; not applied to soundscape sequences (disrupts temporal structure)

**Soundscape-specific:**
- `WindowDropout(p_window=0.1)` — zero out a random window + mask it from loss

---

## Training Strategy

### Two-Phase Training

**Phase A — Clip pre-training (5 epochs):**
- Train CNN backbone only on `ClipDataset` (T=1 sequences)
- RNN weights not updated
- Critical because only ~123 labeled soundscapes exist; clips provide most of the
  species-discriminative signal, especially for species with no soundscape annotations
- LR: `1e-3` (head), `1e-4` (backbone)

**Phase B — Full RCNN fine-tuning (10–15 epochs):**
- Alternate batches: soundscapes (full RCNN, window-level loss) and clips (T=1, 3:1 ratio)
- Differential LRs: backbone at `1e-5`, RNN+head at `1e-4`
- Optimizer: `AdamW(weight_decay=1e-2)`
- Scheduler: `CosineAnnealingWarmRestarts(T_0=5, T_mult=2)`
- Mixed precision: `torch.cuda.amp.GradScaler`
- Gradient clipping: `clip_grad_norm_(params, max_norm=5.0)` — essential for RNN stability

### Pseudo-labeling the Unlabeled Soundscapes (`scripts/pseudo_label.py`)
- After Phase B converges, run inference on the ~10,500 unlabeled soundscapes
- Keep predictions with max confidence > 0.8 as soft pseudo-labels
- Add to Phase B training with reduced loss weight (0.5×)
- This is the primary strategy for leveraging the large unlabeled soundscape pool

### Loss: `src/training/losses.py`

**Focal BCE (primary):**
```
FocalBCELoss(alpha=0.25, gamma=2.0)
# Per (window, species) element; masked by annotation mask before reduction
loss = (focal_bce * class_weights * mask.unsqueeze(-1)).mean()
```

Class weights: `weight_c = (N / (num_classes * count_c))^0.5`

### Class Imbalance — Three Layers
1. `WeightedRandomSampler`: per-sample weight = inverse sqrt of rarest label frequency
2. Focal loss: suppresses easy negatives (most of 234 labels are 0 per window)
3. Augmentation multiplier: 3× for species with < 15 clips (some species have 0 clips)

### Cross-Validation (`scripts/train_folds.py`)
- 5-fold stratified CV on the ~123 labeled soundscapes (split at soundscape level, never split windows)
- With only ~123 labeled soundscapes, folds will be small (~25 val soundscapes each) —
  report mean ± std PSDS across folds; final submission uses all 5 fold models ensembled

---

## Evaluation

### Metrics (`src/evaluation/metrics.py`)

| Metric | When | Purpose |
|---|---|---|
| Segment-F1 @ 0.5 threshold | Every epoch | Fast training signal, checkpoint criterion |
| PSDS1 | End of each fold | Onset-focused, penalizes false alarms |
| PSDS2 | End of each fold | Class-weighted, sensitive to rare species |
| Event-based F1 | Final model only | Onset/offset accuracy |

**PSDS is the primary model selection criterion.**

### `src/evaluation/threshold_search.py`
- Post-training grid search [0.1, 0.9] step 0.05 per species on val fold
- Thresholds saved to `outputs/thresholds/thresholds_fold{k}.json`

---

## Inference Pipeline

### `src/inference/sliding_window.py`
- 12 non-overlapping 5s windows from 60s audio: `[0–5, 5–10, ..., 55–60]`
- Optional: 2.5s stride (24 windows, averaged) for smoother boundaries

### `src/inference/predictor.py`
```python
def predict_soundscape(audio_path) -> np.ndarray:
    # Returns: (12, 234) probability matrix
    windows = sliding_window(audio_path)      # (12, 1, 128, 500)
    mel_seq = stack(windows).unsqueeze(0)     # (1, 12, 1, 128, 500)
    logits = model(mel_seq)                   # (1, 12, 234)
    return torch.sigmoid(logits).squeeze(0).cpu().numpy()
```
- Ensemble: average sigmoid probabilities across 5 fold models before thresholding

### `src/inference/postprocess.py`
- Apply per-class thresholds from `outputs/thresholds/`
- **No temporal smoothing** — RNN already smooths; additional smoothing hurts event-based F1
- Output: `outputs/submissions/submission.csv` with `row_id` = `{filename_stem}_{end_time_sec}`

### Submission format (matches `sample_submission.csv`)
- Row IDs: `BC2026_Test_0001_S05_20250227_010002_5` (filename stem + `_` + end second)
- 234 columns matching taxonomy order

---

## Configuration (`src/config.py`)

```python
@dataclass
class PathConfig:
    data_root: str = 'data/birdclef-2026'   # READ-ONLY
    cache_dir: str = 'cache'
    output_dir: str = 'outputs'

@dataclass
class AudioConfig:
    sample_rate: int = 32000
    window_duration: float = 5.0
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 128
    fmin: int = 40
    fmax: int = 15000
    top_db: float = 80.0

@dataclass
class ModelConfig:
    backbone: str = 'efficientnet_b0'   # or 'cnn10_panns'
    backbone_pretrained: bool = True
    in_chans: int = 1
    rnn_type: str = 'gru'               # 'gru' | 'lstm' | 'attention'
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 2
    rnn_bidirectional: bool = True
    rnn_dropout: float = 0.3
    classifier_dropout: float = 0.3
    num_classes: int = 234

@dataclass
class TrainingConfig:
    batch_size: int = 8                 # soundscape batches (8 × 12 = 96 windows)
    clip_batch_size: int = 32
    soundscape_to_clip_ratio: int = 3
    pretrain_clip_epochs: int = 5
    num_epochs: int = 15
    lr: float = 1e-4
    backbone_lr_multiplier: float = 0.1
    weight_decay: float = 1e-2
    loss: str = 'focal_bce'
    focal_gamma: float = 2.0
    grad_clip_norm: float = 5.0
    num_folds: int = 5
    seed: int = 42
```

---

## Key Design Decisions

### 1. Only ~123 labeled soundscapes — clip pre-training is essential
The ~35K clips in `train_audio/` are the primary source of species-level supervision.
Without Phase A clip pre-training, the CNN backbone will not learn species-discriminative
features before temporal training begins. This is more critical here than in typical SED tasks.

### 2. Pseudo-labeling the ~10,500 unlabeled soundscapes
This is the single highest-leverage improvement available. The unlabeled soundscapes are
recorded in the same Pantanal environment and already serve as background noise sources
for augmentation. Using them for pseudo-labeling (after a first model converges) can
multiply the effective labeled soundscape count by ~85×.

### 3. CNN backbone: EfficientNet-B0 vs PANNs CNN10
Start with EfficientNet-B0 (ImageNet pretrained, fast iteration).
Switch to PANNs CNN10 (AudioSet pretrained) if PSDS plateaus — PANNs features are more
semantically aligned to wildlife audio. `in_chans=1` preferred over 3-channel replication.

### 4. No postprocessing temporal smoothing
The RNN learns implicit smoothing. Additional smoothing post-RNN blurs onset/offset boundaries
and hurts event-based F1. Only apply smoothing for CNN-only baseline comparisons.

### 5. Label encoding: integers vs eBird codes
`train_soundscapes_labels.csv` uses integer `inat_taxon_id`; `sample_submission.csv` column
headers mix integers and eBird codes. `label_encoder.py` must map both to the same column
index using `taxonomy.csv` as the source of truth.

### 6. 5-fold CV with only ~123 labeled soundscapes
Each val fold is ~25 soundscapes — PSDS estimates will have high variance.
Report mean ± std across folds; use all 5 models for the final ensemble.

---

## Implementation Order

1. `src/config.py` + `configs/base.yaml`
2. `src/utils/label_encoder.py` — verify all 234 species map correctly from both ID formats
3. `src/data/audio_io.py` + `src/data/mel_transform.py` → shape `(1, 128, 500)` on a real clip
4. `src/data/soundscape_dataset.py` → shapes `(12, 1, 128, 500)`, `(12, 234)`, `(12,)` mask
5. `src/data/clip_dataset.py` → shapes `(1, 128, 500)`, `(234,)`
6. `src/models/cnn_backbone.py` → verify `(B, feature_dim)` output
7. `src/models/rnn_head.py` → verify `(B, 12, 256)` output
8. `src/models/rcnn_sed.py` → end-to-end: `(B, 12, 1, 128, 500)` → `(B, 12, 234)`
9. `src/training/losses.py` → masked focal BCE on dummy data
10. `src/training/trainer.py` → smoke train 1 epoch on the ~123 labeled soundscapes
11. `src/evaluation/metrics.py` → segment-F1 + PSDS on dummy predictions
12. `src/inference/predictor.py` + `src/inference/sliding_window.py`
13. `scripts/train_folds.py` → full 5-fold run
14. `scripts/pseudo_label.py` → generate pseudo-labels for unlabeled soundscapes
15. `src/inference/export_onnx.py` → ONNX export + parity check
