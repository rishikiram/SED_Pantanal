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

- **Only ~123 soundscapes are labeled** (1,478 label rows ÷ 12 windows = ~123); the remaining ~10,500 soundscapes are unlabeled.
- Species labels are `inat_taxon_id` integers (e.g. `22961`) or eBird codes (e.g. `ashgre1`), semicolon-separated in `primary_label` column of soundscape labels.
- `train_audio/` has 206 folders vs 234 species in taxonomy — some species have no clips.
- `secondary_labels` in `train.csv` is a Python-literal list string (e.g. `[]` or `['22961']`).

---

## Directory Structure

```
SED_Pantanal/
├── data -> birdclef-2026/       # READ-ONLY symlink
│
├── cache/                       # generated, never commit large files
│   └── mels/                    # precomputed mel spectrograms
│
├── outputs/                     # training artifacts
│   ├── checkpoints/             # model weights (best_fold0.pt, epoch{n}_fold0.pt)
│   ├── train_results.json       # training metrics summary
│   ├── predictions/             # val fold predictions (for threshold search)
│   ├── submissions/             # final submission CSVs
│   └── thresholds/              # per-class thresholds per fold (JSON)
│
├── configs/
│   └── base.yaml                # all default hyperparameters
│
├── src/
│   ├── config.py                # dataclass-based config loader from YAML
│   ├── data/
│   │   ├── audio_io.py          # torchaudio load, resample, mono downmix
│   │   ├── mel_transform.py     # mel spectrogram + per-instance normalization
│   │   └── clip_dataset.py      # train_audio clips → (1, 128, 500) + (234,)
│   ├── models/
│   │   ├── cnn_backbone.py      # timm EfficientNet-B0 window encoder
│   │   ├── rnn_head.py          # bidirectional GRU over T' time frames → (B, T', 234)
│   │   └── rcnn_sed.py          # top-level: CNN → freq-pool → GRU → Linear(234)
│   ├── training/
│   │   ├── losses.py            # focal BCE with class weighting
│   │   └── trainer.py           # train/val loops
│   ├── evaluation/
│   │   └── metrics.py           # segment-F1
│   ├── inference/
│   │   ├── transform_and_slide_window.py    # slice 60s audio into 12 non-overlapping 5s windows
│   │   ├── predictor.py         # batch inference, 5-fold ensemble averaging
│   │   └── postprocess.py       # per-class thresholding → submission CSV
│   └── utils/
│       └── label_encoder.py     # species ↔ column index; handles both int IDs and eBird codes
│
├── scripts/
│   ├── train.py                 # training driver (currently single fold / clip pre-training)
│   ├── evaluate_val.py          # F1 evaluation on val fold
│   └── run_inference.py         # produce submission CSV for test_soundscapes/
│
└── tests/
    ├── conftest.py
    ├── test_audio.py
    ├── test_config.py
    ├── test_dataset.py
    ├── test_inference.py
    ├── test_label_encoder.py
    ├── test_losses.py
    ├── test_metrics.py
    └── test_models.py
```

---

## Architecture

Input: `(B, 1, 128, 500)` mel spectrogram per 5s window.

```
CNN Backbone (EfficientNet-B0, in_chans=1, global_pool='')
    → (B, C, H', T')
Freq-pool: x.mean(dim=-2)
    → (B, C, T')
Permute → (B, T', C)
Bidirectional GRU (2 layers, hidden=256)
    → (B, T', 512)
Dropout + Linear(512, 234)
    → (B, T', 234)   # frame-level logits → sigmoid at inference
```

Frame-level outputs are mean-pooled to a single window-level prediction at inference.
Clips and soundscape windows share the exact same forward path and input shape.

---

## Data Pipeline

**Audio parameters** (`configs/base.yaml`): 32kHz, 5s windows, n_fft=1024, hop=320 (10ms → 500 frames/window), n_mels=128, fmin=40Hz, fmax=15kHz, top_db=80.

**`src/data/audio_io.py`**: torchaudio load + resample, zero-pad or crop to target length, mono downmix.

**`src/data/mel_transform.py`**: `MelSpectrogram` + `AmplitudeToDB` + per-instance normalization → `(1, 128, 500)`.

**`src/data/clip_dataset.py`**: reads `train.csv`, parses `secondary_labels` via `ast.literal_eval`, returns `mel: (1, 128, 500)`, `labels: (234,)` (primary=1.0, secondary=0.5). Currently uses only the first 5s of each clip (see Phase 1.1 fix).

**`src/utils/label_encoder.py`**: loads `taxonomy.csv`, maps both `inat_taxon_id` integers and eBird codes to the same column index.

---

## Training (Current State)

Phase A (clip pre-training) is implemented in `scripts/train.py`. After 5 epochs on `ClipDataset`, best val F1 ≈ 0.00085 — very low, reflecting the Phase 1.1 clip sampling bug and lack of soundscape fine-tuning.

**Loss**: Focal BCE (`alpha=0.25, gamma=2.0`), class weights = `sqrt(N / (C * count_c))`, broadcast over `(B, T', 234)`.

**Optimizer**: AdamW, differential LRs (backbone × 0.1), cosine LR schedule, grad clip norm=5.0, mixed precision.

**Class imbalance**: `WeightedRandomSampler` (per-sample weight = inverse sqrt of rarest label frequency) + focal loss.

---

## Evaluation

**`src/evaluation/metrics.py`**: segment-F1 @ 0.5 threshold (used as checkpoint criterion during training).

**PSDS1/PSDS2** are planned but not yet implemented.

---

## Inference Pipeline

**`src/inference/transform_and_slide_window.py`**: slices 60s audio into 12 non-overlapping 5s windows.

**`src/inference/predictor.py`**: `predict_soundscape(path) → (12, 234)` — stacks windows, runs model, mean-pools frame-level sigmoid outputs.

**`src/inference/postprocess.py`**: applies per-class thresholds → `outputs/submissions/submission.csv`.

---

## Key Design Decisions

1. **Clip pre-training is essential** — only ~123 labeled soundscapes exist; ~35K clips provide most species-discriminative signal.
2. **Pseudo-labeling** — after Phase B converges, run inference on ~10,500 unlabeled soundscapes; keep predictions >0.8 confidence as soft labels (0.5× loss weight). Highest-leverage improvement available.
3. **No temporal smoothing** — additional smoothing blurs onset/offset boundaries and hurts event-based F1.
4. **Label encoding** — `train_soundscapes_labels.csv` uses `inat_taxon_id` integers; `sample_submission.csv` uses eBird codes. `label_encoder.py` maps both via `taxonomy.csv`.
5. **5-fold CV** — split at soundscape level (~25 val soundscapes per fold); high variance with only ~123 total. Ensemble all 5 models for submission.

---

## To Implement

### should the mean(sigmoid(logits[T'])) across dim=1 (batch, T', species) be the probability of presence?

### Phase 2 — Soundscape Fine-tuning + CV

Goal: Phase B fine-tuning on labeled soundscape windows; proper cross-validated evaluation.

- `src/data/soundscape_dataset.py` → shapes `(1, 128, 500)`, `(234,)` per labeled window
- `src/data/combined_dataset.py` + `src/data/samplers.py` — interleave clips and soundscape windows
- Update `scripts/train.py` → Phase B fine-tuning loop (3:1 clip ratio, backbone lr=1e-5, head lr=1e-4)
- `src/evaluation/metrics.py` → add PSDS1/PSDS2 alongside segment-F1
- `src/evaluation/threshold_search.py` → per-class threshold grid search [0.1, 0.9] step 0.05 on val fold
- `scripts/train_folds.py` → full 5-fold run

### Phase 3 — Augmentation + Pseudo-labeling

- `src/data/augmentations.py` — waveform augmentations (BackgroundNoiseMix, GainJitter, TimeShift) + spectrogram augmentations (SpecAugment, CutMix)
- `scripts/pseudo_label.py` → generate pseudo-labels for ~10,500 unlabeled soundscapes
