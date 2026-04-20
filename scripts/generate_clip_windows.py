"""Generate train_clip_windows.csv from train.csv.

Each row in the output represents one non-overlapping 5-second window from a clip.
Clips shorter than 5s produce a single row with time_start=0 (will be zero-padded).

Output columns:
    filename        - path relative to train_audio/ (foreign key to train.csv)
    primary_label   - species identifier
    secondary_labels - Python-literal list string
    time_start      - window start in seconds

Usage:
    python scripts/generate_clip_windows.py
    python scripts/generate_clip_windows.py --config configs/base.yaml --output cache/train_clip_windows.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.config import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/base.yaml')
    p.add_argument('--output', default='cache/train_clip_windows.csv')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    train_csv = Path(cfg.paths.data_root) / 'train.csv'
    audio_root = Path(cfg.paths.data_root) / 'train_audio'
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)
    window_sec = cfg.audio.window_duration  # 5.0

    rows = []
    missing = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Scanning clips'):
        audio_path = audio_root / row['filename']
        if not audio_path.exists():
            missing += 1
            continue

        try:
            info = sf.info(str(audio_path))
            duration = info.duration
        except Exception:
            missing += 1
            continue

        # Number of full non-overlapping windows; always emit at least one
        n_windows = max(1, int(duration / window_sec))
        for i in range(n_windows):
            time_start = i * window_sec
            # Don't start a window past the end of the file
            if time_start >= duration:
                break
            rows.append({
                'filename': row['filename'],
                'primary_label': row['primary_label'],
                'secondary_labels': row['secondary_labels'],
                'time_start': time_start,
            })

    out_df = pd.DataFrame(rows, columns=['filename', 'primary_label', 'secondary_labels', 'time_start'])
    out_df.to_csv(output_path, index=False)

    print(f'\nDone.')
    print(f'  Source clips:   {len(df)}')
    print(f'  Missing/errors: {missing}')
    print(f'  Total windows:  {len(out_df)}  (avg {len(out_df) / max(1, len(df) - missing):.1f}x per clip)')
    print(f'  Saved to:       {output_path}')


if __name__ == '__main__':
    main()
