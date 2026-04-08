"""Produce a submission CSV for all soundscapes in test_soundscapes/.

Usage:
    python scripts/run_inference.py --config configs/base.yaml \
        --checkpoints outputs/checkpoints/best_fold0.pt [best_fold1.pt ...] \
        --soundscape_dir data/birdclef-2026/test_soundscapes \
        --output outputs/submissions/submission.csv
"""
import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.inference.postprocess import make_submission
from src.inference.predictor import Predictor
from src.utils.label_encoder import LabelEncoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/base.yaml')
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--soundscape_dir', default='data/birdclef-2026/test_soundscapes')
    p.add_argument('--output', default='outputs/submissions/submission.csv')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    enc = LabelEncoder(f'{cfg.paths.data_root}/taxonomy.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    predictor = Predictor(cfg, args.checkpoints, device)

    soundscape_dir = Path(args.soundscape_dir)
    audio_files = sorted(soundscape_dir.glob('*.ogg'))
    print(f'Found {len(audio_files)} soundscapes')

    soundscape_probs = {}
    for i, path in enumerate(audio_files):
        probs = predictor.predict(str(path))   # (12, 234)
        soundscape_probs[path.stem] = probs
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(audio_files)}')

    df = make_submission(soundscape_probs, enc.species, args.output)
    print(f'Submission saved: {args.output}  ({len(df)} rows)')


if __name__ == '__main__':
    main()
