"""Package src/, configs/base.yaml, and optionally a model checkpoint into a zip for Kaggle.

Usage:
    python scripts/pack_for_kaggle.py
    python scripts/pack_for_kaggle.py --model_weights outputs/checkpoints/best_fold0.pt
    python scripts/pack_for_kaggle.py --model_weights outputs/checkpoints/best_fold0.pt --note "epoch 5, lr 1e-4"
    python scripts/pack_for_kaggle.py --model_weights outputs/checkpoints/best_fold0.pt --output my.zip
"""
import argparse
import datetime
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='birdclef2026-source.zip')
    p.add_argument('--model_weights', default=None, metavar='PATH',
                   help='Checkpoint to include (e.g. outputs/checkpoints/best_fold0.pt)')
    p.add_argument('--note', default=None, metavar='TEXT',
                   help='Optional note written into pack_log.txt inside the zip')
    return p.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)

    includes = [
        *ROOT.rglob('src/**/*.py'),
        ROOT / 'configs/base.yaml',
    ]
    includes = [p for p in includes if '__pycache__' not in p.parts]

    weights_path = None
    if args.model_weights:
        weights_path = Path(args.model_weights)
        if not weights_path.exists():
            raise FileNotFoundError(f'Model weights not found: {weights_path}')

    timestamp = datetime.datetime.now().isoformat(timespec='seconds')
    log_lines = [f'packed: {timestamp}']
    if weights_path:
        log_lines.append(f'weights: {weights_path.name}')
    if args.note:
        log_lines.append(f'note: {args.note}')
    log_content = '\n'.join(log_lines) + '\n'

    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(includes):
            arcname = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name
            zf.write(path, arcname)
        if weights_path:
            zf.write(weights_path, Path('weights') / weights_path.name)
        zf.writestr('pack_log.txt', log_content)

    total = len(includes) + (1 if weights_path else 0) + 1  # +1 for pack_log.txt
    print(f'Created {output}  ({total} files)')
    for path in sorted(includes):
        arcname = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name
        print(f'  {arcname}')
    if weights_path:
        print(f'  weights/{weights_path.name}')
    print(f'  pack_log.txt')
    print(f'\n{log_content}', end='')


if __name__ == '__main__':
    main()
