"""Generate demonstration CSV: python scripts/generate_data.py [--output PATH]."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import SYNTHETIC_DATA_PATH
from data.synthetic import generate_synthetic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SYNTHETIC_DATA_PATH)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--experiments", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = generate_synthetic(args.rows, args.experiments, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} synthetic observations / {frame.experiment_id.nunique()} experiments to {args.output}")


if __name__ == "__main__":
    main()
