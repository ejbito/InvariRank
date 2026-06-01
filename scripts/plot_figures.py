from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invarirank.evaluation.plotting import plot_exposure
from invarirank.utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-style figures.")
    parser.add_argument("--ranked-lists", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    plot_exposure(read_json(args.ranked_lists), args.output, k=args.k)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
