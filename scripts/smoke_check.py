from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invarirank.config import load_config
from invarirank.smoke import smoke_run_model, smoke_validate_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny InvariRank smoke check.")
    parser.add_argument("--config", default="configs/dev/smoke.yaml")
    parser.add_argument(
        "--run-model",
        action="store_true",
        help="Also load the configured HF model and run one ranking pass.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    info = smoke_validate_preflight(cfg)
    print("Preflight smoke check passed:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    if args.run_model:
        records = smoke_run_model(cfg)
        print(f"Model smoke check passed: wrote {len(records)} ranked-list record(s).")


if __name__ == "__main__":
    main()
