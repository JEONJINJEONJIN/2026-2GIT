"""Preflight-check v4 Big Five experiment readiness."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.v4_bigfive_readiness import (
    readiness_has_errors,
    run_readiness_checks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check v4 Big Five readiness.")
    parser.add_argument("--config", default="configs/experiments/v4_bigfive_pilot.yaml")
    parser.add_argument("--vectors_path", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument(
        "--allow_errors",
        action="store_true",
        help="Exit 0 even when readiness errors are found.",
    )
    args = parser.parse_args()

    checks = run_readiness_checks(
        PROJECT_ROOT / args.config,
        PROJECT_ROOT,
        vectors_path=args.vectors_path,
    )
    for check in checks:
        print(f"{check.status.upper():5} {check.name}: {check.details}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["name", "status", "details"])
            writer.writeheader()
            writer.writerows(check.to_dict() for check in checks)

    if readiness_has_errors(checks) and not args.allow_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
