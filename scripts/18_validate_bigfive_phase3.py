"""Validate v4 Big Five Phase 3 final-run readiness."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.v4_bigfive_phase3_validation import (  # noqa: E402
    run_phase3_validation,
    validation_has_errors,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v4 Big Five Phase 3.")
    parser.add_argument("--config", default="configs/experiments/v4_bigfive_final.yaml")
    parser.add_argument("--vectors_path", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument(
        "--allow_missing_external",
        action="store_true",
        help="Downgrade known missing external data/model artifacts to warnings.",
    )
    parser.add_argument(
        "--allow_missing_tuning",
        action="store_true",
        help="Downgrade missing final alpha/layers to warnings.",
    )
    parser.add_argument(
        "--allow_errors",
        action="store_true",
        help="Exit 0 even when validation errors are present.",
    )
    args = parser.parse_args()

    checks = run_phase3_validation(
        PROJECT_ROOT,
        PROJECT_ROOT / args.config,
        vectors_path=args.vectors_path,
        allow_missing_external=args.allow_missing_external,
        allow_missing_tuning=args.allow_missing_tuning,
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

    if validation_has_errors(checks) and not args.allow_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
