"""Validate the v4 Big Five Streamlit demo scaffold."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.demo.bigfive_demo_validation import (  # noqa: E402
    demo_validation_has_errors,
    run_demo_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v4 Big Five demo readiness.")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument(
        "--allow_errors",
        action="store_true",
        help="Exit 0 even when validation errors are present.",
    )
    args = parser.parse_args()

    checks = run_demo_validation(PROJECT_ROOT)
    for check in checks:
        print(f"{check.status.upper():5} {check.name}: {check.details}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["name", "status", "details"])
            writer.writeheader()
            writer.writerows(check.to_dict() for check in checks)

    if demo_validation_has_errors(checks) and not args.allow_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
