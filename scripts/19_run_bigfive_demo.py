"""Launch the v4 Big Five Streamlit demo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency: streamlit. "
            "Install it before launching the demo."
        ) from exc

    app_path = PROJECT_ROOT / "src" / "demo" / "bigfive_app.py"
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
            ],
            cwd=PROJECT_ROOT,
        )
    )


if __name__ == "__main__":
    main()
