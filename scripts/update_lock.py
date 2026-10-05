"""Regenerate ``requirements-lock.txt`` from the current environment, keeping its header.

Usage: python scripts/update_lock.py

Run it after deliberately upgrading or adding a dependency, then run the tests (``pytest``).
"""
from __future__ import annotations

import datetime
import platform
import subprocess
import sys
from pathlib import Path

LOCK = Path("requirements-lock.txt")


def main() -> None:
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze", "--exclude-editable"], capture_output=True,
                            text=True, check=True).stdout.splitlines()
    pins = sorted((l for l in frozen if l and not l.startswith(("-e", "#"))), key=str.lower)
    header = []
    for line in LOCK.read_text().splitlines():
        header.append(line)
        if line.startswith("# The normal install"):
            break
    header = [l if not l.startswith("# Python ") else
              f"# Python {platform.python_version()}, {platform.platform()}, generated {datetime.date.today()}." for l in header]
    LOCK.write_text("\n".join(header + pins) + "\n")
    print(f"{len(pins)} pinned packages -> {LOCK}")


if __name__ == "__main__":
    main()
