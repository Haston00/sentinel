"""
SENTINEL — Quick launch script.
Usage: python run.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    print("=" * 60)
    print("  SENTINEL — Market Intelligence System")
    print("=" * 60)
    print()
    print("Launching Flask dashboard on http://localhost:5000")
    print("Press Ctrl+C to stop.")
    print()

    subprocess.run(
        [sys.executable, "-m", "web.app"],
        cwd=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
