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
    print("Launching Streamlit dashboard...")
    print("Press Ctrl+C to stop.")
    print()

    app_path = PROJECT_ROOT / "dashboard" / "app.py"

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path),
         "--server.headless", "true",
         "--theme.base", "dark"],
        cwd=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
