"""
SENTINEL — Flask Web Application.
Bloomberg Terminal-style market intelligence dashboard.
Launch: python -m web.app
"""

import os
import sys
import subprocess
import threading
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, session, jsonify

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load API keys from Windows registry (local dev)
for var in ["FRED_API_KEY", "NEWSAPI_KEY", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"]:
    if not os.environ.get(var):
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 f"[System.Environment]::GetEnvironmentVariable('{var}', 'User')"],
                capture_output=True, text=True, timeout=5,
            )
            val = result.stdout.strip()
            if val:
                os.environ[var] = val
        except Exception:
            pass

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
app.secret_key = os.environ.get("FLASK_SECRET", "sentinel-market-intel-2026")


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def splash():
    return render_template("splash.html")


@app.route("/dashboard")
@app.route("/dashboard/<path:page>")
def dashboard(page="genius_briefing"):
    return render_template("dashboard.html", active_page=page)


# ── Register API blueprint ────────────────────────────────────
from web.api import api_bp
app.register_blueprint(api_bp, url_prefix="/api")


# ── Pre-warm briefing cache on startup ────────────────────────
def _prewarm():
    """Generate briefing in background so first visitor gets instant load."""
    import time
    time.sleep(5)  # Let gunicorn finish starting
    try:
        from web.api import _generate_briefing_with_news, _briefing_cache
        import time as t
        print("Pre-warming briefing cache...")
        briefing = _generate_briefing_with_news()
        _briefing_cache["data"] = briefing
        _briefing_cache["timestamp"] = t.time()
        print("Briefing cache warm.")
    except Exception as e:
        print(f"Pre-warm failed: {e}")

threading.Thread(target=_prewarm, daemon=True).start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # use_reloader=False prevents watchdog from restarting the server
    # when xml.sax / feedparser libs are accessed during briefing generation
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
