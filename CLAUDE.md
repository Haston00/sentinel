# SENTINEL — Claude Code Instructions

## What This Is
Institutional-grade market forecasting system. Flask web app with Bloomberg Terminal-style dark theme. Covers US equities (11 GICS sectors + 110 stocks), crypto (BTC, ETH, SOL + 17 altcoins), bonds, gold, oil, dollar.

## Live URL
http://165.245.138.144

## Server (DigitalOcean)
- **IP:** 165.245.138.144
- **User:** root
- **App location:** /opt/sentinel/
- **Venv:** /opt/sentinel/venv/
- **Service:** sentinel (systemd)
- **Served by:** gunicorn on port 8001, nginx reverse proxy on port 80

## How to Deploy
After making changes, push to GitHub then deploy:
```bash
git add -A && git commit -m "description" && git push origin main
```
Then pull and restart on server (one command):
```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=no root@165.245.138.144 'cd /opt/sentinel && git pull origin main && systemctl restart sentinel'
```

## Server Commands (via SSH)
```bash
# Check service status
ssh -o BatchMode=yes root@165.245.138.144 'systemctl status sentinel'
# View logs
ssh -o BatchMode=yes root@165.245.138.144 'journalctl -u sentinel --no-pager -n 50'
# Restart service
ssh -o BatchMode=yes root@165.245.138.144 'systemctl restart sentinel'
```

## Architecture
```
sentinel/web/
  ├── app.py           — Flask app, splash route, dashboard route
  ├── api.py           — API blueprint (~33 JSON endpoints)
  ├── templates/
  │   ├── base.html     — CDN links (Tailwind, Plotly, fonts)
  │   ├── splash.html   — Radar sweep animation
  │   └── dashboard.html — Shell with sidebar nav + JS page rendering
  └── static/
      ├── css/styles.css — Bloomberg dark theme
      └── js/
          ├── app.js     — Core: navigation, 23 page renderers, API wrappers
          ├── charts.js  — Plotly rendering helpers
          └── splash.js  — Radar sweep animation

Backend (DO NOT MODIFY without reading first):
  config/, data/, features/, models/, simulation/, forecasting/, utils/
```

## Key Files
- `web/app.py` — Flask app factory, routes
- `web/api.py` — All API endpoints
- `web/static/js/app.js` — All 23 page renderers (client-side)
- `web/static/css/styles.css` — Full Bloomberg dark theme
- `config/settings.py` — Colors, cache settings
- `config/assets.py` — 11 sectors, 20 crypto, macro indicators

## Team
- Brandon Haston (Creator)
- Ibrahim Kuraidly (Collaborator)
