"""
SENTINEL — Main Streamlit Dashboard Entry Point.
Launch: streamlit run dashboard/app.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path so imports work
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Load API keys — try Streamlit Cloud secrets first, then Windows registry
try:
    import streamlit as _st_secrets_check
    for var in ["FRED_API_KEY", "NEWSAPI_KEY", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"]:
        if not os.environ.get(var):
            try:
                val = _st_secrets_check.secrets.get(var, "")
                if val:
                    os.environ[var] = val
            except Exception:
                pass
except Exception:
    pass

# Fallback: Windows registry (local dev on Brandon's machine)
for var in ["FRED_API_KEY", "NEWSAPI_KEY"]:
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

import streamlit as st
import streamlit.components.v1 as components

from config.settings import COLORS, STREAMLIT_LAYOUT, STREAMLIT_PAGE_ICON, STREAMLIT_PAGE_TITLE

# ── Page Config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
bg = COLORS["background"]
surface = COLORS["surface"]
text_color = COLORS["text"]
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {bg};
        }}
        .stSidebar {{
            background-color: {surface};
        }}
        h1, h2, h3 {{
            color: {text_color};
        }}
        .stMetric {{
            background-color: {surface};
            padding: 10px;
            border-radius: 8px;
        }}
        /* Mobile responsive — prevent text cutoff on phones */
        * {{
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
        }}
        @media (max-width: 768px) {{
            h1 {{ font-size: 1.4rem !important; word-break: break-word !important; }}
            h2 {{ font-size: 1.2rem !important; word-break: break-word !important; }}
            h3 {{ font-size: 1.0rem !important; }}
            .stMainBlockContainer {{
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
                max-width: 100vw !important;
                overflow-x: hidden !important;
            }}
            div[data-testid="stHorizontalBlock"] {{
                flex-wrap: wrap !important;
            }}
            div[data-testid="stMetric"] {{
                min-width: 45% !important;
            }}
            section[data-testid="stSidebar"] {{
                width: 250px !important;
                min-width: 250px !important;
            }}
        }}
        /* Hide Streamlit deploy button only — keep sidebar toggle visible */
        .stDeployButton {{
            display: none !important;
        }}
        /* Force sidebar visible on mobile */
        @media (max-width: 768px) {{
            section[data-testid="stSidebar"] {{
                display: block !important;
                z-index: 999 !important;
            }}
            button[data-testid="stSidebarCollapseButton"],
            button[data-testid="stSidebarNavToggle"] {{
                display: block !important;
                z-index: 1000 !important;
            }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Radar Splash Screen HTML ─────────────────────────────────
RADAR_HTML = """<!DOCTYPE html>
<html><head><style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0E1117;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh;overflow:hidden;font-family:'Segoe UI',Consolas,monospace;text-align:center}
@keyframes sweep{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
@keyframes fadeInUp{0%{opacity:0;transform:translateY(40px)}100%{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:.3;transform:scale(1)}50%{opacity:1;transform:scale(1.4)}}
@keyframes blink{0%,100%{opacity:.15}50%{opacity:.85}}
@keyframes expandRing{0%{transform:translate(-50%,-50%) scale(.2);opacity:.9}100%{transform:translate(-50%,-50%) scale(2.5);opacity:0}}
@keyframes glowPulse{0%,100%{box-shadow:0 0 10px #00ff64,0 0 20px rgba(0,255,100,.3)}50%{box-shadow:0 0 20px #00ff64,0 0 50px rgba(0,255,100,.5),0 0 80px rgba(0,255,100,.2)}}
@keyframes drift{0%,100%{transform:translate(0,0)}25%{transform:translate(3px,-2px)}50%{transform:translate(-2px,3px)}75%{transform:translate(2px,1px)}}
@keyframes targetAppear{0%{opacity:0;transform:scale(0)}50%{opacity:1;transform:scale(1.5)}100%{opacity:.8;transform:scale(1)}}
.sc{display:flex;flex-direction:column;align-items:center;justify-content:center;animation:fadeInUp 1s ease-out}
.rw{position:relative;width:min(400px,70vw);height:min(400px,70vw)}
.rbg{position:absolute;top:0;left:0;width:100%;height:100%;border-radius:50%;background:radial-gradient(circle at 50% 50%,#0a1628 0%,#060e1e 40%,#030912 70%,#010408 100%);border:2px solid #1a3a5c;box-shadow:0 0 80px rgba(41,98,255,.12),0 0 150px rgba(41,98,255,.06),inset 0 0 100px rgba(0,0,0,.6);overflow:hidden}
.og{position:absolute;top:-10px;left:-10px;width:calc(100% + 20px);height:calc(100% + 20px);border-radius:50%;border:1px solid rgba(41,98,255,.08);box-shadow:0 0 40px rgba(41,98,255,.06);pointer-events:none}
.rr{position:absolute;border:1px solid rgba(41,98,255,.15);border-radius:50%;top:50%;left:50%;transform:translate(-50%,-50%);pointer-events:none}
.r1{width:25%;height:25%;border-color:rgba(41,98,255,.2)}
.r2{width:50%;height:50%}
.r3{width:75%;height:75%}
.r4{width:95%;height:95%;border-color:rgba(41,98,255,.22)}
.ch,.cv{position:absolute;background:rgba(41,98,255,.08);pointer-events:none}
.ch{width:94%;height:1px;top:50%;left:3%}
.cv{width:1px;height:94%;left:50%;top:3%}
.cd1,.cd2{position:absolute;width:1px;height:94%;top:3%;left:50%;background:rgba(41,98,255,.04);pointer-events:none}
.cd1{transform:rotate(45deg)}.cd2{transform:rotate(-45deg)}
.rs{position:absolute;top:0;left:0;width:100%;height:100%;border-radius:50%;overflow:hidden;animation:sweep 4s linear infinite}
.rs::before{content:'';position:absolute;top:0;left:50%;width:50%;height:50%;transform-origin:0% 100%;background:conic-gradient(from -90deg at 0% 100%,transparent 0deg,rgba(0,200,83,.005) 5deg,rgba(0,200,83,.03) 15deg,rgba(0,200,83,.08) 25deg,rgba(0,200,83,.18) 35deg,rgba(0,200,83,.3) 42deg,rgba(0,255,100,.45) 44deg,transparent 45deg)}
.rs::after{content:'';position:absolute;top:50%;left:50%;width:47%;height:2px;transform-origin:0% 50%;background:linear-gradient(90deg,rgba(0,255,100,.8),rgba(0,255,100,.1),transparent);box-shadow:0 0 12px rgba(0,255,100,.5),0 -2px 8px rgba(0,255,100,.3)}
.rc{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:8px;height:8px;background:#00ff64;border-radius:50%;animation:glowPulse 2s ease-in-out infinite;z-index:10}
.tg{position:absolute;font-weight:900;color:#00C853;text-shadow:0 0 10px rgba(0,200,83,.7),0 0 20px rgba(0,200,83,.3);z-index:5;animation:pulse 3s ease-in-out infinite,drift 8s ease-in-out infinite;pointer-events:none}
.td{color:rgba(0,200,83,.25);text-shadow:0 0 5px rgba(0,200,83,.15);animation:blink 5s ease-in-out infinite,drift 12s ease-in-out infinite}
.tn{animation:targetAppear 1.5s ease-out forwards,pulse 3s ease-in-out 1.5s infinite,drift 10s ease-in-out infinite}
.dr{position:absolute;border:1px solid rgba(0,255,100,.4);border-radius:50%;width:10px;height:10px;pointer-events:none;animation:expandRing 3s ease-out infinite}
.st{font-size:clamp(28px,8vw,56px);font-weight:700;color:#d0ddff;letter-spacing:clamp(8px,3vw,18px);margin-top:35px;text-shadow:0 0 40px rgba(41,98,255,.4),0 2px 4px rgba(0,0,0,.5);animation:fadeInUp 1.2s ease-out .3s both}
.ss{font-size:clamp(10px,2.5vw,14px);color:#4a6a8a;letter-spacing:clamp(3px,1.5vw,8px);margin-top:10px;animation:fadeInUp 1.2s ease-out .6s both}
.sl{font-size:13px;color:#00C853;margin-top:28px;animation:blink 1.5s ease-in-out infinite;letter-spacing:3px}
.sx{font-size:11px;color:#2a4a6a;margin-top:8px;letter-spacing:1px;animation:fadeInUp 1s ease-out 1.5s both}
</style></head><body>
<div class="sc">
<div class="rw">
<div class="og"></div>
<div class="rbg"></div>
<div class="rr r1"></div><div class="rr r2"></div><div class="rr r3"></div><div class="rr r4"></div>
<div class="ch"></div><div class="cv"></div><div class="cd1"></div><div class="cd2"></div>
<div class="rs"></div>
<div class="rc"></div>
<div class="tg" style="top:18%;left:62%;font-size:22px;animation-delay:0s">$</div>
<div class="tg" style="top:32%;left:26%;font-size:20px;animation-delay:.8s">$</div>
<div class="tg" style="top:55%;left:71%;font-size:26px;animation-delay:1.6s">$</div>
<div class="tg" style="top:68%;left:36%;font-size:19px;animation-delay:.4s">$</div>
<div class="tg" style="top:40%;left:56%;font-size:21px;animation-delay:2s">$</div>
<div class="tg" style="top:24%;left:72%;font-size:18px;animation-delay:1.2s">$</div>
<div class="tg" style="top:62%;left:56%;font-size:17px;animation-delay:2.4s">$</div>
<div class="tg" style="top:36%;left:40%;font-size:23px;animation-delay:.6s">$</div>
<div class="tg tn" style="top:45%;left:30%;font-size:24px;animation-delay:2s">$</div>
<div class="tg tn" style="top:75%;left:62%;font-size:20px;animation-delay:4s">$</div>
<div class="tg td" style="top:14%;left:42%;font-size:14px;animation-delay:1.5s">$</div>
<div class="tg td" style="top:78%;left:52%;font-size:13px;animation-delay:2.8s">$</div>
<div class="tg td" style="top:48%;left:20%;font-size:14px;animation-delay:3.2s">$</div>
<div class="tg td" style="top:56%;left:82%;font-size:12px;animation-delay:.3s">$</div>
<div class="tg td" style="top:83%;left:44%;font-size:13px;animation-delay:1.8s">$</div>
<div class="tg td" style="top:20%;left:48%;font-size:12px;animation-delay:2.1s">$</div>
<div class="tg td" style="top:42%;left:76%;font-size:11px;animation-delay:3.5s">$</div>
<div class="tg td" style="top:72%;left:24%;font-size:12px;animation-delay:4.2s">$</div>
<div class="dr" style="top:19%;left:63%;animation-delay:0s"></div>
<div class="dr" style="top:56%;left:72%;animation-delay:1.2s"></div>
<div class="dr" style="top:69%;left:37%;animation-delay:2.4s"></div>
<div class="dr" style="top:33%;left:27%;animation-delay:.8s"></div>
<div class="dr" style="top:41%;left:57%;animation-delay:1.8s"></div>
</div>
<div class="st">SENTINEL</div>
<div class="ss">MARKET INTELLIGENCE SYSTEM</div>
<div class="sl">SCANNING MARKETS...</div>
<div class="sx">EQUITIES &#8226; CRYPTO &#8226; MACRO &#8226; NEWS</div>
<div style="margin-top:40px;font-size:11px;color:#2a4a6a;letter-spacing:2px;animation:fadeInUp 1.5s ease-out 2s both;font-style:italic">from the mind of Brandon Haston</div>
</div>
</body></html>"""


def show_radar():
    """Render the radar splash using a real HTML iframe."""
    components.html(RADAR_HTML, height=720, scrolling=False)


# ── Splash Screen on First Load (10+ seconds) ────────────────
if "initialized" not in st.session_state:
    show_radar()
    time.sleep(10)
    st.session_state["initialized"] = True
    st.rerun()

# ── Navigation ────────────────────────────────────────────────
primary = COLORS["primary"]
PAGES = [
    "Genius Briefing",
    "Deep Analysis",
    "AI Forecast",
    "Probability Forecast",
    "Signal Intelligence",
    "Alpha Screener",
    "Market Overview",
    "Intermarket",
    "Rotation & Breadth",
    "Sector Analysis",
    "Stock Explorer",
    "Crypto",
    "News Intelligence",
    "Regime Monitor",
    "Backtesting",
    "Academy",
    "Home",
]

# Sidebar (desktop)
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding:20px 0;">
            <h1 style="color:{primary}; margin:0;">SENTINEL</h1>
            <p style="color:#888; margin:5px 0;">Market Intelligence System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    sidebar_page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("SENTINEL v1.0")
    st.caption("Data: Yahoo Finance, CoinGecko, FRED, GDELT")

# Top nav dropdown (always visible — essential for mobile)
top_page = st.selectbox(
    "Navigate",
    PAGES,
    index=PAGES.index(sidebar_page),
    label_visibility="collapsed",
)

# Use whichever was changed last
page = top_page

# ── Page Router ───────────────────────────────────────────────
if page == "Genius Briefing":
    from dashboard.views.p12_genius_briefing import render
    render()
elif page == "Deep Analysis":
    from dashboard.views.p16_deep_analysis import render
    render()
elif page == "AI Forecast":
    from dashboard.views.p14_ai_forecast import render
    render()
elif page == "Probability Forecast":
    from dashboard.views.p13_probability_forecast import render
    render()
elif page == "Signal Intelligence":
    from dashboard.views.p08_signal_dashboard import render
    render()
elif page == "Alpha Screener":
    from dashboard.views.p09_alpha_screener import render
    render()
elif page == "Intermarket":
    from dashboard.views.p10_intermarket import render
    render()
elif page == "Rotation & Breadth":
    from dashboard.views.p11_rotation_breadth import render
    render()
elif page == "Market Overview":
    from dashboard.views.p01_market_overview import render
    render()
elif page == "Sector Analysis":
    from dashboard.views.p02_sector_analysis import render
    render()
elif page == "Stock Explorer":
    from dashboard.views.p03_stock_explorer import render
    render()
elif page == "Crypto":
    from dashboard.views.p04_crypto import render
    render()
elif page == "News Intelligence":
    from dashboard.views.p05_news_intelligence import render
    render()
elif page == "Regime Monitor":
    from dashboard.views.p06_regime_monitor import render
    render()
elif page == "Backtesting":
    from dashboard.views.p07_backtesting import render
    render()
elif page == "Academy":
    from dashboard.views.p15_academy import render
    render()
elif page == "Home":
    show_radar()
