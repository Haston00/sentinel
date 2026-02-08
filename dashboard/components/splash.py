"""
SENTINEL — Animated radar splash screen.
Radar sweeps scanning for dollar sign targets.
Uses st.components.v1.html for full CSS animation support.
"""

SPLASH_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    background: #0E1117;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    overflow: hidden;
    font-family: 'Segoe UI', Consolas, monospace;
}

@keyframes sweep {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(40px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 0.3; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.4); }
}
@keyframes blink {
    0%, 100% { opacity: 0.15; }
    50% { opacity: 0.85; }
}
@keyframes expandRing {
    0% { transform: translate(-50%,-50%) scale(0.2); opacity: 0.9; }
    100% { transform: translate(-50%,-50%) scale(2.5); opacity: 0; }
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 10px #00ff64, 0 0 20px rgba(0,255,100,0.3); }
    50% { box-shadow: 0 0 20px #00ff64, 0 0 50px rgba(0,255,100,0.5), 0 0 80px rgba(0,255,100,0.2); }
}
@keyframes scanline {
    0% { opacity: 0; top: 5%; }
    10% { opacity: 0.06; }
    90% { opacity: 0.06; }
    100% { opacity: 0; top: 95%; }
}
@keyframes targetAppear {
    0% { opacity: 0; transform: scale(0); }
    50% { opacity: 1; transform: scale(1.5); }
    100% { opacity: 0.8; transform: scale(1); }
}
@keyframes drift {
    0%, 100% { transform: translate(0, 0); }
    25% { transform: translate(3px, -2px); }
    50% { transform: translate(-2px, 3px); }
    75% { transform: translate(2px, 1px); }
}

.splash-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    animation: fadeInUp 1s ease-out;
}

.radar-wrapper {
    position: relative;
    width: 400px;
    height: 400px;
}

/* Main radar background */
.radar-bg {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    border-radius: 50%;
    background: radial-gradient(circle at 50% 50%, #0a1628 0%, #060e1e 40%, #030912 70%, #010408 100%);
    border: 2px solid #1a3a5c;
    box-shadow: 0 0 80px rgba(41, 98, 255, 0.12), 0 0 150px rgba(41, 98, 255, 0.06), inset 0 0 100px rgba(0, 0, 0, 0.6);
    overflow: hidden;
}

/* Scanline effect */
.radar-bg::after {
    content: '';
    position: absolute;
    left: 0; width: 100%; height: 2px;
    background: rgba(41, 98, 255, 0.06);
    animation: scanline 8s linear infinite;
}

/* Radar rings */
.radar-ring {
    position: absolute;
    border: 1px solid rgba(41, 98, 255, 0.15);
    border-radius: 50%;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
}
.ring-1 { width: 25%; height: 25%; border-color: rgba(41, 98, 255, 0.2); }
.ring-2 { width: 50%; height: 50%; }
.ring-3 { width: 75%; height: 75%; }
.ring-4 { width: 95%; height: 95%; border-color: rgba(41, 98, 255, 0.22); }

/* Crosshair lines */
.crosshair-h, .crosshair-v {
    position: absolute;
    background: rgba(41, 98, 255, 0.08);
    pointer-events: none;
}
.crosshair-h {
    width: 94%; height: 1px;
    top: 50%; left: 3%;
}
.crosshair-v {
    width: 1px; height: 94%;
    left: 50%; top: 3%;
}

/* Diagonal crosshairs */
.crosshair-d1, .crosshair-d2 {
    position: absolute;
    width: 1px;
    height: 94%;
    top: 3%;
    left: 50%;
    background: rgba(41, 98, 255, 0.04);
    pointer-events: none;
}
.crosshair-d1 { transform: rotate(45deg); }
.crosshair-d2 { transform: rotate(-45deg); }

/* Sweep beam - the rotating radar arm */
.radar-sweep {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    border-radius: 50%;
    overflow: hidden;
    animation: sweep 4s linear infinite;
}

/* The green sweep cone */
.radar-sweep::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    width: 50%; height: 50%;
    transform-origin: 0% 100%;
    background: conic-gradient(
        from -90deg at 0% 100%,
        transparent 0deg,
        rgba(0, 200, 83, 0.005) 5deg,
        rgba(0, 200, 83, 0.03) 15deg,
        rgba(0, 200, 83, 0.08) 25deg,
        rgba(0, 200, 83, 0.18) 35deg,
        rgba(0, 200, 83, 0.3) 42deg,
        rgba(0, 255, 100, 0.45) 44deg,
        transparent 45deg
    );
}

/* Bright leading edge of sweep */
.radar-sweep::after {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    width: 47%;
    height: 2px;
    transform-origin: 0% 50%;
    background: linear-gradient(90deg, rgba(0,255,100,0.8), rgba(0,255,100,0.1), transparent);
    box-shadow: 0 0 12px rgba(0,255,100,0.5), 0 -2px 8px rgba(0,255,100,0.3);
}

/* Center dot with glow */
.radar-center {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 8px; height: 8px;
    background: #00ff64;
    border-radius: 50%;
    animation: glowPulse 2s ease-in-out infinite;
    z-index: 10;
}

/* Dollar sign bogeys */
.target {
    position: absolute;
    font-weight: 900;
    color: #00C853;
    text-shadow: 0 0 10px rgba(0,200,83,0.7), 0 0 20px rgba(0,200,83,0.3);
    z-index: 5;
    animation: pulse 3s ease-in-out infinite, drift 8s ease-in-out infinite;
    pointer-events: none;
}

.target-dim {
    color: rgba(0,200,83,0.25);
    text-shadow: 0 0 5px rgba(0,200,83,0.15);
    animation: blink 5s ease-in-out infinite, drift 12s ease-in-out infinite;
}

.target-new {
    animation: targetAppear 1.5s ease-out forwards, pulse 3s ease-in-out 1.5s infinite, drift 10s ease-in-out infinite;
}

/* Detection rings that expand outward from targets */
.detect-ring {
    position: absolute;
    border: 1px solid rgba(0, 255, 100, 0.4);
    border-radius: 50%;
    width: 10px; height: 10px;
    pointer-events: none;
    animation: expandRing 3s ease-out infinite;
}

/* Outer glow ring around whole radar */
.outer-glow {
    position: absolute;
    top: -10px; left: -10px;
    width: calc(100% + 20px);
    height: calc(100% + 20px);
    border-radius: 50%;
    border: 1px solid rgba(41, 98, 255, 0.08);
    box-shadow: 0 0 40px rgba(41, 98, 255, 0.06);
    pointer-events: none;
}

/* Title area */
.splash-title {
    font-size: 56px;
    font-weight: 700;
    color: #d0ddff;
    letter-spacing: 18px;
    margin-top: 35px;
    text-shadow: 0 0 40px rgba(41, 98, 255, 0.4), 0 2px 4px rgba(0,0,0,0.5);
    animation: fadeInUp 1.2s ease-out 0.3s both;
}
.splash-sub {
    font-size: 14px;
    color: #4a6a8a;
    letter-spacing: 8px;
    margin-top: 10px;
    animation: fadeInUp 1.2s ease-out 0.6s both;
}
.splash-loading {
    font-size: 13px;
    color: #00C853;
    margin-top: 28px;
    animation: blink 1.5s ease-in-out infinite;
    letter-spacing: 3px;
}
.splash-status {
    font-size: 11px;
    color: #2a4a6a;
    margin-top: 8px;
    letter-spacing: 1px;
    animation: fadeInUp 1s ease-out 1.5s both;
}
</style>
</head>
<body>

<div class="splash-container">
    <div class="radar-wrapper">
        <div class="outer-glow"></div>
        <div class="radar-bg"></div>

        <!-- Rings -->
        <div class="radar-ring ring-1"></div>
        <div class="radar-ring ring-2"></div>
        <div class="radar-ring ring-3"></div>
        <div class="radar-ring ring-4"></div>

        <!-- Crosshairs -->
        <div class="crosshair-h"></div>
        <div class="crosshair-v"></div>
        <div class="crosshair-d1"></div>
        <div class="crosshair-d2"></div>

        <!-- Sweep beam -->
        <div class="radar-sweep"></div>

        <!-- Center dot -->
        <div class="radar-center"></div>

        <!-- Bright dollar sign targets -->
        <div class="target" style="top:18%; left:62%; animation-delay:0s; font-size:22px;">$</div>
        <div class="target" style="top:32%; left:26%; animation-delay:0.8s; font-size:20px;">$</div>
        <div class="target" style="top:55%; left:71%; animation-delay:1.6s; font-size:26px;">$</div>
        <div class="target" style="top:68%; left:36%; animation-delay:0.4s; font-size:19px;">$</div>
        <div class="target" style="top:40%; left:56%; animation-delay:2.0s; font-size:21px;">$</div>
        <div class="target" style="top:24%; left:72%; animation-delay:1.2s; font-size:18px;">$</div>
        <div class="target" style="top:62%; left:56%; animation-delay:2.4s; font-size:17px;">$</div>
        <div class="target" style="top:36%; left:40%; animation-delay:0.6s; font-size:23px;">$</div>
        <div class="target target-new" style="top:45%; left:30%; font-size:24px; animation-delay:2s;">$</div>
        <div class="target target-new" style="top:75%; left:62%; font-size:20px; animation-delay:4s;">$</div>

        <!-- Dimmer background targets -->
        <div class="target target-dim" style="top:14%; left:42%; animation-delay:1.5s; font-size:14px;">$</div>
        <div class="target target-dim" style="top:78%; left:52%; animation-delay:2.8s; font-size:13px;">$</div>
        <div class="target target-dim" style="top:48%; left:20%; animation-delay:3.2s; font-size:14px;">$</div>
        <div class="target target-dim" style="top:56%; left:82%; animation-delay:0.3s; font-size:12px;">$</div>
        <div class="target target-dim" style="top:83%; left:44%; animation-delay:1.8s; font-size:13px;">$</div>
        <div class="target target-dim" style="top:20%; left:48%; animation-delay:2.1s; font-size:12px;">$</div>
        <div class="target target-dim" style="top:42%; left:76%; animation-delay:3.5s; font-size:11px;">$</div>
        <div class="target target-dim" style="top:72%; left:24%; animation-delay:4.2s; font-size:12px;">$</div>

        <!-- Detection pulse rings expanding from key targets -->
        <div class="detect-ring" style="top:19%; left:63%; animation-delay:0s;"></div>
        <div class="detect-ring" style="top:56%; left:72%; animation-delay:1.2s;"></div>
        <div class="detect-ring" style="top:69%; left:37%; animation-delay:2.4s;"></div>
        <div class="detect-ring" style="top:33%; left:27%; animation-delay:0.8s;"></div>
        <div class="detect-ring" style="top:41%; left:57%; animation-delay:1.8s;"></div>
    </div>

    <div class="splash-title">SENTINEL</div>
    <div class="splash-sub">MARKET INTELLIGENCE SYSTEM</div>
    <div class="splash-loading">SCANNING MARKETS...</div>
    <div class="splash-status">EQUITIES &bull; CRYPTO &bull; MACRO &bull; NEWS</div>
</div>

</body>
</html>
"""


def show_splash():
    """Display the animated radar splash screen using an HTML component."""
    import streamlit.components.v1 as components
    components.html(SPLASH_HTML, height=720, scrolling=False)
