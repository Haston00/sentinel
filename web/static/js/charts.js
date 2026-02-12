/**
 * SENTINEL — Plotly Chart Rendering Helpers
 * Wraps Plotly.newPlot with consistent Bloomberg dark theme defaults.
 */

const SENTINEL_THEME = {
    bgPrimary: '#0a0e1a',
    bgSurface: '#111827',
    bgElevated: '#1a2332',
    border: '#1e293b',
    textPrimary: '#e2e8f0',
    textMuted: '#94a3b8',
    primary: '#2962FF',
    bull: '#00C853',
    bear: '#FF1744',
    warning: '#FF9100',
    gold: '#FFD600',
    gridColor: '#1e293b',
};

const DARK_LAYOUT = {
    paper_bgcolor: SENTINEL_THEME.bgSurface,
    plot_bgcolor: SENTINEL_THEME.bgPrimary,
    font: { color: SENTINEL_THEME.textPrimary, family: "'Inter', sans-serif", size: 12 },
    margin: { l: 50, r: 20, t: 40, b: 40 },
    xaxis: { gridcolor: SENTINEL_THEME.gridColor, zerolinecolor: SENTINEL_THEME.border },
    yaxis: { gridcolor: SENTINEL_THEME.gridColor, zerolinecolor: SENTINEL_THEME.border },
    legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: SENTINEL_THEME.textMuted, size: 11 } },
    modebar: { bgcolor: 'rgba(0,0,0,0)', color: SENTINEL_THEME.textMuted, activecolor: SENTINEL_THEME.primary },
};

const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    displaylogo: false,
};

/**
 * Render a Plotly chart from JSON data returned by the API.
 * @param {string} containerId - DOM element ID
 * @param {object} figJson - Plotly figure JSON {data, layout}
 * @param {object} extraLayout - Additional layout overrides
 */
function renderChart(containerId, figJson, extraLayout = {}) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const data = figJson.data || [];
    const layout = Object.assign({}, DARK_LAYOUT, figJson.layout || {}, extraLayout);

    // Force dark theme overrides
    layout.paper_bgcolor = SENTINEL_THEME.bgSurface;
    layout.plot_bgcolor = SENTINEL_THEME.bgPrimary;
    layout.font = Object.assign({}, layout.font || {}, { color: SENTINEL_THEME.textPrimary });
    if (layout.xaxis) layout.xaxis.gridcolor = SENTINEL_THEME.gridColor;
    if (layout.yaxis) layout.yaxis.gridcolor = SENTINEL_THEME.gridColor;

    Plotly.newPlot(el, data, layout, PLOTLY_CONFIG);
}

/**
 * Create a simple gauge chart (CSS-based, no Plotly needed).
 * @param {string} containerId - DOM element ID
 * @param {number} value - Score 0-100
 * @param {string} label - Label text
 */
function renderGauge(containerId, value, label) {
    const el = document.getElementById(containerId);
    if (!el) return;

    value = Math.max(0, Math.min(100, value));
    const color = value >= 65 ? SENTINEL_THEME.bull :
                  value >= 45 ? SENTINEL_THEME.gold :
                  SENTINEL_THEME.bear;
    const deg = (value / 100) * 360;

    el.innerHTML = `
        <div class="gauge-ring" style="background: conic-gradient(${color} ${deg}deg, ${SENTINEL_THEME.border} ${deg}deg); margin: 0 auto;">
            <div class="gauge-inner">
                <div class="gauge-value" style="color:${color}">${Math.round(value)}</div>
                <div class="gauge-label">${label}</div>
            </div>
        </div>
    `;
}

/**
 * Render a Plotly chart from API endpoint.
 * Fetches data, shows loading state, renders chart.
 * @param {string} containerId
 * @param {string} apiUrl
 * @param {object} extraLayout
 */
async function renderChartFromAPI(containerId, apiUrl, extraLayout = {}) {
    const el = document.getElementById(containerId);
    if (!el) return;

    el.innerHTML = '<div class="loading-overlay"><div class="spinner"></div><span>Loading chart...</span></div>';

    try {
        const resp = await fetch(apiUrl);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const figJson = await resp.json();
        renderChart(containerId, figJson, extraLayout);
    } catch (err) {
        el.innerHTML = `<div class="loading-overlay" style="color:var(--bear);">Failed to load chart: ${err.message}</div>`;
    }
}

/**
 * Format a number as currency.
 */
function fmtCurrency(val, decimals = 2) {
    if (val == null || isNaN(val)) return '—';
    return '$' + Number(val).toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

/**
 * Format a number as percentage.
 */
function fmtPct(val, decimals = 2) {
    if (val == null || isNaN(val)) return '—';
    const sign = val >= 0 ? '+' : '';
    return sign + Number(val).toFixed(decimals) + '%';
}

/**
 * Get CSS class for positive/negative change.
 */
function changeClass(val) {
    if (val > 0) return 'positive';
    if (val < 0) return 'negative';
    return 'neutral';
}

/**
 * Get pill class based on score (0-100).
 */
function scorePillClass(score) {
    if (score >= 65) return 'pill-bull';
    if (score >= 45) return 'pill-gold';
    return 'pill-bear';
}

/**
 * Get score label text.
 */
function scoreLabel(score) {
    if (score >= 80) return 'STRONG BUY';
    if (score >= 65) return 'BUY';
    if (score >= 55) return 'LEAN BULL';
    if (score >= 45) return 'NEUTRAL';
    if (score >= 35) return 'LEAN BEAR';
    if (score >= 20) return 'SELL';
    return 'STRONG SELL';
}
