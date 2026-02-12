/**
 * SENTINEL — Core Application JS
 * Navigation, API fetch wrapper, state management, page rendering.
 */

// ═══════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════

const state = {
    activePage: null,
    cache: {},
    loading: false,
};

// ═══════════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════════

function navigateTo(page) {
    if (state.loading) return;
    state.activePage = page;

    // Update URL without reload
    history.pushState({ page }, '', `/dashboard/${page}`);

    // Update sidebar active state
    document.querySelectorAll('.nav-item').forEach(el => {
        el.classList.toggle('active', el.dataset.page === page);
    });

    // Expand the parent group
    document.querySelectorAll('.nav-group').forEach(g => {
        const items = g.querySelectorAll('.nav-item');
        let hasActive = false;
        items.forEach(item => { if (item.dataset.page === page) hasActive = true; });
        if (hasActive) g.classList.remove('collapsed');
    });

    // Close mobile sidebar
    document.getElementById('sidebar')?.classList.remove('open');

    // Load page content
    loadPage(page);
}

function toggleGroup(header) {
    header.parentElement.classList.toggle('collapsed');
}

function toggleSidebar() {
    document.getElementById('sidebar')?.classList.toggle('open');
}

// Handle browser back/forward
window.addEventListener('popstate', (e) => {
    if (e.state?.page) navigateTo(e.state.page);
});

// ═══════════════════════════════════════════════════════════════
// API FETCH WRAPPER
// ═══════════════════════════════════════════════════════════════

async function apiFetch(url, options = {}) {
    try {
        const resp = await fetch(url, {
            headers: { 'Content-Type': 'application/json', ...options.headers },
            ...options,
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ error: `HTTP ${resp.status}` }));
            throw new Error(err.error || `HTTP ${resp.status}`);
        }
        return await resp.json();
    } catch (err) {
        showToast(err.message, 'error');
        throw err;
    }
}

async function apiPost(url, body) {
    return apiFetch(url, {
        method: 'POST',
        body: JSON.stringify(body),
    });
}

// ═══════════════════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ═══════════════════════════════════════════════════════════════

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ═══════════════════════════════════════════════════════════════
// PAGE LOADER
// ═══════════════════════════════════════════════════════════════

function showLoading(container) {
    container.innerHTML = '<div class="loading-overlay"><div class="spinner"></div><span>Loading...</span></div>';
}

async function loadPage(page) {
    const container = document.getElementById('pageContent');
    if (!container) return;

    showLoading(container);
    state.loading = true;

    try {
        // Each page has its own render function
        const renderer = PAGE_RENDERERS[page];
        if (renderer) {
            await renderer(container);
        } else {
            container.innerHTML = `
                <div class="page-content">
                    <div class="top-bar"><div class="top-bar-left">
                        <h1 class="page-title">${formatPageTitle(page)}</h1>
                    </div></div>
                    <div class="card">
                        <div class="loading-overlay" style="padding:40px">
                            <span style="color:var(--text-dim)">Page coming soon</span>
                        </div>
                    </div>
                </div>
            `;
        }
    } catch (err) {
        container.innerHTML = `
            <div class="page-content">
                <div class="card" style="text-align:center;padding:60px">
                    <p style="color:var(--bear);font-size:16px">Error loading page</p>
                    <p style="color:var(--text-dim);margin-top:8px">${err.message}</p>
                    <button class="btn btn-primary" style="margin-top:16px" onclick="navigateTo('${page}')">Retry</button>
                </div>
            </div>
        `;
    }
    state.loading = false;
}

function formatPageTitle(page) {
    return page.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ═══════════════════════════════════════════════════════════════
// PAGE RENDERERS
// ═══════════════════════════════════════════════════════════════

const PAGE_RENDERERS = {

    // ── Market Overview ──────────────────────────────────────
    market_overview: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Market Overview</h1>
                    <span class="page-subtitle">How the market looks right now</span>
                </div><div class="top-bar-right">
                    <button class="btn btn-sm" onclick="navigateTo('market_overview')">&#8635; Refresh</button>
                </div></div>
                <div id="compositeGauge" style="text-align:center;margin-bottom:20px"></div>
                <div class="metric-grid" id="benchmarkMetrics"></div>
                <div style="font-size:11px;color:var(--text-dim);letter-spacing:2px;margin:20px 0 10px;font-weight:600">S&P 500 — 1 YEAR</div>
                <div class="chart-container" id="spyChart"></div>
                <div style="font-size:11px;color:var(--text-dim);letter-spacing:2px;margin:20px 0 10px;font-weight:600">ALL 11 SECTORS</div>
                <div id="sectorTable"></div>
            </div>
        `;

        const data = await apiFetch('/api/market/overview');

        // Composite gauge
        if (data.composite) {
            renderGauge('compositeGauge', data.composite.score, data.composite.label);
        }

        // VIX + Benchmarks
        let metricsHTML = '';
        if (data.vix != null) {
            const vixColor = data.vix > 25 ? 'negative' : data.vix > 18 ? 'neutral' : 'positive';
            metricsHTML += `<div class="metric-card"><div class="metric-label">VIX</div><div class="metric-value">${data.vix}</div><div class="metric-change ${vixColor}">${data.vix > 25 ? 'HIGH FEAR' : data.vix > 18 ? 'ELEVATED' : 'LOW'}</div></div>`;
        }
        for (const [name, b] of Object.entries(data.benchmarks || {})) {
            metricsHTML += `
                <div class="metric-card">
                    <div class="metric-label">${name}</div>
                    <div class="metric-value">${fmtCurrency(b.price)}</div>
                    <div class="metric-change ${changeClass(b.day_change)}">${fmtPct(b.day_change)}</div>
                </div>
            `;
        }
        document.getElementById('benchmarkMetrics').innerHTML = metricsHTML;

        // SPY chart
        renderChartFromAPI('spyChart', '/api/market/chart/SPY?days=252');

        // Sector table
        const sectors = data.sectors || {};
        let tableHTML = '<table class="data-table"><thead><tr><th>Sector</th><th>ETF</th><th>Price</th><th>Day</th><th>Month</th></tr></thead><tbody>';
        for (const [name, s] of Object.entries(sectors)) {
            tableHTML += `<tr>
                <td style="color:var(--text-primary);font-family:var(--font-sans)">${name}</td>
                <td>${s.etf}</td>
                <td>${fmtCurrency(s.price)}</td>
                <td class="${changeClass(s.day_change)}">${fmtPct(s.day_change)}</td>
                <td class="${changeClass(s.month_change)}">${fmtPct(s.month_change)}</td>
            </tr>`;
        }
        tableHTML += '</tbody></table>';
        document.getElementById('sectorTable').innerHTML = tableHTML;
    },

    // ── Genius Briefing — CNBC-Style Market Report ─────────
    genius_briefing: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">MARKET BRIEFING</h1>
                    <span class="page-subtitle">Your morning market intelligence report</span>
                </div><div class="top-bar-right">
                    <span id="briefingAge" style="color:var(--text-dim);font-size:11px;margin-right:12px"></span>
                    <button class="btn btn-sm" id="genBriefingBtn" onclick="generateBriefing(false)">Refresh</button>
                    <button class="btn btn-sm" style="margin-left:6px;border-color:var(--gold);color:var(--gold)" onclick="generateBriefing(true)">Force New</button>
                </div></div>
                <div id="briefingContent"></div>
            </div>
        `;
        // Auto-generate on page load (uses cache if available)
        generateBriefing(false);
    },

    // ── Macro Dashboard ──────────────────────────────────────
    macro_dashboard: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Macro Dashboard</h1>
                    <span class="page-subtitle">GDP, inflation, employment from FRED</span>
                </div></div>
                <div class="metric-grid" id="macroMetrics"></div>
                <div class="section-header">Select Indicator</div>
                <div style="margin-bottom:16px">
                    <select class="form-select" id="macroSelect" onchange="loadMacroChart()">
                        <option value="">Loading indicators...</option>
                    </select>
                </div>
                <div class="chart-container" id="macroChart" style="min-height:400px"></div>
            </div>
        `;

        const data = await apiFetch('/api/macro/indicators');
        const sel = document.getElementById('macroSelect');
        sel.innerHTML = '';

        let metricsHTML = '';
        const keyIndicators = ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'DGS10', 'T10Y2Y'];
        for (const key of keyIndicators) {
            const ind = data[key];
            if (ind) {
                metricsHTML += `
                    <div class="metric-card">
                        <div class="metric-label">${ind.name}</div>
                        <div class="metric-value">${ind.value.toLocaleString()}</div>
                        <div class="metric-change ${changeClass(ind.change)}">${ind.change >= 0 ? '+' : ''}${ind.change.toFixed(3)}</div>
                    </div>
                `;
            }
        }
        document.getElementById('macroMetrics').innerHTML = metricsHTML;

        for (const [key, ind] of Object.entries(data)) {
            const opt = document.createElement('option');
            opt.value = key;
            opt.textContent = ind.name;
            sel.appendChild(opt);
        }

        if (sel.options.length > 0) {
            sel.value = 'DGS10';
            loadMacroChart();
        }
    },

    // ── Regime Monitor ───────────────────────────────────────
    regime_monitor: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Regime Monitor</h1>
                    <span class="page-subtitle">HMM regime detection + transition probabilities</span>
                </div></div>
                <div id="regimeBanner" class="regime-banner">
                    <div class="loading-overlay"><div class="spinner"></div><span>Detecting regime...</span></div>
                </div>
                <div class="section-header">Regime History</div>
                <div class="chart-container" id="regimeChart"></div>
            </div>
        `;

        try {
            const regime = await apiFetch('/api/regime/current');
            let color = 'var(--primary)';
            if (regime.regime === 'Bull') color = 'var(--bull)';
            else if (regime.regime === 'Bear') color = 'var(--bear)';
            else if (regime.regime === 'Transition') color = 'var(--warning)';

            let probsHTML = '';
            if (regime.probabilities) {
                for (const [k, v] of Object.entries(regime.probabilities)) {
                    probsHTML += `<span class="pill pill-blue">${k}: ${(v * 100).toFixed(1)}%</span> `;
                }
            }

            document.getElementById('regimeBanner').innerHTML = `
                <h2 style="color:${color}">Current Regime: ${regime.regime || 'Unknown'}</h2>
                <div class="regime-tags" style="margin-top:12px">${probsHTML}</div>
            `;
            document.getElementById('regimeBanner').style.borderLeftColor = color;
        } catch (e) {
            document.getElementById('regimeBanner').innerHTML = `<p style="color:var(--text-dim)">Could not detect regime</p>`;
        }

        renderChartFromAPI('regimeChart', '/api/regime/history');
    },

    // ── Alpha Screener ───────────────────────────────────────
    alpha_screener: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Alpha Screener</h1>
                    <span class="page-subtitle">Find highest-conviction setups</span>
                </div></div>
                <div class="card" style="margin-bottom:16px">
                    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
                        <select class="form-select" id="screenerMode">
                            <option value="etf">Sector ETFs (11)</option>
                            <option value="all">All Holdings (110+)</option>
                            <option value="custom">Custom Tickers</option>
                        </select>
                        <input class="form-input" id="screenerCustom" placeholder="AAPL,MSFT,NVDA..." style="display:none;width:300px">
                        <button class="btn btn-primary" onclick="runScreener()">Run Screener</button>
                    </div>
                </div>
                <div id="screenerResults"></div>
            </div>
        `;
        document.getElementById('screenerMode').addEventListener('change', (e) => {
            document.getElementById('screenerCustom').style.display = e.target.value === 'custom' ? '' : 'none';
        });
    },

    // ── Forecaster ───────────────────────────────────────────
    forecaster: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Forecaster</h1>
                    <span class="page-subtitle">XGBoost price forecasts with confidence bands</span>
                </div></div>
                <div class="card" style="margin-bottom:16px">
                    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
                        <input class="form-input" id="forecastTicker" placeholder="SPY" value="SPY" style="width:120px">
                        <select class="form-select" id="forecastHorizon">
                            <option value="1W">1 Week</option>
                            <option value="1M" selected>1 Month</option>
                            <option value="3M">3 Months</option>
                        </select>
                        <button class="btn btn-primary" onclick="runForecast()">Generate Forecast</button>
                    </div>
                </div>
                <div class="chart-container" id="forecastChart" style="min-height:450px">
                    <div class="loading-overlay"><span style="color:var(--text-dim)">Enter a ticker and click Generate</span></div>
                </div>
                <div id="forecastDetails"></div>
            </div>
        `;
    },

    // ── Scenario Lab ─────────────────────────────────────────
    scenario_lab: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Scenario Lab</h1>
                    <span class="page-subtitle">What-if scenario builder</span>
                </div></div>
                <div class="card" style="margin-bottom:16px">
                    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
                        <select class="form-select" id="scenarioType">
                            <option value="rate_hike">Fed Rate Hike</option>
                            <option value="rate_cut">Fed Rate Cut</option>
                            <option value="recession">Recession</option>
                            <option value="inflation_shock">Inflation Shock</option>
                        </select>
                        <input class="form-input" id="scenarioMag" type="number" value="0.25" step="0.25" min="0.25" max="5" style="width:100px">
                        <button class="btn btn-primary" onclick="runScenario()">Run Scenario</button>
                    </div>
                </div>
                <div id="scenarioResults"></div>
            </div>
        `;
    },

    // ── Correlation Matrix ───────────────────────────────────
    correlation_matrix: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Correlation Matrix</h1>
                    <span class="page-subtitle">Asset correlation heatmap</span>
                </div></div>
                <div class="chart-container" id="corrChart" style="min-height:500px"></div>
            </div>
        `;
        renderChartFromAPI('corrChart', '/api/correlation/matrix');
    },

    // ── Sector Heatmap ───────────────────────────────────────
    sector_heatmap: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Sector Heatmap</h1>
                    <span class="page-subtitle">S&P 500 sector performance</span>
                </div></div>
                <div id="sectorHeatmap"></div>
            </div>
        `;

        const data = await apiFetch('/api/sectors/heatmap');
        const sectors = data.sectors || [];

        let html = '<table class="data-table"><thead><tr><th>Sector</th><th>ETF</th><th>Price</th><th>Day</th><th>Week</th><th>Month</th><th>Quarter</th><th>YTD</th></tr></thead><tbody>';
        for (const s of sectors) {
            html += `<tr>
                <td style="color:var(--text-primary);font-family:var(--font-sans)">${s.sector}</td>
                <td>${s.etf}</td>
                <td>${fmtCurrency(s.price)}</td>
                <td class="${changeClass(s.day)}">${fmtPct(s.day)}</td>
                <td class="${changeClass(s.week)}">${fmtPct(s.week)}</td>
                <td class="${changeClass(s.month)}">${fmtPct(s.month)}</td>
                <td class="${changeClass(s.quarter)}">${fmtPct(s.quarter)}</td>
                <td class="${changeClass(s.ytd)}">${fmtPct(s.ytd)}</td>
            </tr>`;
        }
        html += '</tbody></table>';
        document.getElementById('sectorHeatmap').innerHTML = html;
    },

    // ── Factor Explorer ──────────────────────────────────────
    factor_explorer: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Factor Explorer</h1>
                    <span class="page-subtitle">Factor returns analysis (value, momentum, quality)</span>
                </div></div>
                <div style="margin-bottom:16px">
                    <select class="form-select" id="factorSelect" onchange="loadFactor()">
                        <option value="value">Value (VLUE)</option>
                        <option value="momentum">Momentum (MTUM)</option>
                        <option value="quality">Quality (QUAL)</option>
                        <option value="size">Size (SIZE)</option>
                        <option value="min_vol">Min Volatility (USMV)</option>
                        <option value="growth">Growth (IWF)</option>
                    </select>
                </div>
                <div class="metric-grid" id="factorMetrics"></div>
                <div class="chart-container" id="factorChart"></div>
            </div>
        `;
        loadFactor();
    },

    // ── Commodities ──────────────────────────────────────────
    commodities: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Commodities</h1>
                    <span class="page-subtitle">Oil, gold, copper, natural gas</span>
                </div></div>
                <div class="metric-grid" id="commMetrics"></div>
                <div class="section-header">Chart</div>
                <div style="margin-bottom:12px">
                    <select class="form-select" id="commSelect" onchange="loadCommChart()"></select>
                </div>
                <div class="chart-container" id="commChart"></div>
            </div>
        `;

        const data = await apiFetch('/api/commodities/data');
        let metricsHTML = '';
        const sel = document.getElementById('commSelect');

        for (const [name, c] of Object.entries(data)) {
            metricsHTML += `
                <div class="metric-card">
                    <div class="metric-label">${name}</div>
                    <div class="metric-value">${fmtCurrency(c.price)}</div>
                    <div class="metric-change ${changeClass(c.day)}">${fmtPct(c.day)}</div>
                </div>
            `;
            const opt = document.createElement('option');
            opt.value = c.ticker;
            opt.textContent = name;
            sel.appendChild(opt);
        }
        document.getElementById('commMetrics').innerHTML = metricsHTML;
        if (sel.options.length) loadCommChart();
    },

    // ── Crypto Dashboard ─────────────────────────────────────
    crypto_dashboard: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Crypto Dashboard</h1>
                    <span class="page-subtitle">BTC, ETH, altcoin performance</span>
                </div></div>
                <div class="metric-grid" id="cryptoMetrics"></div>
                <div class="section-header">All Crypto Assets</div>
                <div id="cryptoTable"></div>
            </div>
        `;

        const data = await apiFetch('/api/crypto/data');
        let metricsHTML = '';
        let tableHTML = '<table class="data-table"><thead><tr><th>Symbol</th><th>Name</th><th>Price</th><th>24h Change</th><th>Market Cap</th></tr></thead><tbody>';

        const majors = ['BTC', 'ETH', 'SOL'];
        for (const sym of majors) {
            const c = data[sym];
            if (c) {
                metricsHTML += `
                    <div class="metric-card">
                        <div class="metric-label">${c.name}</div>
                        <div class="metric-value">${fmtCurrency(c.price)}</div>
                        <div class="metric-change ${changeClass(c.change_24h)}">${fmtPct(c.change_24h)}</div>
                    </div>
                `;
            }
        }

        for (const [sym, c] of Object.entries(data)) {
            tableHTML += `<tr>
                <td style="color:var(--gold);font-weight:600">${sym}</td>
                <td style="color:var(--text-primary);font-family:var(--font-sans)">${c.name}</td>
                <td>${fmtCurrency(c.price, c.price < 1 ? 4 : 2)}</td>
                <td class="${changeClass(c.change_24h)}">${fmtPct(c.change_24h)}</td>
                <td>${c.market_cap ? '$' + (c.market_cap / 1e9).toFixed(1) + 'B' : '—'}</td>
            </tr>`;
        }
        tableHTML += '</tbody></table>';

        document.getElementById('cryptoMetrics').innerHTML = metricsHTML;
        document.getElementById('cryptoTable').innerHTML = tableHTML;
    },

    // ── Fixed Income ─────────────────────────────────────────
    fixed_income: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Fixed Income</h1>
                    <span class="page-subtitle">Yield curves, spreads, bond ETFs</span>
                </div></div>
                <div class="metric-grid" id="bondMetrics"></div>
                <div class="section-header">Treasury Yield Curve</div>
                <div class="chart-container" id="yieldCurveChart"></div>
            </div>
        `;

        const data = await apiFetch('/api/fixed-income/curves');
        let metricsHTML = '';
        for (const [name, b] of Object.entries(data)) {
            if (name === 'yield_curve_chart') continue;
            metricsHTML += `
                <div class="metric-card">
                    <div class="metric-label">${name}</div>
                    <div class="metric-value">${fmtCurrency(b.price)}</div>
                    <div class="metric-change ${changeClass(b.day)}">${fmtPct(b.day)}</div>
                </div>
            `;
        }
        document.getElementById('bondMetrics').innerHTML = metricsHTML;

        if (data.yield_curve_chart) {
            renderChart('yieldCurveChart', data.yield_curve_chart);
        }
    },

    // ── Forex Dashboard ──────────────────────────────────────
    forex_dashboard: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Forex Dashboard</h1>
                    <span class="page-subtitle">Major currency pairs, DXY</span>
                </div></div>
                <div class="metric-grid" id="fxMetrics"></div>
                <div class="section-header">FX Pairs</div>
                <div id="fxTable"></div>
            </div>
        `;

        const data = await apiFetch('/api/forex/pairs');
        let metricsHTML = '';
        let tableHTML = '<table class="data-table"><thead><tr><th>Pair</th><th>Rate</th><th>Day</th><th>Week</th></tr></thead><tbody>';

        for (const [name, p] of Object.entries(data)) {
            metricsHTML += `
                <div class="metric-card">
                    <div class="metric-label">${name}</div>
                    <div class="metric-value">${p.price.toFixed(4)}</div>
                    <div class="metric-change ${changeClass(p.day)}">${fmtPct(p.day)}</div>
                </div>
            `;
            tableHTML += `<tr>
                <td style="color:var(--text-primary);font-family:var(--font-sans)">${name}</td>
                <td>${p.price.toFixed(4)}</td>
                <td class="${changeClass(p.day)}">${fmtPct(p.day)}</td>
                <td class="${changeClass(p.week)}">${fmtPct(p.week)}</td>
            </tr>`;
        }
        tableHTML += '</tbody></table>';

        document.getElementById('fxMetrics').innerHTML = metricsHTML;
        document.getElementById('fxTable').innerHTML = tableHTML;
    },

    // ── Portfolio ─────────────────────────────────────────────
    portfolio: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Portfolio</h1>
                    <span class="page-subtitle">Holdings, allocation, P&L</span>
                </div><div class="top-bar-right">
                    <button class="btn btn-sm" onclick="navigateTo('portfolio')">&#8635; Refresh</button>
                </div></div>
                <div class="metric-grid" id="portMetrics"></div>
                <div class="section-header">Holdings</div>
                <div id="portHoldings"></div>
            </div>
        `;

        try {
            const data = await apiFetch('/api/portfolio/holdings');
            const port = data.portfolio || {};
            const pnlClass = (port.total_pnl || 0) >= 0 ? 'positive' : 'negative';

            document.getElementById('portMetrics').innerHTML = `
                <div class="metric-card"><div class="metric-label">Portfolio Value</div><div class="metric-value">${fmtCurrency(port.total_value)}</div></div>
                <div class="metric-card"><div class="metric-label">Cash</div><div class="metric-value">${fmtCurrency(port.cash)}</div></div>
                <div class="metric-card"><div class="metric-label">Total P&L</div><div class="metric-value ${pnlClass}">${fmtCurrency(port.total_pnl)}</div><div class="metric-change ${pnlClass}">${fmtPct(port.total_pnl_pct)}</div></div>
                <div class="metric-card"><div class="metric-label">Positions</div><div class="metric-value">${(port.positions_detail || []).length}</div></div>
            `;

            const positions = port.positions_detail || [];
            if (positions.length === 0) {
                document.getElementById('portHoldings').innerHTML = '<div class="card"><p style="color:var(--text-dim);text-align:center;padding:30px">No positions. Go to Paper Trading to make your first trade.</p></div>';
            } else {
                let html = '<table class="data-table"><thead><tr><th>Ticker</th><th>Shares</th><th>Avg Cost</th><th>Price</th><th>Value</th><th>P&L</th><th>Weight</th></tr></thead><tbody>';
                for (const p of positions) {
                    const cls = p.pnl >= 0 ? 'positive' : 'negative';
                    html += `<tr>
                        <td style="color:var(--primary);font-weight:600">${p.ticker}</td>
                        <td>${p.shares}</td>
                        <td>${fmtCurrency(p.avg_cost)}</td>
                        <td>${fmtCurrency(p.current_price)}</td>
                        <td>${fmtCurrency(p.market_value)}</td>
                        <td class="${cls}">${fmtCurrency(p.pnl)} (${fmtPct(p.pnl_pct)})</td>
                        <td>${p.weight}%</td>
                    </tr>`;
                }
                html += '</tbody></table>';
                document.getElementById('portHoldings').innerHTML = html;
            }
        } catch (e) {
            document.getElementById('portMetrics').innerHTML = '<div class="card"><p style="color:var(--text-dim)">Could not load portfolio</p></div>';
        }
    },

    // ── Paper Trading ────────────────────────────────────────
    paper_trading: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Paper Trading</h1>
                    <span class="page-subtitle">Simulated trade execution</span>
                </div></div>
                <div class="card" style="margin-bottom:16px">
                    <h3 style="color:var(--primary);margin-bottom:12px;font-size:14px">EXECUTE TRADE</h3>
                    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
                        <select class="form-select" id="tradeAction">
                            <option value="buy">BUY</option>
                            <option value="sell">SELL</option>
                        </select>
                        <input class="form-input" id="tradeTicker" placeholder="AAPL" style="width:100px">
                        <input class="form-input" id="tradeShares" type="number" placeholder="Shares" min="1" style="width:100px">
                        <input class="form-input" id="tradeThesis" placeholder="Trade thesis..." style="width:250px">
                        <button class="btn btn-primary" onclick="executeTrade()">Execute</button>
                    </div>
                </div>
                <div id="tradeResult"></div>
                <div class="section-header">Trade History</div>
                <div id="tradeHistory"></div>
            </div>
        `;
        loadTradeHistory();
    },

    // ── Risk Analytics ───────────────────────────────────────
    risk_analytics: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Risk Analytics</h1>
                    <span class="page-subtitle">VaR, drawdown, Sharpe ratio</span>
                </div></div>
                <div class="metric-grid" id="riskMetrics"></div>
            </div>
        `;

        try {
            const stats = await apiFetch('/api/risk/analytics');
            let html = '';
            for (const [key, val] of Object.entries(stats)) {
                if (typeof val === 'number') {
                    html += `<div class="metric-card"><div class="metric-label">${key.replace(/_/g, ' ')}</div><div class="metric-value">${typeof val === 'number' && Math.abs(val) < 10 ? val.toFixed(2) : val.toLocaleString()}</div></div>`;
                }
            }
            document.getElementById('riskMetrics').innerHTML = html || '<div class="card"><p style="color:var(--text-dim);text-align:center;padding:30px">Risk analytics require active portfolio positions</p></div>';
        } catch (e) {
            document.getElementById('riskMetrics').innerHTML = '<div class="card"><p style="color:var(--text-dim)">Could not load risk data</p></div>';
        }
    },

    // ── Option Flow ──────────────────────────────────────────
    option_flow: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Option Flow</h1>
                    <span class="page-subtitle">Unusual options activity</span>
                </div></div>
                <div id="optionFlowData"></div>
            </div>
        `;

        const data = await apiFetch('/api/options/flow');
        const options = data.options || [];
        if (options.length === 0) {
            document.getElementById('optionFlowData').innerHTML = `<div class="card"><p style="color:var(--text-dim);text-align:center;padding:40px">${data.note || 'No unusual options activity detected'}</p></div>`;
        } else {
            let html = '<table class="data-table"><thead><tr><th>Ticker</th><th>Type</th><th>Strike</th><th>Expiry</th><th>Volume</th><th>OI</th><th>Premium</th></tr></thead><tbody>';
            for (const o of options) {
                html += `<tr><td>${o.ticker}</td><td class="${o.type === 'CALL' ? 'positive' : 'negative'}">${o.type}</td><td>${fmtCurrency(o.strike)}</td><td>${o.expiry}</td><td>${o.volume?.toLocaleString()}</td><td>${o.open_interest?.toLocaleString()}</td><td>${fmtCurrency(o.premium)}</td></tr>`;
            }
            html += '</tbody></table>';
            document.getElementById('optionFlowData').innerHTML = html;
        }
    },

    // ── Earnings Calendar ────────────────────────────────────
    earnings_calendar: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Earnings Calendar</h1>
                    <span class="page-subtitle">Upcoming earnings, estimates</span>
                </div></div>
                <div id="earningsData"></div>
            </div>
        `;

        const data = await apiFetch('/api/earnings/calendar');
        const earnings = data.earnings || [];
        if (earnings.length === 0) {
            document.getElementById('earningsData').innerHTML = `<div class="card"><p style="color:var(--text-dim);text-align:center;padding:40px">${data.note || 'No upcoming earnings data available'}</p></div>`;
        } else {
            let html = '<table class="data-table"><thead><tr><th>Date</th><th>Ticker</th><th>Company</th><th>EPS Est.</th><th>Revenue Est.</th></tr></thead><tbody>';
            for (const e of earnings) {
                html += `<tr><td>${e.date}</td><td style="color:var(--primary)">${e.ticker}</td><td style="font-family:var(--font-sans)">${e.company}</td><td>${e.eps_estimate}</td><td>${e.revenue_estimate}</td></tr>`;
            }
            html += '</tbody></table>';
            document.getElementById('earningsData').innerHTML = html;
        }
    },

    // ── Insider Tracker ──────────────────────────────────────
    insider_tracker: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Insider Tracker</h1>
                    <span class="page-subtitle">SEC Form 4 filings</span>
                </div></div>
                <div id="insiderData"></div>
            </div>
        `;

        const data = await apiFetch('/api/insiders/recent');
        const trades = data.trades || [];
        if (trades.length === 0) {
            document.getElementById('insiderData').innerHTML = `<div class="card"><p style="color:var(--text-dim);text-align:center;padding:40px">${data.note || 'No recent insider trades'}</p></div>`;
        } else {
            let html = '<table class="data-table"><thead><tr><th>Date</th><th>Ticker</th><th>Insider</th><th>Type</th><th>Shares</th><th>Value</th></tr></thead><tbody>';
            for (const t of trades) {
                const cls = t.type === 'Buy' ? 'positive' : 'negative';
                html += `<tr><td>${t.date}</td><td style="color:var(--primary)">${t.ticker}</td><td style="font-family:var(--font-sans)">${t.insider}</td><td class="${cls}">${t.type}</td><td>${t.shares?.toLocaleString()}</td><td>${fmtCurrency(t.value)}</td></tr>`;
            }
            html += '</tbody></table>';
            document.getElementById('insiderData').innerHTML = html;
        }
    },

    // ── Dark Pool ────────────────────────────────────────────
    dark_pool: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Dark Pool</h1>
                    <span class="page-subtitle">Dark pool volume, prints</span>
                </div></div>
                <div class="card"><p style="color:var(--text-dim);text-align:center;padding:60px">Dark pool data requires premium data feeds.<br>Coming soon in a future update.</p></div>
            </div>
        `;
    },

    // ── Learning Center ──────────────────────────────────────
    learning_center: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Learning Center</h1>
                    <span class="page-subtitle">Market education and resources</span>
                </div></div>
                <div class="grid-2">
                    <div class="card">
                        <h3 style="color:var(--primary);margin-bottom:12px">Technical Analysis</h3>
                        <p style="color:var(--text-muted);font-size:13px;line-height:1.7">
                            Learn about candlestick patterns, moving averages, RSI, MACD, Bollinger Bands,
                            and how SENTINEL's composite scoring engine combines them into a single conviction score.
                        </p>
                    </div>
                    <div class="card">
                        <h3 style="color:var(--primary);margin-bottom:12px">Regime Detection</h3>
                        <p style="color:var(--text-muted);font-size:13px;line-height:1.7">
                            Hidden Markov Models detect whether the market is in a Bull, Bear, or Transition regime.
                            Understanding the current regime is critical for position sizing and strategy selection.
                        </p>
                    </div>
                    <div class="card">
                        <h3 style="color:var(--primary);margin-bottom:12px">Factor Investing</h3>
                        <p style="color:var(--text-muted);font-size:13px;line-height:1.7">
                            Value, momentum, quality, and size factors explain most of the cross-section of returns.
                            Use the Factor Explorer to see which factors are leading the market right now.
                        </p>
                    </div>
                    <div class="card">
                        <h3 style="color:var(--primary);margin-bottom:12px">Risk Management</h3>
                        <p style="color:var(--text-muted);font-size:13px;line-height:1.7">
                            Position sizing, Value-at-Risk, maximum drawdown, and the Sharpe ratio.
                            Professional risk management separates winning traders from losers.
                        </p>
                    </div>
                </div>
            </div>
        `;
    },

    // ── Settings ─────────────────────────────────────────────
    settings: async (container) => {
        container.innerHTML = `
            <div class="page-content">
                <div class="top-bar"><div class="top-bar-left">
                    <h1 class="page-title">Settings</h1>
                    <span class="page-subtitle">Preferences and configuration</span>
                </div></div>
                <div class="card">
                    <h3 style="color:var(--primary);margin-bottom:16px;font-size:14px">SYSTEM INFO</h3>
                    <table class="data-table">
                        <tbody>
                            <tr><td style="color:var(--text-muted)">Version</td><td>SENTINEL v2.0 (Flask)</td></tr>
                            <tr><td style="color:var(--text-muted)">Theme</td><td>Bloomberg Terminal Dark</td></tr>
                            <tr><td style="color:var(--text-muted)">Data Sources</td><td>Yahoo Finance, CoinGecko, FRED, GDELT, NewsAPI</td></tr>
                            <tr><td style="color:var(--text-muted)">Models</td><td>XGBoost, HMM Regime, Composite Scoring</td></tr>
                            <tr><td style="color:var(--text-muted)">Cache TTL</td><td>5 min (stocks), 3 min (crypto), 6 hrs (macro)</td></tr>
                        </tbody>
                    </table>
                </div>
                <div class="card" style="margin-top:16px">
                    <h3 style="color:var(--primary);margin-bottom:16px;font-size:14px">PORTFOLIO</h3>
                    <button class="btn" style="border-color:var(--bear);color:var(--bear)" onclick="if(confirm('Reset portfolio to $100K?')) resetPortfolio()">Reset Portfolio</button>
                </div>
                <div class="card" style="margin-top:16px">
                    <h3 style="color:var(--primary);margin-bottom:16px;font-size:14px">SESSION</h3>
                    <a href="/logout" class="btn">Logout</a>
                </div>
            </div>
        `;
    },
};


// ═══════════════════════════════════════════════════════════════
// ACTION FUNCTIONS (called by page buttons)
// ═══════════════════════════════════════════════════════════════

async function generateBriefing(force = false) {
    const btn = document.getElementById('genBriefingBtn');
    const content = document.getElementById('briefingContent');
    if (!content) return;

    if (btn) btn.disabled = true;
    content.innerHTML = `
        <div class="card" style="text-align:center;padding:60px">
            <div class="spinner" style="margin:0 auto 16px"></div>
            <p style="color:var(--gold);font-size:16px;font-weight:600">${force ? 'GENERATING FRESH BRIEFING' : 'LOADING BRIEFING'}</p>
            <p style="color:var(--text-dim);font-size:13px;margin-top:8px" id="briefingStatus">${force ? 'Scanning all markets...' : 'Checking for cached data...'}</p>
        </div>`;

    // Animate status messages only for fresh generation
    const statusEl = document.getElementById('briefingStatus');
    const msgs = ['Scoring all 11 sectors...', 'Checking bonds, gold, oil, dollar...', 'Pulling crypto prices...', 'Reading news headlines...', 'Analyzing market drivers...', 'Building your briefing...'];
    let mi = 0;
    const statusTimer = force ? setInterval(() => { if (statusEl) statusEl.textContent = msgs[mi++ % msgs.length]; }, 3000) : null;

    try {
        const b = await apiPost('/api/briefing/generate', { force });
        if (statusTimer) clearInterval(statusTimer);

        // Show cache status
        const ageEl = document.getElementById('briefingAge');
        if (ageEl && b._cached) {
            const mins = Math.floor((b._age_seconds || 0) / 60);
            ageEl.textContent = mins > 0 ? `cached ${mins}m ago` : 'just cached';
        } else if (ageEl) {
            ageEl.textContent = 'just generated';
        }

        let html = '';

        // Figure out overall market mood for colors
        const moodGood = b.regime?.includes('STRONG') || b.regime?.includes('POSITIVE') || b.regime?.includes('HEALTHY');
        const moodBad = b.regime?.includes('WARNING') || b.regime?.includes('WORSE') || b.regime?.includes('CAREFUL');
        const moodColor = moodGood ? 'var(--bull)' : moodBad ? 'var(--bear)' : 'var(--warning)';
        const moodEmoji = moodGood ? '&#9650;' : moodBad ? '&#9660;' : '&#9654;';

        // ══════════════════════════════════════════════════════════
        // 1. THE BIG PICTURE — one banner, plain English
        // ══════════════════════════════════════════════════════════
        html += `
            <div style="background:linear-gradient(135deg,#111827,#1a2332);border:1px solid var(--border);border-radius:8px;padding:28px 32px;margin-bottom:20px;border-left:5px solid ${moodColor}">
                <div style="font-size:11px;color:var(--text-dim);letter-spacing:2px;margin-bottom:10px">THE MARKET RIGHT NOW</div>
                <div style="font-size:20px;font-weight:700;color:${moodColor};margin-bottom:12px">${moodEmoji} ${b.regime || 'CHECKING...'}</div>
                <p style="font-size:15px;color:var(--text-primary);line-height:1.8">${b.regime_detail || ''}</p>
                ${b.timestamp ? `<div style="color:var(--text-dim);font-size:11px;margin-top:14px">${b.timestamp}</div>` : ''}
            </div>`;

        // ══════════════════════════════════════════════════════════
        // 2. TODAY'S BIG NEWS — headlines FIRST, this is what Brandon wants
        // ══════════════════════════════════════════════════════════
        if (b.news?.length) {
            html += `
                <div class="card" style="padding:24px 28px;margin-bottom:20px;border-left:5px solid var(--primary)">
                    <div style="font-size:11px;color:var(--primary);letter-spacing:2px;margin-bottom:16px;font-weight:600">TODAY'S BIG NEWS</div>`;
            for (const article of b.news.slice(0, 12)) {
                const title = article.title || article.Title || '';
                const source = article.source || article.Source || '';
                const url = article.url || article.URL || '#';
                if (title) {
                    html += `
                        <div style="padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.05)">
                            <a href="${url}" target="_blank" style="color:var(--text-primary);text-decoration:none;font-size:14px;line-height:1.6;display:block;font-weight:500">${title}</a>
                            ${source ? `<span style="color:var(--text-dim);font-size:11px;margin-top:2px;display:inline-block">${source}</span>` : ''}
                        </div>`;
                }
            }
            html += '</div>';
        }

        // ══════════════════════════════════════════════════════════
        // 3. HOW THE INDEXES ARE DOING — simple up/down arrows
        // ══════════════════════════════════════════════════════════
        if (b.market_scores && Object.keys(b.market_scores).length) {
            html += '<div class="metric-grid" style="margin-bottom:20px">';
            for (const [name, ms] of Object.entries(b.market_scores)) {
                const up = ms.day_change >= 0;
                const color = up ? 'var(--bull)' : 'var(--bear)';
                const arrow = up ? '&#9650;' : '&#9660;';
                const word = up ? 'UP' : 'DOWN';
                html += `
                    <div class="metric-card" style="border-top:3px solid ${color}">
                        <div class="metric-label">${name}</div>
                        <div style="font-size:24px;font-weight:700;color:${color};font-family:var(--font-mono)">${arrow} ${Math.abs(ms.day_change).toFixed(1)}%</div>
                        <div style="font-size:11px;color:var(--text-dim);margin-top:4px">${word} today &bull; ${ms.ret_1m >= 0 ? '+' : ''}${ms.ret_1m.toFixed(1)}% this month</div>
                    </div>`;
            }
            html += '</div>';
        }

        // ══════════════════════════════════════════════════════════
        // 4. WHAT'S DRIVING THE MARKET — the WHY behind the moves
        // ══════════════════════════════════════════════════════════
        if (b.market_drivers?.length) {
            html += `
                <div class="card" style="padding:24px 28px;margin-bottom:20px">
                    <div style="font-size:11px;color:var(--warning);letter-spacing:2px;margin-bottom:14px;font-weight:600">WHY IS THE MARKET MOVING?</div>`;
            for (const d of b.market_drivers) {
                const c = d.direction === 'BULLISH' ? 'var(--bull)' : d.direction === 'BEARISH' ? 'var(--bear)' : 'var(--warning)';
                const icon = d.direction === 'BULLISH' ? '&#9650;' : d.direction === 'BEARISH' ? '&#9660;' : '&#8226;';
                html += `<div style="margin-bottom:12px;padding:12px 16px;background:rgba(255,255,255,0.02);border-radius:6px;border-left:3px solid ${c}">
                    <div style="font-weight:600;color:var(--text-primary);font-size:14px"><span style="color:${c}">${icon}</span> ${d.signal || d.title || ''}</div>
                    <div style="color:var(--text-muted);font-size:13px;margin-top:4px;line-height:1.7">${d.detail || d.description || ''}</div>
                </div>`;
            }
            html += '</div>';
        }

        // ══════════════════════════════════════════════════════════
        // 5. HEADLINE — the analyst's big picture summary
        // ══════════════════════════════════════════════════════════
        if (b.headline) {
            html += `
                <div class="card" style="padding:24px 28px;margin-bottom:20px;border-left:4px solid var(--gold)">
                    <div style="font-size:11px;color:var(--gold);letter-spacing:2px;margin-bottom:8px;font-weight:600">ANALYST TAKE</div>
                    <p style="font-size:15px;color:var(--text-primary);line-height:1.8">${b.headline}</p>
                </div>`;
        }

        // ══════════════════════════════════════════════════════════
        // 6. WINNERS & LOSERS — who's hot and who's not
        // ══════════════════════════════════════════════════════════
        if (b.top_movers) {
            const flyers = b.top_movers.high_flyers || [];
            const sinkers = b.top_movers.sinking_ships || [];
            if (flyers.length || sinkers.length) {
                html += `<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px">
                    <div class="card" style="padding:20px 24px;border-top:3px solid var(--bull)">
                        <div style="font-size:11px;color:var(--bull);letter-spacing:2px;margin-bottom:12px;font-weight:600">&#9650; WINNING TODAY</div>`;
                for (const m of flyers.slice(0, 5)) {
                    html += `<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04)">
                        <span style="color:var(--text-primary);font-weight:600">${m.name}</span>
                        <span style="color:var(--bull);font-weight:700;font-family:var(--font-mono)">${m.day_change >= 0 ? '+' : ''}${m.day_change.toFixed(1)}%</span>
                    </div>`;
                }
                html += `</div><div class="card" style="padding:20px 24px;border-top:3px solid var(--bear)">
                        <div style="font-size:11px;color:var(--bear);letter-spacing:2px;margin-bottom:12px;font-weight:600">&#9660; LOSING TODAY</div>`;
                for (const m of sinkers.slice(0, 5)) {
                    html += `<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04)">
                        <span style="color:var(--text-primary);font-weight:600">${m.name}</span>
                        <span style="color:var(--bear);font-weight:700;font-family:var(--font-mono)">${m.day_change >= 0 ? '+' : ''}${m.day_change.toFixed(1)}%</span>
                    </div>`;
                }
                html += '</div></div>';
            }
        }

        // ══════════════════════════════════════════════════════════
        // 7. CRYPTO — prices and what's happening
        // ══════════════════════════════════════════════════════════
        if (b.crypto && Object.keys(b.crypto).length) {
            html += `<div class="card" style="padding:24px 28px;margin-bottom:20px;border-left:4px solid var(--gold)">
                <div style="font-size:11px;color:var(--gold);letter-spacing:2px;margin-bottom:14px;font-weight:600">CRYPTO</div>
                <div class="metric-grid" style="margin-bottom:12px">`;
            for (const [sym, c] of Object.entries(b.crypto)) {
                const color = c.change_24h >= 0 ? 'var(--bull)' : 'var(--bear)';
                const priceStr = c.price >= 1000 ? '$' + c.price.toLocaleString(undefined, {maximumFractionDigits:0}) : '$' + c.price.toFixed(2);
                html += `<div class="metric-card" style="border-top:3px solid ${color}">
                    <div class="metric-label">${sym}</div>
                    <div class="metric-value">${priceStr}</div>
                    <div class="metric-change ${changeClass(c.change_24h)}">${fmtPct(c.change_24h)} 24h</div>
                </div>`;
            }
            html += '</div>';
            for (const [sym, c] of Object.entries(b.crypto)) {
                if (c.description) {
                    html += `<p style="color:var(--text-muted);font-size:13px;line-height:1.7;margin-bottom:6px"><strong style="color:var(--text-primary)">${sym}:</strong> ${c.description}</p>`;
                }
            }
            html += '</div>';
        }

        // ══════════════════════════════════════════════════════════
        // 8. SECTORS — simple hot/cold, no scores
        // ══════════════════════════════════════════════════════════
        if (b.sector_rankings?.length) {
            const hot = b.sector_rankings.slice(0, 3);
            const cold = b.sector_rankings.slice(-3).reverse();
            html += `<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px">
                <div class="card" style="padding:20px 24px">
                    <div style="font-size:11px;color:var(--bull);letter-spacing:2px;margin-bottom:12px;font-weight:600">HOTTEST SECTORS</div>`;
            for (const s of hot) {
                html += `<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04)">
                    <span style="color:var(--text-primary);font-weight:500">${s.sector}</span>
                    <span class="${changeClass(s.ret_1m)}" style="font-weight:600;font-family:var(--font-mono)">${fmtPct(s.ret_1m)} /mo</span>
                </div>`;
            }
            html += `</div><div class="card" style="padding:20px 24px">
                    <div style="font-size:11px;color:var(--bear);letter-spacing:2px;margin-bottom:12px;font-weight:600">COLDEST SECTORS</div>`;
            for (const s of cold) {
                html += `<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04)">
                    <span style="color:var(--text-primary);font-weight:500">${s.sector}</span>
                    <span class="${changeClass(s.ret_1m)}" style="font-weight:600;font-family:var(--font-mono)">${fmtPct(s.ret_1m)} /mo</span>
                </div>`;
            }
            html += '</div></div>';
        }

        // ══════════════════════════════════════════════════════════
        // 9. WHAT TO WATCH — risk factors in plain English
        // ══════════════════════════════════════════════════════════
        if (b.risk_factors?.length) {
            html += `<div class="card" style="padding:24px 28px;margin-bottom:20px">
                <div style="font-size:11px;color:var(--warning);letter-spacing:2px;margin-bottom:14px;font-weight:600">WHAT COULD GO WRONG</div>`;
            for (const rf of b.risk_factors) {
                html += `<div style="color:var(--text-primary);font-size:14px;line-height:1.7;margin-bottom:10px;padding-left:16px;border-left:2px solid var(--warning)">&#9888; ${rf}</div>`;
            }
            html += '</div>';
        }

        // ══════════════════════════════════════════════════════════
        // 10. THE BOTTOM LINE — what it all means
        // ══════════════════════════════════════════════════════════
        if (b.bottom_line) {
            html += `
                <div style="background:linear-gradient(135deg,#111827,#1a2332);border:1px solid var(--gold);border-radius:8px;padding:28px 32px;margin-bottom:20px">
                    <div style="font-size:11px;color:var(--gold);letter-spacing:2px;margin-bottom:10px;font-weight:600">THE BOTTOM LINE</div>
                    <p style="font-size:16px;color:var(--text-primary);line-height:1.8;font-weight:500">${b.bottom_line}</p>
                </div>`;
        }

        content.innerHTML = html;
    } catch (e) {
        if (statusTimer) clearInterval(statusTimer);
        content.innerHTML = `<div class="card" style="text-align:center;padding:40px"><p style="color:var(--bear);font-size:16px">Failed to generate briefing</p><p style="color:var(--text-dim);margin-top:8px">${e.message}</p><button class="btn btn-primary" style="margin-top:16px" onclick="generateBriefing(true)">Try Again</button></div>`;
    }
    if (btn) btn.disabled = false;
}

async function runScreener() {
    const mode = document.getElementById('screenerMode')?.value || 'etf';
    const custom = document.getElementById('screenerCustom')?.value || '';
    const results = document.getElementById('screenerResults');
    if (!results) return;

    results.innerHTML = '<div class="loading-overlay"><div class="spinner"></div><span>Scanning universe...</span></div>';

    try {
        const data = await apiPost('/api/screener/run', { mode, tickers: custom });
        const rows = data.results || [];

        if (rows.length === 0) {
            results.innerHTML = '<div class="card"><p style="color:var(--text-dim);text-align:center;padding:30px">No results</p></div>';
            return;
        }

        let html = '<table class="data-table"><thead><tr><th>#</th><th>Ticker</th><th>Score</th><th>Label</th><th>Trend</th><th>Momentum</th><th>RSI</th><th>Volume</th></tr></thead><tbody>';
        rows.forEach((r, i) => {
            html += `<tr>
                <td><span class="rank-number">${i + 1}</span></td>
                <td style="color:var(--primary);font-weight:600">${r.ticker}</td>
                <td><span class="score-badge ${r.score >= 65 ? 'bullish' : r.score >= 45 ? 'neutral' : 'bearish'}">${r.score}</span></td>
                <td>${r.label}</td>
                <td>${r.trend}</td>
                <td>${r.momentum}</td>
                <td>${r.rsi}</td>
                <td>${r.volume}</td>
            </tr>`;
        });
        html += '</tbody></table>';
        results.innerHTML = html;
    } catch (e) {
        results.innerHTML = `<div class="card"><p style="color:var(--bear)">${e.message}</p></div>`;
    }
}

async function runForecast() {
    const ticker = document.getElementById('forecastTicker')?.value || 'SPY';
    const horizon = document.getElementById('forecastHorizon')?.value || '1M';
    const chartDiv = document.getElementById('forecastChart');
    const detailsDiv = document.getElementById('forecastDetails');

    if (!chartDiv) return;
    chartDiv.innerHTML = '<div class="loading-overlay"><div class="spinner"></div><span>Running forecast...</span></div>';

    try {
        const data = await apiPost('/api/forecast/run', { ticker, horizon });
        if (data.chart) {
            renderChart('forecastChart', data.chart);
        }
        if (data.forecast && detailsDiv) {
            let html = '<div class="metric-grid" style="margin-top:16px">';
            for (const [h, fc] of Object.entries(data.forecast)) {
                html += `<div class="metric-card"><div class="metric-label">${h} Forecast</div><div class="metric-value ${changeClass(fc.point)}">${fmtPct(fc.point ? fc.point * 100 : 0)}</div></div>`;
            }
            html += '</div>';
            detailsDiv.innerHTML = html;
        }
    } catch (e) {
        chartDiv.innerHTML = `<div class="loading-overlay"><p style="color:var(--bear)">${e.message}</p></div>`;
    }
}

async function runScenario() {
    const type = document.getElementById('scenarioType')?.value;
    const mag = parseFloat(document.getElementById('scenarioMag')?.value || 0.25);
    const results = document.getElementById('scenarioResults');
    if (!results) return;

    results.innerHTML = '<div class="loading-overlay"><div class="spinner"></div></div>';

    try {
        const data = await apiPost('/api/scenario/run', { type, magnitude: mag });
        let html = `<div class="regime-banner" style="border-left-color:var(--warning)"><h2 style="font-size:18px">${data.description || 'Scenario'}</h2></div>`;
        html += '<div class="metric-grid">';
        const impacts = { equities: 'Equities', bonds: 'Bonds', gold: 'Gold', dollar: 'Dollar' };
        for (const [key, label] of Object.entries(impacts)) {
            const val = data[key] || 0;
            html += `<div class="metric-card"><div class="metric-label">${label}</div><div class="metric-value ${changeClass(val)}">${fmtPct(val)}</div></div>`;
        }
        html += '</div>';
        results.innerHTML = html;
    } catch (e) {
        results.innerHTML = `<div class="card"><p style="color:var(--bear)">${e.message}</p></div>`;
    }
}

function loadMacroChart() {
    const sel = document.getElementById('macroSelect');
    if (!sel?.value) return;
    renderChartFromAPI('macroChart', `/api/macro/chart/${sel.value}`);
}

async function loadFactor() {
    const sel = document.getElementById('factorSelect');
    if (!sel?.value) return;

    const metricsDiv = document.getElementById('factorMetrics');
    const chartDiv = document.getElementById('factorChart');
    if (chartDiv) chartDiv.innerHTML = '<div class="loading-overlay"><div class="spinner"></div></div>';

    try {
        const data = await apiFetch(`/api/factors/${sel.value}`);
        if (data.chart) renderChart('factorChart', data.chart);
        if (data.stats && metricsDiv) {
            metricsDiv.innerHTML = `
                <div class="metric-card"><div class="metric-label">Price</div><div class="metric-value">${fmtCurrency(data.stats.price)}</div></div>
                <div class="metric-card"><div class="metric-label">Day</div><div class="metric-value ${changeClass(data.stats.day)}">${fmtPct(data.stats.day)}</div></div>
                <div class="metric-card"><div class="metric-label">Month</div><div class="metric-value ${changeClass(data.stats.month)}">${fmtPct(data.stats.month)}</div></div>
                <div class="metric-card"><div class="metric-label">Year</div><div class="metric-value ${changeClass(data.stats.year)}">${fmtPct(data.stats.year)}</div></div>
            `;
        }
    } catch (e) {}
}

function loadCommChart() {
    const sel = document.getElementById('commSelect');
    if (!sel?.value) return;
    renderChartFromAPI('commChart', `/api/commodities/chart/${sel.value}`);
}

async function executeTrade() {
    const ticker = document.getElementById('tradeTicker')?.value?.toUpperCase();
    const action = document.getElementById('tradeAction')?.value;
    const shares = parseFloat(document.getElementById('tradeShares')?.value || 0);
    const thesis = document.getElementById('tradeThesis')?.value || '';
    const resultDiv = document.getElementById('tradeResult');

    if (!ticker || !shares) {
        showToast('Enter ticker and shares', 'error');
        return;
    }

    try {
        const result = await apiPost('/api/portfolio/trade', { ticker, action, shares, thesis });
        if (result.success) {
            resultDiv.innerHTML = `<div class="card" style="border-left:4px solid var(--bull)"><p style="color:var(--bull);font-weight:600">${result.message || 'Trade executed'}</p></div>`;
            showToast(`${action.toUpperCase()} ${shares} ${ticker} executed`, 'success');
            loadTradeHistory();
        } else {
            resultDiv.innerHTML = `<div class="card" style="border-left:4px solid var(--bear)"><p style="color:var(--bear)">${result.error || result.message || 'Trade failed'}</p></div>`;
        }
    } catch (e) {
        resultDiv.innerHTML = `<div class="card"><p style="color:var(--bear)">${e.message}</p></div>`;
    }
}

async function loadTradeHistory() {
    const div = document.getElementById('tradeHistory');
    if (!div) return;

    try {
        const data = await apiFetch('/api/portfolio/history');
        const trades = data.trades || [];
        if (trades.length === 0) {
            div.innerHTML = '<div class="card"><p style="color:var(--text-dim);text-align:center;padding:20px">No trades yet</p></div>';
            return;
        }

        let html = '<table class="data-table"><thead><tr><th>Date</th><th>Action</th><th>Ticker</th><th>Shares</th><th>Price</th><th>Value</th></tr></thead><tbody>';
        for (const t of trades.slice(-20).reverse()) {
            const cls = t.action === 'buy' ? 'positive' : 'negative';
            html += `<tr><td>${t.date || t.timestamp || ''}</td><td class="${cls}" style="text-transform:uppercase">${t.action}</td><td style="color:var(--primary)">${t.ticker}</td><td>${t.shares}</td><td>${fmtCurrency(t.price)}</td><td>${fmtCurrency(t.total)}</td></tr>`;
        }
        html += '</tbody></table>';
        div.innerHTML = html;
    } catch (e) {
        div.innerHTML = '<div class="card"><p style="color:var(--text-dim)">Could not load history</p></div>';
    }
}

async function resetPortfolio() {
    try {
        await apiPost('/api/portfolio/reset', {});
        showToast('Portfolio reset to $100K', 'success');
    } catch (e) {
        showToast('Failed to reset: ' + e.message, 'error');
    }
}
