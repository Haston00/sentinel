"""
SENTINEL — Asset universe definitions.
GICS sectors, crypto universe, macro indicator catalog, benchmark tickers.
"""

# ── GICS Sectors with ETF Proxies and Top Holdings ────────────
SECTORS = {
    "Technology": {
        "etf": "XLK",
        "holdings": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ADBE", "AMD", "INTC", "CSCO", "ORCL"],
    },
    "Healthcare": {
        "etf": "XLV",
        "holdings": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN"],
    },
    "Financials": {
        "etf": "XLF",
        "holdings": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK"],
    },
    "Consumer Discretionary": {
        "etf": "XLY",
        "holdings": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG"],
    },
    "Communication Services": {
        "etf": "XLC",
        "holdings": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR"],
    },
    "Industrials": {
        "etf": "XLI",
        "holdings": ["GE", "CAT", "UNP", "HON", "UPS", "BA", "RTX", "DE", "LMT", "ADP"],
    },
    "Consumer Staples": {
        "etf": "XLP",
        "holdings": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "EL"],
    },
    "Energy": {
        "etf": "XLE",
        "holdings": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "FANG", "PSX", "VLO", "OXY"],
    },
    "Utilities": {
        "etf": "XLU",
        "holdings": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC"],
    },
    "Real Estate": {
        "etf": "XLRE",
        "holdings": ["PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB"],
    },
    "Materials": {
        "etf": "XLB",
        "holdings": ["LIN", "APD", "SHW", "FCX", "ECL", "NEM", "NUE", "DOW", "DD", "PPG"],
    },
}

# All sector ETFs for quick access
SECTOR_ETFS = [s["etf"] for s in SECTORS.values()]

# Broad market benchmarks
BENCHMARKS = {
    "S&P 500": "SPY",
    "Nasdaq 100": "QQQ",
    "Dow Jones": "DIA",
    "Russell 2000": "IWM",
    "Total Market": "VTI",
}

# Bond / Rate proxies
BOND_TICKERS = {
    "20Y+ Treasury": "TLT",
    "7-10Y Treasury": "IEF",
    "1-3Y Treasury": "SHY",
    "High Yield": "HYG",
    "Investment Grade": "LQD",
}

# Commodity proxies
COMMODITY_TICKERS = {
    "Gold": "GLD",
    "Silver": "SLV",
    "Crude Oil": "USO",
    "Natural Gas": "UNG",
    "Broad Commodities": "DJP",
}

# Dollar index proxy
DOLLAR_TICKER = "UUP"

# Volatility
VIX_TICKER = "^VIX"

# ── Crypto Universe ───────────────────────────────────────────
CRYPTO_MAJOR = {
    "bitcoin": {"symbol": "BTC", "name": "Bitcoin"},
    "ethereum": {"symbol": "ETH", "name": "Ethereum"},
    "solana": {"symbol": "SOL", "name": "Solana"},
}

CRYPTO_ALTCOINS = {
    "binancecoin": {"symbol": "BNB", "name": "BNB"},
    "ripple": {"symbol": "XRP", "name": "XRP"},
    "cardano": {"symbol": "ADA", "name": "Cardano"},
    "dogecoin": {"symbol": "DOGE", "name": "Dogecoin"},
    "avalanche-2": {"symbol": "AVAX", "name": "Avalanche"},
    "polkadot": {"symbol": "DOT", "name": "Polkadot"},
    "chainlink": {"symbol": "LINK", "name": "Chainlink"},
    "polygon-ecosystem-token": {"symbol": "POL", "name": "Polygon"},
    "uniswap": {"symbol": "UNI", "name": "Uniswap"},
    "litecoin": {"symbol": "LTC", "name": "Litecoin"},
    "cosmos": {"symbol": "ATOM", "name": "Cosmos"},
    "stellar": {"symbol": "XLM", "name": "Stellar"},
    "arbitrum": {"symbol": "ARB", "name": "Arbitrum"},
    "optimism": {"symbol": "OP", "name": "Optimism"},
    "near": {"symbol": "NEAR", "name": "NEAR Protocol"},
    "injective-protocol": {"symbol": "INJ", "name": "Injective"},
    "sui": {"symbol": "SUI", "name": "Sui"},
}

# Combined crypto universe
CRYPTO_ALL = {**CRYPTO_MAJOR, **CRYPTO_ALTCOINS}

# ── FRED Macro Indicators ────────────────────────────────────
MACRO_INDICATORS = {
    # GDP & Output
    "GDP": {"series": "GDP", "name": "Real GDP", "frequency": "quarterly"},
    "INDPRO": {"series": "INDPRO", "name": "Industrial Production", "frequency": "monthly"},

    # Labor Market
    "UNRATE": {"series": "UNRATE", "name": "Unemployment Rate", "frequency": "monthly"},
    "PAYEMS": {"series": "PAYEMS", "name": "Nonfarm Payrolls", "frequency": "monthly"},
    "ICSA": {"series": "ICSA", "name": "Initial Jobless Claims", "frequency": "weekly"},

    # Inflation
    "CPIAUCSL": {"series": "CPIAUCSL", "name": "CPI (All Urban)", "frequency": "monthly"},
    "CPILFESL": {"series": "CPILFESL", "name": "Core CPI", "frequency": "monthly"},
    "PCEPI": {"series": "PCEPI", "name": "PCE Price Index", "frequency": "monthly"},
    "T5YIE": {"series": "T5YIE", "name": "5Y Breakeven Inflation", "frequency": "daily"},

    # Interest Rates & Yields
    "FEDFUNDS": {"series": "FEDFUNDS", "name": "Fed Funds Rate", "frequency": "monthly"},
    "DGS2": {"series": "DGS2", "name": "2Y Treasury Yield", "frequency": "daily"},
    "DGS10": {"series": "DGS10", "name": "10Y Treasury Yield", "frequency": "daily"},
    "DGS30": {"series": "DGS30", "name": "30Y Treasury Yield", "frequency": "daily"},
    "T10Y2Y": {"series": "T10Y2Y", "name": "10Y-2Y Spread", "frequency": "daily"},
    "T10Y3M": {"series": "T10Y3M", "name": "10Y-3M Spread", "frequency": "daily"},

    # Financial Conditions
    "BAMLH0A0HYM2": {"series": "BAMLH0A0HYM2", "name": "HY OAS Spread", "frequency": "daily"},
    "VIXCLS": {"series": "VIXCLS", "name": "VIX (FRED)", "frequency": "daily"},

    # Money & Credit
    "M2SL": {"series": "M2SL", "name": "M2 Money Supply", "frequency": "monthly"},

    # Housing
    "HOUST": {"series": "HOUST", "name": "Housing Starts", "frequency": "monthly"},
    "CSUSHPINSA": {"series": "CSUSHPINSA", "name": "Case-Shiller Home Price", "frequency": "monthly"},

    # PMI (ISM)
    "MANEMP": {"series": "MANEMP", "name": "Manufacturing Employment", "frequency": "monthly"},
}

# ── All Tickers for Bulk Download ─────────────────────────────
def get_all_stock_tickers():
    """Return deduplicated list of all stock tickers in the universe."""
    tickers = set()
    for sector in SECTORS.values():
        tickers.add(sector["etf"])
        tickers.update(sector["holdings"])
    for t in BENCHMARKS.values():
        tickers.add(t)
    for t in BOND_TICKERS.values():
        tickers.add(t)
    for t in COMMODITY_TICKERS.values():
        tickers.add(t)
    tickers.add(DOLLAR_TICKER)
    tickers.add(VIX_TICKER)
    return sorted(tickers)


def get_sector_for_ticker(ticker: str) -> str | None:
    """Return GICS sector name for a given ticker, or None."""
    for sector_name, info in SECTORS.items():
        if ticker == info["etf"] or ticker in info["holdings"]:
            return sector_name
    return None
