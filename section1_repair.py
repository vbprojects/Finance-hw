"""Replacement Section 1 cells for case_study2.ipynb.

This module does not modify the notebook. The assembly script can use
``get_cells()`` to obtain notebook-ready ``{cell_type, source}`` records.
"""

import json
from pathlib import Path


def get_cells():
    notebook = json.loads(Path(__file__).with_name("case_study2.ipynb").read_text(encoding="utf-8"))
    cells = [
        {"cell_type": cell["cell_type"], "source": "".join(cell["source"])}
        for cell in notebook["cells"][:10]
    ]

    cells[0]["source"] = """# Case Study 2: The G10 FX Carry Trade

## Section 1 — Data Pipeline: Building the G10 Universe

All returns and carry P&L are in percentage points per holding period. FRED rates and Yahoo Finance FX closes are downloaded by the code and cached in the relative `data_cache/` folder. The run prints the data vintage; FRED and Yahoo may revise or extend their series after that date.
"""

    # The ECB source is daily. A cache containing an early observation from a
    # month must not be mistaken for a complete month on a later run.
    old_cache_logic = '''    target_month = latest_expected_rate_month(now)
    out = {}
    status = []
    for currency, series_id in FRED_SERIES.items():
        path = cache_dir / f"fred_{series_id}.csv"
        cached = None
        if path.exists():
            cached = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            cached = pd.to_numeric(cached, errors="coerce").dropna().sort_index()
        latest = None if cached is None or cached.empty else cached.index.max().to_period("M").to_timestamp()
        need_refresh = (
            cached is None or cached.empty or
            cached.index.min() > pd.Timestamp(START) or
            latest < target_month
        )'''
    new_cache_logic = '''    target_month = latest_expected_rate_month(now)
    utc_now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    utc_now = utc_now.tz_localize("UTC") if utc_now.tzinfo is None else utc_now.tz_convert("UTC")
    expected_ecb_day = (utc_now.normalize().tz_localize(None) - pd.offsets.BDay(1)).normalize()
    out = {}
    status = []
    for currency, series_id in FRED_SERIES.items():
        path = cache_dir / f"fred_{series_id}.csv"
        cached = None
        if path.exists():
            cached = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            cached = pd.to_numeric(cached, errors="coerce").dropna().sort_index()
        latest_raw_day = None if cached is None or cached.empty else cached.index.max().normalize()
        latest = None if latest_raw_day is None else latest_raw_day.to_period("M").to_timestamp()
        ecb_stale = series_id == "ECBDFR" and (latest_raw_day is None or latest_raw_day < expected_ecb_day)
        # A holiday or delayed FRED feed merits one retry per UTC day. An
        # obviously old cache is refreshed even if its file was touched today.
        checked_today = (path.exists() and
                         pd.Timestamp.fromtimestamp(path.stat().st_mtime, tz="UTC").date() == utc_now.date())
        if (ecb_stale and checked_today and latest_raw_day is not None and
                latest_raw_day >= expected_ecb_day - pd.Timedelta(days=7)):
            ecb_stale = False
        need_refresh = (
            cached is None or cached.empty or
            cached.index.min() > pd.Timestamp(START) or
            latest < target_month or ecb_stale
        )'''
    if old_cache_logic in cells[2]["source"]:
        cells[2]["source"] = cells[2]["source"].replace(old_cache_logic, new_cache_logic)
    elif new_cache_logic not in cells[2]["source"]:
        raise RuntimeError("Unexpected Section 1 rate-cache code; update the repair script")

    cells[3]["source"] = """### 1.2 — Spot FX Rates

Yahoo tickers with `XXXUSD=X` already quote USD per foreign unit. For `USDXXX=X`, I invert the quote. Every resulting series is therefore **USD per one unit of foreign currency**. A rise means a gain for a long foreign-currency, short-USD position. USD is the numeraire and has spot price 1 USD, so it has no separate Yahoo ticker.

| Currency | Yahoo ticker | Conversion |
|---|---|---|
| EUR | `EURUSD=X` | direct |
| JPY | `USDJPY=X` | inverse |
| GBP | `GBPUSD=X` | direct |
| CHF | `USDCHF=X` | inverse |
| CAD | `USDCAD=X` | inverse |
| AUD | `AUDUSD=X` | direct |
| NZD | `NZDUSD=X` | direct |
| SEK | `USDSEK=X` | inverse |
| NOK | `USDNOK=X` | inverse; checked against Yahoo `EURUSD=X / EURNOK=X` |

Yahoo's direct NOK quote contains isolated bad prints, including `USDNOK=X` = 7.73532 on 2020-03-20, when its EUR/NOK cross implied roughly 11.35 and the [Federal Reserve's NOK quote](https://fred.stlouisfed.org/series/DEXNOUS) was 11.6842. I retain the direct Yahoo quote except when it disagrees with the independent Yahoo cross by more than 5%; those dates use the cross and are listed below. New Year's Day and Christmas Day quotes are excluded because the thin holiday feed contains artificial jumps. The raw quotes remain in the cache for inspection; this check changes only the analytical spot table. There is no smoothing or interpolation.
"""

    cells[4]["source"] = """### 1.3 — The Cache

On a fresh run, the code downloads each FRED series and the Yahoo close table to CSV files under `data_cache/`. Later runs read those raw files. Monthly FRED series are refreshed only if their latest observation is behind the conservatively expected published month. The daily EUR rate is checked against the last completed UTC business day, so an incomplete cached month cannot silently become a final monthly mean later; a holiday or delayed feed is retried at most once per UTC day unless the cache is clearly old. The Yahoo file is refreshed if it lacks the last completed UTC business day, with the same once-per-day guard for holidays or delayed quotes. Adding the independent `EURNOK=X` quality-check ticker invalidates an older cache without that column. Transformations and the data-quality check are recomputed on every run, so code changes do not reuse stale derived P&L. Delete `data_cache/` to force a full refresh. Paths are relative and no private API keys are needed.
"""

    cells[5]["source"] = """FX_TICKERS = {
    "EUR": ("EURUSD=X", False),
    "JPY": ("USDJPY=X", True),
    "GBP": ("GBPUSD=X", False),
    "CHF": ("USDCHF=X", True),
    "CAD": ("USDCAD=X", True),
    "AUD": ("AUDUSD=X", False),
    "NZD": ("NZDUSD=X", False),
    "SEK": ("USDSEK=X", True),
    "NOK": ("USDNOK=X", True),
}
NOK_CROSS_TICKER = "EURNOK=X"

def last_completed_fx_day(now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    return (now.normalize().tz_localize(None) - pd.offsets.BDay(1)).normalize()

def download_yahoo_close(tickers, end_date):
    raw = yf.download(
        tickers, start=START, end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False, threads=False,
    )
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    missing = [ticker for ticker in tickers if ticker not in close.columns or close[ticker].dropna().empty]
    if missing:
        raise RuntimeError(f"Yahoo returned no close prices for: {missing}")
    return close[tickers].sort_index()

def load_fx_spot(cache_dir=CACHE_DIR, now=None):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "yahoo_fx_close.csv"
    tickers = [FX_TICKERS[c][0] for c in FOREIGN] + [NOK_CROSS_TICKER]
    target_day = last_completed_fx_day(now)
    today = target_day + pd.offsets.BDay(1)
    cached = pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None
    if cached is not None:
        cached = cached.sort_index()
    cache_checked_today = path.exists() and pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").date() == pd.Timestamp.now(tz="UTC").date()
    complete_columns = cached is not None and set(tickers).issubset(cached.columns)
    has_history = cached is not None and not cached.empty and cached.index.min() <= pd.Timestamp(START)
    recent = cached is not None and not cached.empty and cached.index.max() >= target_day
    if not (complete_columns and has_history and (recent or cache_checked_today)):
        close = download_yahoo_close(tickers, today)
        close.to_csv(path)
        origin = "downloaded"
    else:
        close = cached[tickers]
        origin = "cache"
    if close.index.max() < target_day - pd.Timedelta(days=7):
        raise RuntimeError(f"Yahoo FX data are stale: last row {close.index.max().date()}")

    converted = {}
    for currency, (ticker, invert) in FX_TICKERS.items():
        values = pd.to_numeric(close[ticker], errors="coerce")
        converted[currency] = 1.0 / values if invert else values
    spot = pd.DataFrame(converted, index=close.index).where(lambda frame: frame > 0)

    # Check isolated NOK bad prints against an independent Yahoo triangulation.
    nok_cross = pd.to_numeric(close["EURUSD=X"], errors="coerce") / pd.to_numeric(close[NOK_CROSS_TICKER], errors="coerce")
    nok_cross = nok_cross.where(nok_cross > 0)
    discrepancy = (spot["NOK"] / nok_cross - 1).abs()
    holiday = ((spot.index.month == 1) & (spot.index.day == 1)) | ((spot.index.month == 12) & (spot.index.day == 25))
    repair = (discrepancy > 0.05) & ~holiday & nok_cross.notna()
    qc = pd.DataFrame({
        "NOK direct (USD/NOK inverse)": spot["NOK"],
        "NOK EUR cross": nok_cross,
        "relative disagreement": discrepancy,
    })
    qc["action"] = "direct quote kept"
    qc.loc[repair, "action"] = "EUR cross substituted"
    qc.loc[holiday, "action"] = "holiday row excluded"
    spot.loc[repair, "NOK"] = nok_cross.loc[repair]
    spot.loc[holiday, :] = np.nan
    qc = qc.loc[repair | holiday].copy()
    return spot, origin, qc

spot_usd_raw, fx_source, fx_qc_report = load_fx_spot()
print(f"Yahoo FX source: {fx_source}; rows: {len(spot_usd_raw):,}; latest date: {spot_usd_raw.index.max().date()}")
print(f"NOK cross substitutions: {(fx_qc_report['action'] == 'EUR cross substituted').sum():,}; holiday rows excluded: {(fx_qc_report['action'] == 'holiday row excluded').sum():,}")
display(fx_qc_report.loc[fx_qc_report["action"] == "EUR cross substituted"].round(4))
display(spot_usd_raw.tail())
"""

    cells[6]["source"] = """### 1.4 — Constructing Per-Currency Carry P&L

For currency $c$, I model a **long foreign currency, short USD** holding from the prior common FX observation to the current one. Spot log return is $100[\\log(S_{c,t})-\\log(S_{c,t-1})]$, with $S$ in USD per foreign unit. Each leg's annualized rate is compounded over the **actual calendar-day gap** $d$: $100[(1+r/100)^{d/365}-1]$. Carry P&L is spot log return plus accrued foreign interest minus accrued USD funding interest. The same convention is used for all currencies and downstream baskets. These are modeled log P&L percentage points, not executable dealer quotes.

I retain only dates where **all nine** FX quotes are present after the documented quality check. A missing date rolls into the next holding period's calendar-day accrual. Monthly rate observations become usable two calendar months after their observation month; I shift once more by one FX observation so each holding period uses information available at its start. Rows without a previous spot observation or all ten available rates are dropped explicitly. The independent NOK cross is used only for the isolated discrepancies listed above; ordinary large market moves, such as the 2015 CHF break, remain in the data.
"""

    cells[8]["source"] = """#### Section 1 interpretation

The summary below describes **always-long foreign-currency positions funded in USD**. It checks units, signs, and the effect of Yahoo's NOK data defects; it is not a claim of statistically significant carry. The 2020-03-20 `USDNOK=X` bad print would otherwise create an artificial gain near 33% followed by a loss near 42%, swamping the basket risk analysis. Triangulation corrects that isolated quote without changing genuine moves in the other eight currencies. The two-month rate lag is deliberately conservative, while EUR policy rates and CHF/NZD/SEK three-month proxies limit how literally the rate spread can be interpreted. Statistical significance and portfolio performance are assessed in later sections.
"""

    cells[9]["source"] = cells[9]["source"].replace(
        "pd.Timestamp.now(tz='UTC').date()",
        "pd.Timestamp.now(tz='America/New_York').date()",
    )
    return cells
