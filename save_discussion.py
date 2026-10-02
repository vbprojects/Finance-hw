"""Add dated, human-readable discussion of the executed 2026-09-30 run."""

from pathlib import Path

import nbformat


path = Path(__file__).with_name("case_study2.ipynb")
nb = nbformat.read(path, as_version=4)

discussions = {
    "section1_summary =": """#### Saved Section 1 discussion — data as of 2026-09-30

The common sample has **4,295** FX holding periods from 2010-03-02 through 2026-09-30. The always-long USD-funded legs do not automatically earn positive carry: CHF has the highest annualized sample mean at about **−0.07%**, while JPY has the lowest at **−4.69%**. These are descriptive means of individual legs, before any monthly rank-and-sort strategy. The 2020 NOK quote defect was corrected against the independent EUR/NOK cross; the corrected NOK worst modeled period is **−5.15%**, so a single bad print no longer dominates the risk results. The EUR policy rate and CHF, NZD, and SEK three-month rate proxies make the cross-country spreads approximate rather than executable dealer funding rates.
""",
    "aud_jpy_pnl =": """#### Saved Section 2 discussion — data as of 2026-09-30

The nine traded log spot levels look highly persistent. CHF is the only level series for which the ADF test rejects a unit root at 5% (**p = 0.030**); the other eight do not. All nine return series reject a unit root, and even the largest absolute short-lag return autocorrelation, SEK's **0.057**, is small. USD/USD is a deterministic numeraire, so no unit-root test applies to it.

The IID mean tests find **no significantly positive** USD-funded carry leg at 5%; JPY has a significantly *negative* mean (**p = 0.035**). Average rate spread and realized carry mean have only a **0.25** Spearman rank correlation across currencies: NZD has the highest average spread, yet CHF has the highest (still slightly negative) realized mean. The AUD/JPY trade averages **1.672 bp per day**, but its 1,000-resample IID bootstrap 95% interval is **[−0.534, 4.051] bp per day**. That interval crosses zero, so the apparent positive single-pair mean is fragile; an IID bootstrap also cannot reproduce clustered crash episodes.
""",
    "import statsmodels.api as sm": """#### Saved Section 3 discussion — data as of 2026-09-30

On the recent 1,250-observation comparison window, **ARMA(0,0)** has the best BIC for all nine carry series. The full-sample GARCH fits tell a different story about variance: Student-t innovations win AIC and BIC for every currency, and estimated volatility persistence ranges from **0.986 for CHF** to **0.998 for EUR**. Squared standardized residuals still show a Ljung–Box signal for GBP and NOK, so a GARCH(1,1) fit does not capture every feature of their volatility.

The monthly Fama estimates are imprecise. AUD's point estimate is **−0.308**, but its HAC 95% interval is **[−3.110, 2.493]**; no currency's interval excludes the UIP benchmark of beta = 1 on the low side. These regressions do **not** establish a basket-wide forward-premium puzzle. The portfolio test that follows is therefore exploratory and must stand on its out-of-sample timing and net results, rather than an asserted universal anomaly.
""",
    "weights_pair =": """#### Saved Section 4 discussion — data as of 2026-09-30

The walk-forward comparison covers **3,776** common holding days beginning 2012-03-01. At the base-case 5 bp one-way turnover charge and 10 bp annual spread, the equal-weight basket earns **0.65%** annualized with Sharpe **0.20** and maximum drawdown **−9.04%**. The inverse-volatility basket earns **0.48%**, Sharpe **0.15**, and drawdown **−9.37%**. The 100%-gross AUD/JPY benchmark has the highest Sharpe, **0.30**, but a much deeper **−18.25%** drawdown. Basket diversification reduced the worst peak-to-trough loss relative to that pair, yet did not improve risk-adjusted return; inverse-volatility sizing did not improve the equal-weight basket in this sample.

Friction reduces the inverse-volatility annualized mean from **0.65% gross to 0.48% net**. Under the more severe 10 bp turnover and 25 bp spread assumptions, its mean falls to **0.27%**. The nine standalone legs have mean pairwise correlation **0.51**, and an average **4.5 of six** active legs lose together on the worst 5% of inverse-volatility basket days. That common risk limits the protection gained by holding several currencies.
""",
    "eqm = performance_metrics(equal_net)": """#### Saved Section 5 discussion and decision — data as of 2026-09-30

The net inverse-volatility basket's annualized mean is **0.48%**, volatility **3.33%**, and Sharpe **0.15**. Its skewness is **−1.01**, excess kurtosis **9.63**, and maximum drawdown **−9.37%**; costs and spread consume **2.49% of initial notional** over the 14.6-year live sample. The latest one-day GARCH-t 95% gross-market VaR is **0.203%**, and realized gross losses exceeded the advance threshold on **223 of 3,776** days (**5.9%**). The VaR is an approximate marginal-GARCH portfolio forecast and should not be mistaken for a crash-loss bound.

During the August 2024 JPY unwind, the worst early-August gross loss was **0.88% on August 5**, versus a prior-close VaR of **0.25%**. VaR rose to **0.31%** on the next holding day, after the loss. This is more consistent with collecting a small premium while bearing occasional large losses than with a smooth, dependable edge. I would **not recommend a full-risk allocation** from this evidence: the net Sharpe is low, the tails are adverse, and the alternative sizing did not reduce drawdown. The first change I would test is a fixed-budget JPY call hedge against a funding-currency unwind, with the option premium included and tail-risk improvement evaluated out of sample before any capital decision.
""",
    "trend_weights =": """#### Saved bonus discussion — data as of 2026-09-30

The 63-observation trend filter improves the July–September 2024 episode from **−1.97%** to **−1.03%** modeled P&L, but it does not improve the full sample. Annualized net return changes from **+0.48% to −0.42%**, maximum drawdown worsens from **−9.37% to −13.89%**, and annualized turnover rises from **1.37 to 4.49 times capital**. Skewness becomes slightly less negative (**−1.01 to −0.92**), while excess kurtosis rises (**9.63 to 12.34**). The extra trading and missed carry overwhelm this simple filter's local crash benefit; I would reject this version and test a different risk control before deployment.
""",
}

new_cells = []
found = set()
for cell in nb.cells:
    # Keep this script safe to rerun without duplicating dated discussion cells.
    if cell.cell_type == "markdown" and cell.source.startswith("#### Saved "):
        continue
    new_cells.append(cell)
    if cell.cell_type != "code":
        continue
    for prefix, prose in discussions.items():
        if cell.source.startswith(prefix):
            new_cells.append(nbformat.v4.new_markdown_cell(prose))
            found.add(prefix)
if found != set(discussions):
    raise RuntimeError(f"Missing discussion anchors: {set(discussions) - found}")

nb.cells = new_cells
nbformat.validate(nb)
nbformat.write(nb, path)
print(f"Saved {len(discussions)} dated discussion cells in {path}")
