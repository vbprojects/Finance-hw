"""Notebook cells for Sections 2 and 3 of Case Study 2.

`get_cells()` returns nbformat-compatible cell descriptors (source is a string).
The cells expect the variables constructed by Section 1 of case_study2.ipynb.
"""


def _md(source):
    return {"cell_type": "markdown", "source": source.strip() + "\n"}


def _code(source):
    return {"cell_type": "code", "source": source.strip() + "\n"}


def get_cells():
    return [
        _md(r"""
## Section 2 — Is the single-pair story robust?

### 2.1 — Random walk and stationarity, basket-wide

All diagnostics below use the **same aligned daily sample** as the carry series. USD is the numeraire, so its USD/USD spot price is exactly one; it is included explicitly in the summary as a deterministic identity, not put through a unit-root test. The other nine USD-per-foreign-currency quotes are independent observable series. A failure to reject a unit root is *compatible* with a random walk, not proof of one; the return autocorrelations check a different implication.
"""),
        _code(r"""
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller, pacf

diagnostic_spot = spot_usd.loc[carry_pnl.index, FOREIGN].copy()
log_spot = np.log(diagnostic_spot)
spot_returns = log_spot.diff().dropna()
rw_rows = []
rw_acf = {}
for currency in FOREIGN:
    price = log_spot[currency].dropna()
    ret = spot_returns[currency].dropna()
    price_acf = acf(price, nlags=10, fft=True)
    return_acf = acf(ret, nlags=10, fft=True)
    rw_acf[currency] = (price_acf, return_acf)
    price_adf_p = adfuller(price, regression="c", maxlag=10,
                           autolag="AIC", result_object=False)[1]
    return_adf_p = adfuller(ret, regression="c", maxlag=10,
                            autolag="AIC", result_object=False)[1]
    rw_rows.append({
        "currency": currency,
        "log-price ACF(1)": price_acf[1],
        "return ACF(1)": return_acf[1],
        "largest |return ACF|, lags 1–10": np.max(np.abs(return_acf[1:])),
        "ADF price p": price_adf_p,
        "ADF return p": return_adf_p,
    })
rw_summary = pd.DataFrame(rw_rows).set_index("currency")
rw_summary.loc["USD", :] = np.nan
rw_summary = rw_summary.reindex(CURRENCIES)
rw_summary["spot status"] = "market quote"
rw_summary.loc["USD", "spot status"] = "deterministic numeraire"
display(rw_summary.round(4))

fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
lags = np.arange(1, 11)
for ax, currency in zip(axes.flat, FOREIGN):
    price_acf, return_acf = rw_acf[currency]
    ax.plot(lags, price_acf[1:], "o-", ms=3, lw=1, label="Log price")
    ax.plot(lags, return_acf[1:], "s-", ms=3, lw=1, label="Log return")
    ax.axhline(0, color="black", lw=.7)
    ax.axhline(1.96 / np.sqrt(len(spot_returns)), color="gray", ls=":", lw=.8)
    ax.axhline(-1.96 / np.sqrt(len(spot_returns)), color="gray", ls=":", lw=.8)
    ax.set_title(currency)
    ax.set_ylim(-.2, 1.08)
    ax.grid(alpha=.2)
for ax in axes[-1, :]:
    ax.set_xlabel("Lag (trading days)")
for ax in axes[:, 0]:
    ax.set_ylabel("Sample autocorrelation")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .96), ncol=2)
fig.suptitle("G10 spot diagnostics: level persistence versus return memory", y=.995)
fig.tight_layout(rect=(0, 0, 1, .91))
plt.show()

price_rejections = rw_summary.index[(rw_summary["ADF price p"] < .05).fillna(False)].tolist()
return_rejections = rw_summary.index[(rw_summary["ADF return p"] < .05).fillna(False)].tolist()
largest_return_acf = rw_summary.loc[FOREIGN, "largest |return ACF|, lags 1–10"].idxmax()
display(Markdown(
    f"At the 5% ADF level, the log-price unit-root null is rejected for "
    f"**{', '.join(price_rejections) if price_rejections else 'none of the nine traded currencies'}**; "
    f"the return unit-root null is rejected for "
    f"**{', '.join(return_rejections) if return_rejections else 'none'}**. "
    f"**{largest_return_acf}** has the largest absolute short-lag return autocorrelation "
    f"({rw_summary.loc[largest_return_acf, 'largest |return ACF|, lags 1–10']:.3f}). "
    "Thus any exception should be read alongside both the level and return diagnostics. "
    + ("CHF is the level-test exception here; its history includes the 2015 end of the franc/euro floor, "
       "although this test alone cannot attribute the result to that regime change. "
       if "CHF" in price_rejections else "")
    +
    "USD/USD is constant by construction and cannot be diagnosed as a stochastic random walk."
))
"""),
        _md(r"""
### 2.2 — Is average carry positive?

The lecture's IID mean test uses $t=\bar r/(s/\sqrt n)$ and a two-sided 5% threshold. The series are daily P&L in **percentage points**, so the mean and standard error below are converted to **basis points per day**. These p-values are descriptive: volatility clustering and testing nine currencies can make nominal significance look stronger than a robust strategy result. The bootstrap next revisits selected legs.
"""),
        _code(r"""
rate_spread_annual_pct = rates_at_start[FOREIGN].sub(rates_at_start["USD"], axis=0).loc[carry_pnl.index]
mean_rows = []
for currency in FOREIGN:
    sample = carry_pnl[currency].dropna()
    n = len(sample)
    mean_pct = sample.mean()
    se_pct = sample.std(ddof=1) / np.sqrt(n)
    t_stat = mean_pct / se_pct
    p_two_sided = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    mean_rows.append({
        "currency": currency,
        "n days": n,
        "mean daily P&L (bp)": mean_pct * 100,
        "SE (bp)": se_pct * 100,
        "t statistic": t_stat,
        "two-sided p": p_two_sided,
        "mean rate spread (annual %pt)": rate_spread_annual_pct[currency].mean(),
        "annualized mean P&L (%)": mean_pct * 252,
    })
mean_test_summary = pd.DataFrame(mean_rows).set_index("currency")
mean_test_summary["positive at 5%?"] = (
    (mean_test_summary["mean daily P&L (bp)"] > 0)
    & (mean_test_summary["two-sided p"] < .05)
)
display(mean_test_summary.sort_values("mean rate spread (annual %pt)", ascending=False).round(4))

positive_carry = mean_test_summary.index[mean_test_summary["positive at 5%?"]].tolist()
not_positive = [c for c in FOREIGN if c not in positive_carry]
negative_significant = mean_test_summary.index[
    (mean_test_summary["mean daily P&L (bp)"] < 0)
    & (mean_test_summary["two-sided p"] < .05)
].tolist()
rank_corr = stats.spearmanr(
    mean_test_summary["mean rate spread (annual %pt)"],
    mean_test_summary["annualized mean P&L (%)"],
).statistic
highest_yield = mean_test_summary["mean rate spread (annual %pt)"].idxmax()
highest_pnl = mean_test_summary["annualized mean P&L (%)"].idxmax()
display(Markdown(
    f"By this IID two-sided test, **{', '.join(positive_carry) if positive_carry else 'none'}** "
    f"show significant positive mean carry; **{', '.join(not_positive) if not_positive else 'none'}** "
    f"do not. Significant *negative* mean carry appears for "
    f"**{', '.join(negative_significant) if negative_significant else 'none'}**. "
    "The cross-currency Spearman rank correlation between average quoted "
    f"rate differential and realized average P&L is **{rank_corr:.2f}**. "
    f"The highest average yield spread belongs to **{highest_yield}**, while the "
    f"highest realized mean P&L belongs to **{highest_pnl}**. "
    "This is a direct check on whether higher-yield currencies also deliver more carry; "
    "the spot component can overwhelm the interest differential even when UIP fails on average."
))
"""),
        _md(r"""
### 2.3 — Bootstrap the lecture pair and two cross-sectional contrasts

Long AUD/short JPY is reconstructed as the **AUD/USD long carry minus JPY/USD long carry** on identical dates. USD spot changes and USD funding then cancel algebraically, leaving AUD/JPY spot P&L plus the AUD-minus-JPY accrued differential. I use 1,000 IID day resamples with a fixed seed, as requested. The bootstrap interval measures sensitivity to which days appear in this sample; it does not preserve volatility clusters or establish out-of-sample profitability.
"""),
        _code(r"""
aud_jpy_pnl = (carry_pnl["AUD"] - carry_pnl["JPY"]).rename("AUD/JPY")
high_spread_currency = mean_test_summary["mean rate spread (annual %pt)"].idxmax()
low_spread_currency = mean_test_summary["mean rate spread (annual %pt)"].idxmin()
bootstrap_samples = {
    "AUD/JPY": aud_jpy_pnl,
    f"{high_spread_currency}/USD": carry_pnl[high_spread_currency],
    f"{low_spread_currency}/USD": carry_pnl[low_spread_currency],
}
# If AUD or JPY is an extreme, AUD/JPY remains a distinct cross-currency trade.
rng = np.random.default_rng(20260930)
boot_means = {}
bootstrap_rows = []
for name, series in bootstrap_samples.items():
    values_bp = series.dropna().to_numpy(dtype=float) * 100
    draws = np.array([rng.choice(values_bp, size=len(values_bp), replace=True).mean()
                      for _ in range(1000)])
    boot_means[name] = draws
    lo, hi = np.quantile(draws, [.025, .975])
    bootstrap_rows.append({
        "trade": name,
        "sample mean (bp/day)": values_bp.mean(),
        "bootstrap 95% low (bp/day)": lo,
        "bootstrap 95% high (bp/day)": hi,
        "share of resamples > 0": (draws > 0).mean(),
    })
bootstrap_summary = pd.DataFrame(bootstrap_rows).set_index("trade")
display(bootstrap_summary.round(4))

fig, axes = plt.subplots(1, len(boot_means), figsize=(13, 3.5))
for ax, (name, draws) in zip(np.atleast_1d(axes), boot_means.items()):
    row = bootstrap_summary.loc[name]
    ax.hist(draws, bins=30, density=True, color="steelblue", alpha=.75)
    ax.axvline(0, color="black", ls="--", label="Zero")
    ax.axvline(row["sample mean (bp/day)"], color="darkorange", lw=2, label="Sample mean")
    ax.axvspan(row["bootstrap 95% low (bp/day)"], row["bootstrap 95% high (bp/day)"],
               color="darkorange", alpha=.18, label="95% bootstrap interval")
    ax.set_title(name)
    ax.set_xlabel("Bootstrapped mean (bp per trading day)")
    ax.grid(alpha=.2)
axes[0].set_ylabel("Density")
axes[-1].legend(fontsize=8)
fig.suptitle("IID bootstrap distributions of daily carry means (1,000 resamples)")
fig.tight_layout()
plt.show()

aud_boot = bootstrap_summary.loc["AUD/JPY"]
if aud_boot["bootstrap 95% low (bp/day)"] > 0:
    aud_boot_conclusion = "The interval is entirely positive, so its positive sample mean survives this day-resampling check. "
elif aud_boot["bootstrap 95% high (bp/day)"] < 0:
    aud_boot_conclusion = "The interval is entirely negative, contradicting a positive-mean claim for this sample. "
else:
    aud_boot_conclusion = "The interval includes zero, so the lecture pair's positive-mean claim is fragile under this resampling check. "
display(Markdown(
    f"For the lecture's AUD/JPY trade, the sample mean is "
    f"**{aud_boot['sample mean (bp/day)']:.3f} bp/day** and the IID bootstrap "
    f"95% interval is **[{aud_boot['bootstrap 95% low (bp/day)']:.3f}, "
    f"{aud_boot['bootstrap 95% high (bp/day)']:.3f}] bp/day**. "
    + aud_boot_conclusion
    + "The interval cannot rule out regime shifts or clustered crash losses."
))
"""),
        _md(r"""
## Section 3 — Mean structure, volatility, and UIP across the basket

The ARMA comparison uses the most recent **1,250 common daily observations** (about five years) so the bounded grid remains practical to rerun and focuses on a coherent recent regime. The GARCH fits use the full aligned carry sample, where more tail observations help estimate volatility. These are explanatory full-sample models; Section 5 must refit using only history available at each rebalance before using forecasts to trade.
"""),
        _md(r"""
### 3.1 — ARMA identification

The ACF/PACF figure shows lags 1–10 for each daily carry series. I compare ARMA(0,0), (1,0), (0,1), (1,1), (2,0), and (0,2) with a constant, choosing the **lowest BIC**. The BIC improvement over white noise is evidence of mean structure, but a small improvement or a nonzero order alone is not evidence of useful out-of-sample predictability.
"""),
        _code(r"""
from statsmodels.tsa.arima.model import ARIMA

ARMA_MAX_OBS = 1250
arma_sample = carry_pnl.tail(ARMA_MAX_OBS)
arma_orders = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]

fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True)
for ax, currency in zip(axes.flat, FOREIGN):
    y = arma_sample[currency].dropna().to_numpy()
    acf_vals = acf(y, nlags=10, fft=True)[1:]
    pacf_vals = pacf(y, nlags=10, method="ywm")[1:]
    ax.plot(range(1, 11), acf_vals, "o-", ms=3, label="ACF")
    ax.plot(range(1, 11), pacf_vals, "s-", ms=3, label="PACF")
    band = 1.96 / np.sqrt(len(y))
    ax.axhline(0, color="black", lw=.7)
    ax.axhline(band, color="gray", ls=":", lw=.8)
    ax.axhline(-band, color="gray", ls=":", lw=.8)
    ax.set_title(currency)
    ax.grid(alpha=.2)
for ax in axes[-1, :]:
    ax.set_xlabel("Lag (trading days)")
for ax in axes[:, 0]:
    ax.set_ylabel("Sample autocorrelation")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .96), ncol=2)
fig.suptitle(f"Daily carry ACF/PACF: {arma_sample.index.min().date()}–{arma_sample.index.max().date()}", y=.995)
fig.tight_layout(rect=(0, 0, 1, .91))
plt.show()

def fit_arma_grid(series, orders=arma_orders):
    results = {}
    y = series.dropna().to_numpy(dtype=float)
    for p, q in orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(y, order=(p, 0, q), trend="c").fit(
                    method_kwargs={"maxiter": 50}
                )
            if np.isfinite(fit.bic):
                results[(p, q)] = fit
        except (ValueError, np.linalg.LinAlgError):
            continue
    if (0, 0) not in results:
        raise RuntimeError("ARMA white-noise baseline failed to fit")
    best_order = min(results, key=lambda order: results[order].bic)
    return best_order, results

arma_fits = {}
arma_rows = []
for currency in FOREIGN:
    best_order, candidate_fits = fit_arma_grid(arma_sample[currency])
    arma_fits[currency] = candidate_fits[best_order]
    bic_gain = candidate_fits[(0, 0)].bic - candidate_fits[best_order].bic
    arma_rows.append({
        "currency": currency,
        "best ARMA(p,q) by BIC": f"({best_order[0]},{best_order[1]})",
        "best BIC": candidate_fits[best_order].bic,
        "BIC gain vs (0,0)": bic_gain,
        "clear mean structure?": best_order != (0, 0) and bic_gain >= 10,
    })
arma_summary = pd.DataFrame(arma_rows).set_index("currency")
display(arma_summary.round(2))
mean_outliers = arma_summary.index[arma_summary["clear mean structure?"]].tolist()
display(Markdown(
    f"On the common recent sample, **{', '.join(mean_outliers) if mean_outliers else 'no currency'}** "
    "has a BIC improvement of at least 10 over a constant-mean white-noise model. "
    "This threshold identifies statistical outliers in the basket; even for them, "
    "an in-sample ARMA gain must survive walk-forward testing before it can be called exploitable."
))
"""),
        _md(r"""
### 3.2 — GARCH(1,1) for each currency

I fit Normal and Student-$t$ innovations to each full-sample daily carry series. The table reports parameters from the **AIC-winning** fit and shows both AIC and BIC distribution choices. Persistence is $\hat\alpha_1+\hat\beta_1$. Ljung–Box p-values at 10 lags assess remaining autocorrelation in standardized residuals and their squares; an ARCH-LM p-value on demeaned raw P&L gives a direct preliminary test of volatility clustering. A high persistence estimate does not mean the unconditional volatility is constant, nor does a small Ljung–Box p-value validate a model.
"""),
        _code(r"""
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

garch_fits = {}                 # AIC-winning fitted results, one per currency.
garch_candidates = {}           # Both Normal and Student-t results per currency.
garch_rows = []
for currency in FOREIGN:
    y = carry_pnl[currency].dropna().to_numpy(dtype=float)
    candidates = {}
    for distribution in ("normal", "t"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1,
                             dist=distribution, rescale=False).fit(disp="off")
        if not np.isfinite(fit.aic):
            raise RuntimeError(f"Non-finite GARCH likelihood for {currency} ({distribution})")
        candidates[distribution] = fit
    garch_candidates[currency] = candidates
    aic_winner = min(candidates, key=lambda dist: candidates[dist].aic)
    bic_winner = min(candidates, key=lambda dist: candidates[dist].bic)
    best = candidates[aic_winner]
    garch_fits[currency] = best
    z = np.asarray(best.std_resid, dtype=float)
    z = z[np.isfinite(z)]
    lb_p = acorr_ljungbox(z, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    lb_sq_p = acorr_ljungbox(z ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        arch_p = het_arch(y - y.mean(), nlags=10)[1]
    alpha = best.params["alpha[1]"]
    beta = best.params["beta[1]"]
    garch_rows.append({
        "currency": currency,
        "alpha1": alpha,
        "beta1": beta,
        "persistence": alpha + beta,
        "AIC winner": "Student-t" if aic_winner == "t" else "Normal",
        "BIC winner": "Student-t" if bic_winner == "t" else "Normal",
        "AIC normal": candidates["normal"].aic,
        "AIC t": candidates["t"].aic,
        "LB residual p (10)": lb_p,
        "LB squared p (10)": lb_sq_p,
        "ARCH-LM p (10)": arch_p,
    })
garch_summary = pd.DataFrame(garch_rows).set_index("currency")
display(garch_summary.round(4))

clustered = garch_summary.index[garch_summary["ARCH-LM p (10)"] < .05].tolist()
strongest = garch_summary["persistence"].idxmax()
weakest = garch_summary["persistence"].idxmin()
leftover_vol = garch_summary.index[garch_summary["LB squared p (10)"] < .05].tolist()
t_aic_count = (garch_summary["AIC winner"] == "Student-t").sum()
display(Markdown(
    f"The raw-return ARCH-LM test detects clustering at 5% in "
    f"**{', '.join(clustered) if clustered else 'none'}**. Estimated GARCH "
    f"persistence is highest in **{strongest}** "
    f"({garch_summary.loc[strongest, 'persistence']:.3f}) and lowest in **{weakest}** "
    f"({garch_summary.loc[weakest, 'persistence']:.3f}). "
    f"Squared standardized residuals still reject no autocorrelation for "
    f"**{', '.join(leftover_vol) if leftover_vol else 'none'}**, pointing to "
    "remaining volatility structure if any. These relative estimates show why "
    "equal notional across all legs is not equal risk. "
    f"Student-$t$ innovations win AIC in **{t_aic_count} of {len(FOREIGN)}** fits, "
    "consistent with heavier tails than a Normal shock model."
))
"""),
        _md(r"""
### 3.3 — Fama/UIP regression across the basket

There is a quote-direction subtlety. Section 1 defines $S$ as **USD per foreign unit** for long-currency P&L. The prompt's stated UIP null $\beta=1$ instead corresponds to $s=\log(\text{foreign units per USD})=-\log S$: a high foreign rate predicts foreign-currency depreciation, so $\Delta s$ should be positive. I therefore regress the *next month's* change in this inverse log quote on the **foreign-minus-USD interest differential accrued over that month's actual calendar-day span**. Both sides are in percentage points, making the UIP benchmark $\beta=1$. The rate at month-end $t$ is drawn from the conservatively publication-lagged Section 1 series; the spot change begins only after $t$. Standard errors are HAC with three monthly lags. A coefficient below one means the expected depreciation falls short of the quoted interest advantage, which supports positive gross long-foreign carry for positive spreads; it does not by itself guarantee a profitable cost-adjusted basket.
"""),
        _code(r"""
import statsmodels.api as sm

# Retain the actual date of each month's final common FX observation.
month_end_spot = spot_usd.loc[carry_pnl.index, FOREIGN].groupby(
    spot_usd.loc[carry_pnl.index].index.to_period("M")
).tail(1)
month_end_rates = rates_known.reindex(month_end_spot.index, method="ffill")
next_days = (month_end_spot.index.to_series().shift(-1) -
             month_end_spot.index.to_series()).dt.days
monthly_foreign_accrual = accrued_pct(month_end_rates[FOREIGN], next_days)
monthly_usd_accrual = accrued_pct(month_end_rates["USD"], next_days)
monthly_spread = monthly_foreign_accrual.sub(monthly_usd_accrual, axis=0)
next_inverse_spot_change = -100 * (np.log(month_end_spot).shift(-1) - np.log(month_end_spot))

fama_models = {}
fama_rows = []
for currency in FOREIGN:
    reg_data = pd.DataFrame({
        "next inverse spot change (%)": next_inverse_spot_change[currency],
        "accrued foreign-minus-USD spread (%)": monthly_spread[currency],
    }).dropna()
    if len(reg_data) < 36 or reg_data.iloc[:, 1].std() == 0:
        raise RuntimeError(f"Insufficient variation for {currency} monthly Fama regression")
    explanatory = sm.add_constant(reg_data.iloc[:, 1], has_constant="add")
    model = sm.OLS(reg_data.iloc[:, 0], explanatory).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    fama_models[currency] = model
    beta = model.params.iloc[1]
    se = model.bse.iloc[1]
    fama_rows.append({
        "currency": currency,
        "months": len(reg_data),
        "beta": beta,
        "HAC SE": se,
        "95% CI low": beta - 1.96 * se,
        "95% CI high": beta + 1.96 * se,
        "below UIP β=1 at 5%?": beta + 1.96 * se < 1,
        "R-squared": model.rsquared,
    })
fama_summary = pd.DataFrame(fama_rows).set_index("currency")
display(fama_summary.round(3))

below_one = fama_summary.index[fama_summary["beta"] < 1].tolist()
below_zero = fama_summary.index[fama_summary["beta"] < 0].tolist()
reject_uip_below = fama_summary.index[fama_summary["below UIP β=1 at 5%?"]].tolist()
if reject_uip_below:
    fama_basket_read = (
        "Those statistically below one provide pair-specific evidence that foreign-currency "
        "depreciation has failed to offset the rate advantage fully. They motivate testing a "
        "cross-sectional carry basket, though returns and costs remain decisive."
    )
else:
    fama_basket_read = (
        "None has a 95% upper bound below one, so these regressions do **not** establish "
        "a basket-wide forward-premium puzzle. Section 4 is an exploratory carry backtest, "
        "and any allocation case must rest on its realized, cost-adjusted results."
    )
display(Markdown(
    f"Point estimates are below UIP's **β=1** for **{', '.join(below_one) if below_one else 'none'}** "
    f"and negative for **{', '.join(below_zero) if below_zero else 'none'}**. "
    f"The upper 95% confidence bound is below one for "
    f"**{', '.join(reject_uip_below) if reject_uip_below else 'none'}**. "
    "A point estimate below one alone is weak evidence when its confidence interval is wide. "
    + fama_basket_read
))
"""),
    ]
