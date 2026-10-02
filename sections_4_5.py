"""Notebook cells for Sections 4–6 of Case Study 2.

This is a build helper. The submitted notebook is self-contained after these
cells are inserted by the notebook assembly script.
"""


def _md(source):
    return {"cell_type": "markdown", "source": source}


def _code(source):
    return {"cell_type": "code", "source": source}


def get_cells():
    return [
        _md(r"""## Section 4 — Building the G10 Carry Basket

### 4.1 Rank-and-sort basket

I rebalance **after the last common FX close of each month**, so the resulting positions first earn the return on the next common FX observation (the first one of the next month). The rank signal uses the Section 1 rates *known by that close*: a monthly observation has already been delayed by two months. At each rebalance I buy the three highest foreign-minus-USD rates and short the three lowest. Equal weights are +1/6 and −1/6, giving 50% long notional, 50% short notional, and 100% gross exposure.

### 4.2 Volatility-targeted basket

The risk-parity version retains the same selected currencies and normalizes inverse **one-step-ahead GARCH-t forecast volatility** separately within the long and short books to 50% per side.

The GARCH fits in Section 3 are descriptive full-sample estimates; using them directly to size earlier trades would leak future information. The live backtest below instead starts after 504 common observations, refits GARCH(1,1)-Student-t models on at most the trailing 756 observations **quarterly**, and updates their variance forecasts with each newly observed return. Quarterly parameter refreshes are a practical compromise: volatility responds daily through the GARCH recursion, while its slower-moving parameters need not be re-estimated daily. Every forecast used for the next holding period depends only on data through the previous close. The correlation estimate used for portfolio VaR similarly uses only the trailing 252 observed days.

Daily P&L is the weight times each Section 1 carry P&L, in percentage points. Within a month weights are held fixed; there is no day-by-day rebalancing back to target weights. This is a constant-notional approximation rather than a fully self-financing mark-to-market portfolio."""),
        _code(r"""import warnings
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model


def _fit_garch_t(history_pct):
    # Fit one currency using only observations available at the signal close.
    model = arch_model(
        history_pct.to_numpy(dtype=float), mean="Constant", vol="GARCH",
        p=1, q=1, dist="StudentsT", rescale=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(disp="off", show_warning=False, options={"maxiter": 250})
    if fit.convergence_flag != 0:
        raise RuntimeError(f"GARCH-t optimizer returned flag {fit.convergence_flag}")
    par = fit.params
    next_variance = float(fit.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
    if not np.isfinite(next_variance) or next_variance <= 0:
        raise RuntimeError("Non-positive one-step GARCH variance")
    return {
        "mu": float(par["mu"]), "omega": float(par["omega"]),
        "alpha": float(par["alpha[1]"]), "beta": float(par["beta[1]"]),
        "nu": float(par["nu"]), "next_variance": next_variance,
    }


def _inverse_vol_weights(high, low, next_variance, currencies):
    out = pd.Series(0.0, index=currencies)
    high_inv = 1.0 / np.sqrt(next_variance[high])
    low_inv = 1.0 / np.sqrt(next_variance[low])
    out.loc[high] = 0.5 * high_inv / high_inv.sum()
    out.loc[low] = -0.5 * low_inv / low_inv.sum()
    return out


def walk_forward_g10(carry, rate_data, min_history=504, fit_window=756):
    # A signal at index i sets weights/VaR for i+1, never for i.
    # Refit failures reuse only previously known parameters and are logged.
    carry = carry[FOREIGN].sort_index().dropna(how="any")
    dates = carry.index
    if len(carry) < min_history + 30:
        raise ValueError("Insufficient common history for a walk-forward test")
    # Section 1 applied its two-month release lag to rates_known already.
    known = rate_data.reindex(dates, method="ffill")[CURRENCIES]
    differentials = known[FOREIGN].sub(known["USD"], axis=0)
    if differentials.isna().any().any():
        raise ValueError("Missing as-of rate differential in the common sample")

    w_equal = pd.DataFrame(0.0, index=dates, columns=FOREIGN)
    w_vol = pd.DataFrame(0.0, index=dates, columns=FOREIGN)
    var95 = pd.Series(np.nan, index=dates, name="ex_ante_95pct_VaR_pct")
    sigma_forecast = pd.Series(np.nan, index=dates, name="ex_ante_vol_pct")
    active_equal = pd.Series(0.0, index=FOREIGN)
    active_vol = pd.Series(0.0, index=FOREIGN)
    parameters = {}
    next_var = pd.Series(np.nan, index=FOREIGN)
    refits, failures, rebalances = [], [], []

    for i in range(len(dates) - 1):
        # At this close y_i has just been observed. The recursion yields the
        # variance of y_(i+1), from parameters estimated in earlier closes.
        if parameters:
            for c in FOREIGN:
                p = parameters[c]
                residual = float(carry.iloc[i][c] - p["mu"])
                next_var[c] = max(
                    p["omega"] + p["alpha"] * residual**2 +
                    p["beta"] * float(next_var[c]), 1e-10,
                )

        is_month_end = dates[i].to_period("M") != dates[i + 1].to_period("M")
        if is_month_end and i + 1 >= min_history:
            trade_month = dates[i + 1].to_period("M")
            must_refit = not parameters or trade_month.month in (1, 4, 7, 10)
            if must_refit:
                history = carry.iloc[max(0, i + 1 - fit_window):i + 1]
                for c in FOREIGN:
                    try:
                        fit_info = _fit_garch_t(history[c])
                        parameters[c] = fit_info
                        next_var[c] = fit_info["next_variance"]
                        refits.append((dates[i], c, len(history)))
                    except Exception as exc:
                        if c not in parameters:
                            raise RuntimeError(
                                f"Initial GARCH-t fit failed for {c} on {dates[i].date()}"
                            ) from exc
                        failures.append((dates[i], c, str(exc)))
            if not parameters:
                raise RuntimeError("No GARCH models available at first rebalance")
            ranks = differentials.iloc[i].sort_values()
            low = list(ranks.index[:3])
            high = list(ranks.index[-3:])
            active_equal = pd.Series(0.0, index=FOREIGN)
            active_equal.loc[high] = 1.0 / 6.0
            active_equal.loc[low] = -1.0 / 6.0
            active_vol = _inverse_vol_weights(high, low, next_var, FOREIGN)
            rebalances.append({
                "signal_close": dates[i], "first_holding_day": dates[i + 1],
                "long": ", ".join(high), "short": ", ".join(low),
                "top_rate_spread_pp": float(ranks.iloc[-1]),
                "bottom_rate_spread_pp": float(ranks.iloc[0]),
            })

        # Both the position and the risk forecast stamped on i+1 were set at
        # the close of i, before the return at i+1 exists.
        w_equal.iloc[i + 1] = active_equal
        w_vol.iloc[i + 1] = active_vol
        if active_vol.abs().sum() > 0:
            risk_history = carry.iloc[max(0, i - 251):i + 1]
            corr = risk_history.corr().fillna(0.0).to_numpy(copy=True)
            np.fill_diagonal(corr, 1.0)
            leg_sigma = np.sqrt(next_var.to_numpy(dtype=float))
            weighted_sigma = active_vol.to_numpy(dtype=float) * leg_sigma
            variance = float(weighted_sigma @ corr @ weighted_sigma)
            portfolio_sigma = np.sqrt(max(variance, 1e-12))
            selected = [c for c in FOREIGN if active_vol[c] != 0]
            # A common-t approximation uses the heaviest marginal tails.
            nu = max(2.05, min(parameters[c]["nu"] for c in selected))
            t05_unit_variance = stats.t.ppf(0.05, df=nu) * np.sqrt((nu - 2) / nu)
            conditional_mean = sum(active_vol[c] * parameters[c]["mu"] for c in FOREIGN)
            var95.iloc[i + 1] = max(0.0, -(conditional_mean + t05_unit_variance * portfolio_sigma))
            sigma_forecast.iloc[i + 1] = portfolio_sigma

    live = w_vol.abs().sum(axis=1).gt(0)
    first_live = int(np.flatnonzero(live.to_numpy())[0])
    return {
        "equal_weights": w_equal.iloc[first_live:],
        "vol_weights": w_vol.iloc[first_live:],
        "var95": var95.iloc[first_live:],
        "sigma_forecast": sigma_forecast.iloc[first_live:],
        "rebalances": pd.DataFrame(rebalances).set_index("first_holding_day"),
        "refits": pd.DataFrame(refits, columns=["date", "currency", "window_rows"]),
        "fit_failures": pd.DataFrame(failures, columns=["date", "currency", "reason"]),
    }


wf = walk_forward_g10(carry_pnl, rates_known)
weights_equal = wf["equal_weights"]
weights_vol = wf["vol_weights"]
live_dates = weights_vol.index
assert weights_equal.index.equals(weights_vol.index)
assert np.allclose(weights_equal.sum(axis=1), 0.0)
assert np.allclose(weights_vol.sum(axis=1), 0.0)
assert np.allclose(weights_equal.abs().sum(axis=1), 1.0)
assert np.allclose(weights_vol.abs().sum(axis=1), 1.0)
assert all(wf["rebalances"]["signal_close"] < wf["rebalances"].index)
print(f"Walk-forward sample: {live_dates.min().date()} to {live_dates.max().date()} ({len(live_dates):,} observations)")
print(f"Quarterly currency refits: {len(wf['refits']):,}; reused fits after optimizer failures: {len(wf['fit_failures']):,}")
display(wf["rebalances"].head())
display(wf["rebalances"].tail())
display(pd.DataFrame({"equal-weight gross daily P&L (%)": (weights_equal * carry_pnl.loc[live_dates]).sum(axis=1),
                      "inverse-vol gross daily P&L (%)": (weights_vol * carry_pnl.loc[live_dates]).sum(axis=1)}).head())"""),
        _md(r"""### 4.3 Transaction costs and funding spread

I charge trading cost on the **sum of absolute changes in currency weights** at each monthly rebalance, including entry into the first live portfolio. Thus a 5 bp one-way cost and 100% turnover reduce capital by 0.05%. I also charge an annual spread against *each* open long and short currency exposure. A 10 bp annual spread on 100% gross notional costs approximately 0.10% per year, accrued over the same actual calendar-day gaps as Section 1. These are explicit scenario assumptions, not observed broker quotes. The reference rates themselves are still imperfect proxies, especially for EUR, CHF, NZD, and SEK.

The base case is **5 bp one-way turnover cost and 10 bp annual funding spread**. The sensitivity grid also shows 0, 5, and 10 bp trading cost and 0, 10, and 25 bp funding spread. Costs do not change the signal or weights, so each scenario uses exactly the same walk-forward positions."""),
        _code(r"""BASE_TRADING_BPS = 5.0
BASE_FUNDING_BPS = 10.0


def backtest_weights(weights, trading_bps=0.0, funding_bps=0.0):
    # Return daily gross/net P&L in percent of initial notional.
    aligned = carry_pnl.loc[weights.index, weights.columns]
    gross = weights.mul(aligned).sum(axis=1)
    changes = weights.diff()
    changes.iloc[0] = weights.iloc[0]  # initial entry from cash
    turnover = changes.abs().sum(axis=1)
    transaction_cost = turnover * trading_bps / 100.0
    gaps = calendar_days.reindex(weights.index).astype(float)
    if gaps.isna().any():
        raise ValueError("Calendar-day gaps unavailable for a holding period")
    funding_cost = weights.abs().sum(axis=1) * (funding_bps / 100.0) * gaps / 365.0
    return pd.DataFrame({
        "gross": gross, "turnover": turnover,
        "transaction_cost": transaction_cost, "funding_cost": funding_cost,
        "net": gross - transaction_cost - funding_cost,
    })


def performance_metrics(backtest, return_col="net"):
    r = backtest[return_col].dropna()
    annual_return = float(r.mean() * 252)
    annual_vol = float(r.std(ddof=1) * np.sqrt(252))
    # Section 1 P&L is spot log change plus simple accrued-rate differential,
    # interpreted as a fixed-notional, additive P&L rather than reinvested wealth.
    wealth = 1.0 + r.cumsum() / 100.0
    wealth = pd.concat([pd.Series([1.0], index=[r.index[0] - pd.Timedelta(days=1)]), wealth])
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "annualized return (%)": annual_return,
        "annualized volatility (%)": annual_vol,
        "Sharpe": annual_return / annual_vol if annual_vol > 0 else np.nan,
        "skewness": float(stats.skew(r, bias=False)),
        "excess kurtosis": float(stats.kurtosis(r, fisher=True, bias=False)),
        "maximum drawdown (%)": float(100 * drawdown.min()),
        "annualized turnover (x capital)": float(backtest["turnover"].sum() * 252 / len(r)),
        "total costs + spread (% initial capital)": float(
            (backtest["transaction_cost"] + backtest["funding_cost"]).sum()
        ),
        "cumulative modeled P&L (%)": float(r.sum()),
    }


equal_gross = backtest_weights(weights_equal)
vol_gross = backtest_weights(weights_vol)
equal_net = backtest_weights(weights_equal, BASE_TRADING_BPS, BASE_FUNDING_BPS)
vol_net = backtest_weights(weights_vol, BASE_TRADING_BPS, BASE_FUNDING_BPS)

sensitivity_rows = []
for label, weights in (("Equal-weight", weights_equal), ("Inverse-vol", weights_vol)):
    for trading_bps in (0, 5, 10):
        for funding_bps in (0, 10, 25):
            test = backtest_weights(weights, trading_bps, funding_bps)
            met = performance_metrics(test)
            sensitivity_rows.append({
                "portfolio": label, "turnover cost (bp)": trading_bps,
                "annual spread (bp)": funding_bps,
                "annualized return (%)": met["annualized return (%)"],
                "Sharpe": met["Sharpe"],
                "cumulative P&L (%)": met["cumulative modeled P&L (%)"],
            })
sensitivity = pd.DataFrame(sensitivity_rows).set_index(
    ["portfolio", "turnover cost (bp)", "annual spread (bp)"]
)
display(sensitivity.round(3))

for label, gross, net in (("Equal-weight", equal_gross, equal_net), ("Inverse-vol", vol_gross, vol_net)):
    gross_total, net_total = gross["gross"].sum(), net["net"].sum()
    if gross_total > 0:
        survival = f"{100 * net_total / gross_total:.1f}% of gross cumulative modeled P&L survives"
    else:
        survival = "gross cumulative modeled P&L is non-positive, so a survival ratio is not meaningful"
    display(Markdown(
        f"**{label}:** Gross annualized mean {performance_metrics(gross)['annualized return (%)']:.2f}%; "
        f"base-case net annualized mean {performance_metrics(net)['annualized return (%)']:.2f}%. "
        f"Total base-case costs and spread are {net['transaction_cost'].sum() + net['funding_cost'].sum():.2f}% "
        f"of initial notional over the sample; {survival}."
    ))"""),
        _md(r"""### 4.4 Basket versus AUD/JPY

The single-pair comparison holds **+0.5 AUD and −0.5 JPY**, labeled **AUD/JPY (100% gross)**, against the same 100% gross currency notional as either basket. Its P&L is one-half of an unscaled +1 AUD/−1 JPY cross trade; the scaling does not affect Sharpe, though it does affect returns and drawdown. The USD funding legs in the two per-currency carry series cancel, leaving the AUD/JPY cross carry. A naive long-all-nine benchmark holds +1/9 of each foreign currency against USD. All four histories start on the first walk-forward holding date, share identical dates, and include the same base-case per-notional cost/spread assumptions. A static pair or naive benchmark pays only its initial entry trading cost; the monthly baskets pay when their weights change."""),
        _code(r"""weights_pair = pd.DataFrame(0.0, index=live_dates, columns=FOREIGN)
weights_pair.loc[:, "AUD"] = 0.5
weights_pair.loc[:, "JPY"] = -0.5
weights_naive = pd.DataFrame(1.0 / len(FOREIGN), index=live_dates, columns=FOREIGN)
pair_net = backtest_weights(weights_pair, BASE_TRADING_BPS, BASE_FUNDING_BPS)
naive_net = backtest_weights(weights_naive, BASE_TRADING_BPS, BASE_FUNDING_BPS)

comparison_runs = {
    "AUD/JPY (100% gross)": pair_net,
    "Equal-weight basket": equal_net,
    "Inverse-vol basket": vol_net,
}
comparison = pd.DataFrame({name: performance_metrics(run) for name, run in comparison_runs.items()}).T
display(comparison[["annualized return (%)", "annualized volatility (%)", "Sharpe", "maximum drawdown (%)"]].round(3))

best_basket = comparison.loc[["Equal-weight basket", "Inverse-vol basket"], "Sharpe"].idxmax()
sharpes = comparison["Sharpe"]
display(Markdown(
    f"The better basket by net Sharpe is **{best_basket}** at **{sharpes[best_basket]:.2f}**, "
    f"versus **{sharpes['AUD/JPY (100% gross)']:.2f}** for AUD/JPY, a difference of "
    f"**{sharpes[best_basket] - sharpes['AUD/JPY (100% gross)']:+.2f} Sharpe points**. "
    + ("In this sample, cross-currency diversification improved risk-adjusted performance. "
       if sharpes[best_basket] > sharpes["AUD/JPY (100% gross)"] else
       "In this sample, diversification did not improve risk-adjusted performance. ")
    + f"The pair's maximum drawdown was {comparison.loc['AUD/JPY (100% gross)', 'maximum drawdown (%)']:.2f}%, "
      f"versus {comparison.loc[best_basket, 'maximum drawdown (%)']:.2f}% for {best_basket.lower()}. "
      "These are historical sample comparisons, not estimates of future Sharpe."
))

# Several holdings may share the same adverse FX shock. The diagnostic uses
# the same common dates as the three-way performance comparison.
cross_corr = carry_pnl.loc[live_dates, FOREIGN].corr().to_numpy()
mean_pairwise_corr = float(cross_corr[np.triu_indices(len(FOREIGN), 1)].mean())
leg_contributions = weights_vol * carry_pnl.loc[live_dates, FOREIGN]
bad_days = vol_net["net"] <= vol_net["net"].quantile(0.05)
losing_legs_on_bad_days = float((leg_contributions.loc[bad_days] < 0).sum(axis=1).mean())
display(Markdown(
    f"**Common risk:** The mean pairwise correlation of the nine standalone "
    f"USD-funded carry returns is **{mean_pairwise_corr:.2f}**. On the worst "
    f"5% of inverse-vol basket days, an average of **{losing_legs_on_bad_days:.1f} "
    "of six active legs** lost money together. Holding several currencies "
    "therefore does not remove a shared FX crash exposure."
))"""),
        _md(r"""## Section 5 — Full Walk-Forward Backtest and Risk Tearsheet

### 5.1 Recorded live loop

The Section 4 engine is the Section 5 walk-forward loop: it records daily net and gross returns, monthly weight transitions, turnover, transaction charges, funding charges, refit dates, and an ex-ante one-day loss threshold. A quarterly GARCH-t refit sees only the preceding three years (or less at the start); daily variance recursion uses newly realized returns after they occur. The most recent available rate differential is used at the last close of each month, and the next day's return is earned only by the new weights. The **95% conditional VaR** uses each selected leg's GARCH-t one-step forecast, a trailing 252-day correlation matrix, and the heaviest-tailed fitted leg's standardized Student-t 5th percentile. This is an approximate portfolio VaR, since separate univariate GARCH fits do not model joint FX crash dependence directly."""),
        _code(r"""recorded = pd.DataFrame({
    "gross_pct": vol_net["gross"],
    "net_pct": vol_net["net"],
    "turnover": vol_net["turnover"],
    "transaction_cost_pct": vol_net["transaction_cost"],
    "funding_cost_pct": vol_net["funding_cost"],
    "ex_ante_95pct_VaR_pct": wf["var95"],
    "ex_ante_vol_pct": wf["sigma_forecast"],
})
assert recorded.notna().all().all()
assert (recorded["turnover"] > 0).sum() <= len(wf["rebalances"])
display(recorded.head())
display(weights_vol.head())
display(pd.DataFrame({
    "number of monthly rebalances": [len(wf["rebalances"])],
    "number of currency GARCH-t refits": [len(wf["refits"])],
    "total turnover (x capital)": [recorded["turnover"].sum()],
    "total trading charges (%)": [recorded["transaction_cost_pct"].sum()],
    "total funding charges (%)": [recorded["funding_cost_pct"].sum()],
}).round(3))"""),
        _md(r"""### 5.2 One-page risk tearsheet

The table reports the **net inverse-vol basket** in the base case. Annualized return is 252 times mean daily P&L; annualized volatility is daily standard deviation times √252; the Sharpe uses zero excess return because the carry P&L is already measured relative to USD funding. Max drawdown is calculated from the **additive fixed-notional wealth path** $1+\sum_t\mathrm{P\&L}_t/100$, consistent with Section 1's log-spot-plus-simple-accrual approximation and the absence of reinvestment. The VaR row is the latest **one-day gross market-loss** threshold, a positive loss magnitude in percent of initial notional, before deterministic trading and funding charges. Turnover is the sum of absolute monthly weight changes annualized by 252 / number of live observations. The total cost row sums daily charges in percent of initial notional, without reinvesting or capital changes."""),
        _code(r"""tearsheet = performance_metrics(vol_net)
tearsheet["95% conditional VaR, latest 1-day (%)"] = float(wf["var95"].iloc[-1])
metric_order = [
    "annualized return (%)", "annualized volatility (%)", "Sharpe",
    "skewness", "excess kurtosis", "maximum drawdown (%)",
    "95% conditional VaR, latest 1-day (%)", "annualized turnover (x capital)",
    "total costs + spread (% initial capital)",
]
display(pd.DataFrame({"Metric": metric_order,
                      "Value": [tearsheet[k] for k in metric_order]}).set_index("Metric").round(3))

returns_for_chart = pd.DataFrame({
    "Inverse-vol G10 (net)": vol_net["net"],
    "Equal-weight G10 (net)": equal_net["net"],
    "AUD/JPY (net)": pair_net["net"],
    "Long all nine (net)": naive_net["net"],
})
wealth_paths = 1.0 + returns_for_chart.cumsum() / 100.0
wealth_anchor = pd.DataFrame(1.0, index=[returns_for_chart.index[0] - pd.Timedelta(days=1)],
                             columns=returns_for_chart.columns)
wealth_paths = pd.concat([wealth_anchor, wealth_paths])
cumulative_pnl = 100 * (wealth_paths - 1.0)
drawdown_paths = 100 * (wealth_paths / wealth_paths.cummax() - 1.0)

fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
cumulative_pnl.plot(ax=axes[0, 0], lw=1.3)
axes[0, 0].set(title="Cumulative strategy P&L", ylabel="Return on initial capital (%)", xlabel="Date")
axes[0, 0].legend(fontsize=8)
drawdown_paths[["Inverse-vol G10 (net)", "Equal-weight G10 (net)", "AUD/JPY (net)"]].plot(ax=axes[0, 1], lw=1)
axes[0, 1].set(title="Running drawdown", ylabel="Drawdown (%)", xlabel="Date")
axes[0, 1].legend(fontsize=8)
axes[1, 0].hist(vol_net["net"], bins=60, color="steelblue", alpha=0.85)
axes[1, 0].axvline(vol_net["net"].quantile(0.05), color="darkred", ls="--", label="Empirical 5th percentile")
axes[1, 0].set(title="Net daily P&L distribution", xlabel="Daily P&L (%)", ylabel="Observations")
axes[1, 0].legend(fontsize=8)
axes[1, 1].plot((-vol_net["gross"]).index, -vol_net["gross"], alpha=0.35, lw=0.6, label="Realized gross market loss")
axes[1, 1].plot(wf["var95"].index, wf["var95"], lw=1, color="darkred", label="Ex-ante 95% GARCH-t VaR")
axes[1, 1].set(title="One-day loss versus conditional VaR", ylabel="Loss / VaR (%)", xlabel="Date")
axes[1, 1].legend(fontsize=8)
plt.show()

breaches = ((-vol_net["gross"]) > wf["var95"]).sum()
display(Markdown(
    f"**VaR check:** {breaches:,} of {len(vol_net):,} live days "
    f"({100 * breaches / len(vol_net):.1f}%) had gross market losses above the prior-close 95% VaR estimate. "
    f"The latest estimate is **{wf['var95'].iloc[-1]:.3f}%** for one day. "
    "A 95% VaR describes a loss threshold, not the worst possible loss."
))"""),
        _md(r"""#### Carry-unwind episode: August 2024

The August 2024 JPY unwind is inside this sample if the downloaded source histories cover that month. The top panel below shows the **net** portfolio drawdown around that episode. The lower panel overlays **gross market losses** and the *previous-close* VaR that applied to each holding day, keeping deterministic charges out of both sides of the VaR comparison. The accompanying text checks whether the worst early-August gross loss exceeded its advance risk estimate and whether that estimate increased on the following trading day."""),
        _code(r"""episode_start, episode_end = pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30")
episode = vol_net.loc[episode_start:episode_end]
if len(episode) >= 10:
    episode_drawdown = drawdown_paths.loc[episode.index, "Inverse-vol G10 (net)"]
    local_var = wf["var95"].loc[episode.index]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, constrained_layout=True)
    axes[0].plot(episode_drawdown.index, episode_drawdown, color="navy", label="Net inverse-vol basket")
    axes[0].set(title="August 2024 carry unwind: basket drawdown", ylabel="Drawdown from running peak (%)")
    axes[0].legend()
    axes[1].bar(episode.index, -episode["gross"], width=1.5, color="gray", alpha=0.55, label="Realized gross market loss")
    axes[1].plot(local_var.index, local_var, color="darkred", lw=1.5, label="Prior-close 95% VaR")
    axes[1].set(title="Advance loss estimate versus realized loss", ylabel="Loss / VaR (%)", xlabel="Date")
    axes[1].legend()
    plt.show()

    early_august = episode.loc["2024-08-01":"2024-08-12", "gross"]
    worst_date = early_august.idxmin()
    worst_loss = -float(early_august.min())
    predicted = float(local_var.loc[worst_date])
    later_dates = local_var.index[local_var.index > worst_date]
    next_var = float(local_var.loc[later_dates[0]]) if len(later_dates) else np.nan
    july_var = float(local_var.loc["2024-07-15":"2024-07-31"].median())
    breach_text = "exceeded" if worst_loss > predicted else "stayed below"
    reaction_text = ("rose after the loss" if next_var > predicted else "did not rise on the next holding day")
    display(Markdown(
        f"The worst gross market day from August 1–12 was **{worst_date.date()}**, a **{worst_loss:.2f}%** "
        f"loss. Its advance 95% VaR was **{predicted:.2f}%**, so the loss {breach_text} the "
        f"model's threshold. July 15–31 median VaR was **{july_var:.2f}%**; the next "
        f"holding day's VaR was **{next_var:.2f}%** and {reaction_text}. "
        f"The running drawdown reached **{episode_drawdown.min():.2f}%** in this window. A higher post-shock "
        "forecast is a response to observed damage, not advance protection against it."
    ))
else:
    display(Markdown("The source histories do not include enough July–September 2024 observations to assess the August 2024 carry unwind."))"""),
        _md(r"""### 5.3 Interpretation and PM decision

The following paragraphs are generated from the notebook's computed outputs, so the interpretation updates when the source data cache is refreshed. The recommendation is conditional on this historical sample and the stated cost assumptions."""),
        _code(r"""eqm = performance_metrics(equal_net)
vm = performance_metrics(vol_net)
eg = performance_metrics(equal_gross)
vg = performance_metrics(vol_gross)
dd_eq = abs(eqm["maximum drawdown (%)"])
dd_vol = abs(vm["maximum drawdown (%)"])
dd_reduction = dd_eq - dd_vol
dd_comparison = (f"an improvement of {dd_reduction:.2f} percentage points"
                 if dd_reduction >= 0 else
                 f"a deterioration of {-dd_reduction:.2f} percentage points")
tail_change = vm["skewness"] - eqm["skewness"]
meaningful = dd_reduction > 0.15 * dd_eq and tail_change >= -0.1

display(Markdown(
    f"**Sizing and tail risk.** The equal-weight basket lost as much as **{dd_eq:.2f}%** "
    f"from peak to trough; inverse-vol sizing lost **{dd_vol:.2f}%**, "
    f"**{dd_comparison}** in maximum drawdown. "
    f"Their net skewness values were **{eqm['skewness']:.2f}** and **{vm['skewness']:.2f}**, "
    f"respectively; excess kurtosis was **{eqm['excess kurtosis']:.2f}** and "
    f"**{vm['excess kurtosis']:.2f}**. "
    + ("On those two measures the risk reduction was material, though this is still only a historical sample."
       if meaningful else "On these measures inverse-vol sizing did not deliver a clear, material tail-risk reduction.")
))

gross_vs_net = vg["annualized return (%)"] - vm["annualized return (%)"]
eq_gross_vs_net = eg["annualized return (%)"] - eqm["annualized return (%)"]
harsh_return = float(sensitivity.loc[("Inverse-vol", 10, 25), "annualized return (%)"])
if vg["annualized return (%)"] > 0 >= vm["annualized return (%)"]:
    friction_verdict = "The base-case costs reverse a positive gross mean into a negative net one."
elif vm["annualized return (%)"] > 0:
    friction_verdict = "The base-case costs narrow, but do not reverse, the positive gross mean."
else:
    friction_verdict = "The gross mean was already non-positive, and costs deepen that result."
display(Markdown(
    f"**Friction sensitivity.** At the base-case 5 bp trading charge and 10 bp annual spread, "
    f"inverse-vol annualized mean fell from **{vg['annualized return (%)']:.2f}%** gross "
    f"to **{vm['annualized return (%)']:.2f}%** net, a **{gross_vs_net:.2f}-point** drag. "
    f"Equal-weight fell by **{eq_gross_vs_net:.2f} points**. The sensitivity table above "
    f"puts the inverse-vol mean at **{harsh_return:.2f}%** with 10 bp turnover cost "
    f"and a 25 bp annual spread. {friction_verdict}"
))

worst_day = float(vol_net["net"].min())
premium_risk = vm["skewness"] < -0.5 or vm["excess kurtosis"] > 3 or breaches > 0.07 * len(vol_net)
description = ("a small premium exposed to occasional large losses" if premium_risk else
               "a modest historical edge with no especially strong negative-tail signal in this sample")
display(Markdown(
    f"**Shape of returns.** Net skewness is **{vm['skewness']:.2f}**, excess kurtosis "
    f"**{vm['excess kurtosis']:.2f}**, the worst day **{worst_day:.2f}%**, and maximum "
    f"drawdown **{vm['maximum drawdown (%)']:.2f}%**. With **{breaches}** advance 95% VaR "
    f"breaches, I describe this sample as **{description}**. The August 2024 panel gives "
    "the time ordering of one known unwind, which a single Sharpe number cannot show."
))

if vm["Sharpe"] > 0.5 and vm["annualized return (%)"] > 0 and vm["maximum drawdown (%)"] > -20:
    recommendation = "I would consider a small, monitored pilot allocation"
else:
    recommendation = "I would not allocate a full-risk live book on this evidence"
display(Markdown(
    f"**PM recommendation.** {recommendation}. The net annualized return is "
    f"**{vm['annualized return (%)']:.2f}%** with Sharpe **{vm['Sharpe']:.2f}** and "
    f"maximum drawdown **{vm['maximum drawdown (%)']:.2f}%**. A single change I would "
    "test first is a fixed-budget JPY call hedge against a funding-currency unwind, "
    "because the August 2024 loss exceeded the advance VaR estimate. I would price "
    "the option premium and require an out-of-sample reduction in tail losses before "
    "increasing capital."
))"""),
        _md(r"""## Section 6 — Bonus: Three-Month Trend Overlay

At each *monthly* signal close I compare each selected currency's USD spot to its trailing 63-common-observation mean, calculated through that close. I retain a carry long only above its mean and a carry short only below its mean. Disallowed legs are set to zero; I **do not reallocate** their weight to surviving legs, so gross exposure may fall below 100% and the idle allocation earns USD cash, worth zero excess P&L in this framework. The rule is fixed before looking at the results, uses no future prices, and pays transaction/funding charges on the resulting positions exactly as above."""),
        _code(r"""trend_weights = pd.DataFrame(0.0, index=live_dates, columns=FOREIGN)
rebalance_dates = list(wf["rebalances"].index)
all_carry_dates = carry_pnl.index
for k, trade_date in enumerate(rebalance_dates):
    start_position = live_dates.get_loc(trade_date)
    end_position = live_dates.get_loc(rebalance_dates[k + 1]) if k + 1 < len(rebalance_dates) else len(live_dates)
    previous_close = all_carry_dates[all_carry_dates.get_loc(trade_date) - 1]
    recent_spot = spot_usd.loc[:previous_close, FOREIGN].tail(63)
    if len(recent_spot) < 63:
        raise ValueError("Trend overlay requires 63 known spot observations")
    base = weights_vol.loc[trade_date]
    current, average = recent_spot.iloc[-1], recent_spot.mean()
    allowed = ((base > 0) & (current > average)) | ((base < 0) & (current < average))
    trend_weights.iloc[start_position:end_position, :] = base.where(allowed, 0.0).to_numpy()

trend_net = backtest_weights(trend_weights, BASE_TRADING_BPS, BASE_FUNDING_BPS)
trend_metrics = performance_metrics(trend_net)
overlay_comparison = pd.DataFrame({
    "Inverse-vol carry": vm,
    "With 63-day trend": trend_metrics,
}).T
display(overlay_comparison[["annualized return (%)", "Sharpe", "skewness", "excess kurtosis",
                            "maximum drawdown (%)", "annualized turnover (x capital)"]].round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
pd.DataFrame({
    "Carry only": vol_net["net"].cumsum(),
    "With trend": trend_net["net"].cumsum(),
}).plot(ax=axes[0], lw=1.2)
axes[0].set(title="Trend overlay: cumulative net P&L", xlabel="Date", ylabel="Return on initial capital (%)")
pd.DataFrame({
    "Carry only": vol_net["net"].loc[episode_start:episode_end].cumsum(),
    "With trend": trend_net["net"].loc[episode_start:episode_end].cumsum(),
}).plot(ax=axes[1], lw=1.3)
axes[1].set(title="July–September 2024 cumulative net P&L", xlabel="Date", ylabel="Local P&L (%)")
plt.show()

episode_carry = vol_net["net"].loc[episode_start:episode_end].sum()
episode_trend = trend_net["net"].loc[episode_start:episode_end].sum()
dd_gain = abs(vm["maximum drawdown (%)"]) - abs(trend_metrics["maximum drawdown (%)"])
trend_dd_description = (f"reduced drawdown by {dd_gain:.2f} percentage points"
                        if dd_gain >= 0 else
                        f"increased drawdown by {-dd_gain:.2f} percentage points")
display(Markdown(
    f"The overlay changed annualized return from **{vm['annualized return (%)']:.2f}%** to "
    f"**{trend_metrics['annualized return (%)']:.2f}%**, skewness from "
    f"**{vm['skewness']:.2f}** to **{trend_metrics['skewness']:.2f}**, and maximum "
    f"drawdown from **{vm['maximum drawdown (%)']:.2f}%** to "
    f"**{trend_metrics['maximum drawdown (%)']:.2f}%** "
    f"({trend_dd_description}). "
    f"July–September 2024 modeled P&L was **{episode_carry:.2f}%** without and "
    f"**{episode_trend:.2f}%** with the overlay. "
    + ("The added trend rule improved historical drawdown after costs."
       if dd_gain > 0 else "The added trend rule did not improve historical drawdown after costs.")
))"""),
    ]
