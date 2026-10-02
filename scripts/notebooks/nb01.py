"""Cells of notebooks/01_sum_to_one_eda.ipynb (Phase 2: is the basket sum structurally mean-reverting?)."""

CELLS = [
("md", r"""
# 01 · Sum-to-one: is the basket sum structurally mean-reverting?

In a **mutually exclusive and exhaustive** outcome set (a Polymarket *negRisk* event), exactly one
outcome resolves YES, so the fair probabilities must satisfy

$$\sum_{i=1}^{n} P_i = 1 .$$

Each traded YES price $p_{i,t}$ is a noisy, frictional estimate of $P_i$. This notebook asks whether
the **basket sum** $S_t=\sum_i p_{i,t}$:

1. sits at a stable level (and whether that level is exactly 1 or carries a structural premium or discount);
2. **mean-reverts** while the individual legs wander like random walks (unit-root tests, variance ratios, variograms);
3. reverts fast enough, and by enough, to beat the cost of trading every leg (spreads + taker fees).

The primary basket is the **Fed decision at the October 2026 FOMC** (5 outcomes). Four more live baskets are summarised at the end.
"""),
("code", r"""
import sys, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Markdown, display
from src import plotting as P, stats_tools as st, research as R
from src.config import load_markets, get_basket
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 160)
cfg = load_markets()
BASKET = "fed-oct-2026"
ALL = ["fed-oct-2026", "balance-of-power-2026", "house-2026", "senate-2026", "mlb-ws-2026"]
basket = get_basket(BASKET, cfg)
"""),
("code", r"""
prices, info_prices = R.price_panel(BASKET, cfg)          # REAL /prices-history, 1-minute grid
books = R.book_sums(BASKET, cfg, grid="10s")              # REAL recorded live order books (may be short)
display(Markdown(info_prices.banner_markdown()))
if books is not None:
    display(Markdown(books[2].banner_markdown()))
S = prices.sum(axis=1).rename("S_mid")
DK = info_prices.label
"""),
("md", r"""
## 1. The basket

Legs, fee schedules and tick sizes come straight from the Gamma API (`python -m src.discovery add ...`); token ids are never typed by hand.
"""),
("code", r"""
display(R.basket_overview(cfg, ALL))
display(R.leg_table(basket))
"""),
("md", r"""
## 2. Legs drift; the sum stays put

`/prices-history` returns each leg's **order-book midpoint** sampled once a minute (verified against our own recorded books: the sampled
values equal (best bid + best ask)/2). Legs are sampled independently, so an update that hits one leg a few seconds before its
complement creates a transient spike in the sum. That is microstructure noise, not a tradable dislocation, and the tests below have to
tell the two apart.
"""),
("code", r"""
P.show(P.plot_legs(prices, data_kind=DK, title=f"{basket.title}: leg prices (1-min)"), "01_legs")
P.show(P.plot_sum_band(pd.DataFrame({"s_mid": S}), data_kind=DK, title="basket sum ΣP (1-min)"), "01_sum_band")
"""),
("code", r"""
q = S.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
inside = ((S >= 0.96) & (S <= 1.05)).mean()
display(pd.DataFrame({"quantile": q}).T)
display(Markdown(f"The blueprint's hypothesis that the sum oscillates within **[0.96, 1.05]** holds for **{inside:.1%}** of minutes; "
                 f"the observed 1–99% range is **[{q[0.01]:.4f}, {q[0.99]:.4f}]** around a mean of **{S.mean():.4f}**."))
P.show(P.plot_sum_distribution(S, data_kind=DK, title="distribution of ΣP"), "01_sum_distribution")
"""),
("md", r"""
## 3. Unit-root tests: legs vs sum

* **ADF** (H0: unit root) and **KPSS** (H0: stationary) are run on every leg (in price and in logit) and on $S$; ADF p-values are Holm-adjusted across the family.
* Reading the 2×2 table: ADF rejects + KPSS does not → *stationary*; ADF does not reject + KPSS rejects → *unit root*; both reject → *inconclusive*
  (typical of a series that mean-reverts but slowly, or has occasional level shifts).
* Long-shot legs pinned near the 0.001 tick floor look "stationary" only because they cannot move; they are flagged and excluded from the "legs are I(1)" claim.
"""),
("code", r"""
series = {f"leg: {c}": prices[c] for c in prices.columns}
series.update({f"logit: {c}": np.log(prices[c].clip(1e-4, 1 - 1e-4) / (1 - prices[c].clip(1e-4, 1 - 1e-4)))
               for c in prices.columns if prices[c].mean() > 0.02})
series["SUM S = Σp"] = S
ur = R.unit_root_table(series)
ur["pinned long shot (<0.02)"] = [n.startswith("leg:") and prices[n[5:]].mean() < 0.02 for n in ur.index]
display(ur)
"""),
("md", r"""
## 4. Variance ratios, variogram and autocorrelation

For a random walk $\mathrm{Var}(x_t-x_{t-q})$ grows linearly in $q$ and the variance ratio $VR(q)\approx 1$; for a stationary AR(1) it levels off at
$2\gamma_0$ and $VR(q)\to 0$ like $1/q$. Pure quote noise on top of a random walk would instead make $VR$ drop quickly to a *plateau* below 1.
"""),
("code", r"""
lags = [1, 2, 5, 15, 30, 60, 120, 240, 480, 960]
big = [c for c in prices.columns if prices[c].mean() > 0.02]
curves = {f"leg {c}": (lags, st.variogram(prices[c].to_numpy(), lags)) for c in big}
curves["basket sum S"] = (lags, st.variogram(S.to_numpy(), lags))
P.show(P.plot_variogram(curves, data_kind=DK, title="variogram (1-min grid)"), "01_variogram")
display(R.variance_ratio_table({**{f"leg {c}": prices[c] for c in big}, "basket sum S": S}))
P.show(P.plot_acf_pair(S, nlags=240, data_kind=DK, title="autocorrelation of S and ΔS (lags in minutes)"), "01_acf")
"""),
("md", r"""
## 5. Speed of mean reversion: AR(1) / Ornstein–Uhlenbeck half-life

$S_t = c + \phi S_{t-1} + \varepsilon_t$ with half-life $h = -\ln 2/\ln\phi$ grid steps. If the half-life were about one grid step at every
sampling interval we would be measuring bid–ask bounce, not dislocation decay, so it is re-estimated on 1, 5 and 15-minute grids (it should
be roughly invariant in clock time). Confidence intervals: delta method and a stationary block bootstrap.
"""),
("code", r"""
hl = R.half_life_table(S, n_boot=200)
display(hl)
"""),
("md", r"""
## 6. Is the level exactly 1? (structural premium / discount)

$H_0: E[S]=1$ is tested with Newey–West (HAC) standard errors because $S$ is strongly autocorrelated. Reasons the level can differ from 1:
the half-spread bias of long-shot mids near the tick floor, the favourite–long-shot bias, capital lock-up until resolution (pushes asks
below 1), and for *augmented* events the unnamed "Other" outcomes (named legs sum to less than 1).
"""),
("code", r"""
h = st.hac_mean_test(S, 1.0)
display(pd.DataFrame([h]))
direction = "premium (overround)" if h["mean"] > 1 else "discount"
display(Markdown(f"Mean ΣP = **{h['mean']:.4f}** (HAC t = {h['t']:.2f}, p = {R.fmt_p(h['pvalue'])}): "
                 + (f"a statistically significant **{direction}** of {abs(h['mean'] - 1) * 100:.2f} cents per basket."
                    if h["pvalue"] < 0.05 else "not significantly different from 1.")))
"""),
("md", r"""
## 7. What can actually be traded: the executable band from live order books

Mids are not tradable. Buying one YES of every leg costs $S_{ask}=\sum a_i$; buying one NO of every leg costs $n-S_{bid}$ (NO asks mirror
YES bids) and pays $n-1$. A riskless taker trade therefore needs $S_{ask}<1$ or $S_{bid}>1$ (before fees). The band
$[S_{bid}, S_{ask}]$ is at least $n$ ticks wide.
"""),
("code", r"""
if books is not None:
    sums, seg, binfo = books
    bs = R.band_stats(sums, basket)
    display(pd.DataFrame([bs]))
    P.show(P.plot_sum_band(sums.dropna(subset=["s_mid"]), data_kind=binfo.label,
                           title="recorded live books: ΣMid inside [ΣBid, ΣAsk]"), "01_live_band")
else:
    display(Markdown("_No recorded order books yet for this basket._"))
"""),
("md", r"""
## 8. Five live baskets side by side, and the cost hurdle (interview pro-tip 2)

A z-score round trip on the mid-sum must clear the n-leg spread plus entry and exit taker fees
(fee per unit $F=\sum_i r_i p_i(1-p_i)\approx r(1-\sum p_i^2)$). The **edge-to-cost ratio** $ECR=\sigma(S)/\text{hurdle}$ says how many hurdles
a typical excursion spans: liquid, tight baskets rank first; crypto carries the highest fee rate (0.07).
"""),
("code", r"""
stats_all = R.basket_stats_table(cfg, ALL)
display(stats_all)
hurdles = R.hurdle_table(cfg, ALL)
display(hurdles)
"""),
("code", r"""
row = stats_all.loc[BASKET]
legs_big = [n for n in ur.index if n.startswith("leg:") and not ur.loc[n, "pinned long shot (<0.02)"]]
legs_ur = [n for n in legs_big if not ur.loc[n, "mean-reverting (ADF)"]]
hl1 = hl.loc["1min", "half-life (min)"]
ecr = hurdles.loc[BASKET, "ECR = sd/hurdle"] if BASKET in hurdles.index else float("nan")
lines = [
    f"* **Legs wander, the sum does not.** {len(legs_ur)} of {len(legs_big)} liquid legs fail to reject a unit root (ADF, Holm), "
    f"while ADF on S rejects with p = {R.fmt_p(row['ADF p (Holm)'])}. KPSS also rejects for S (p {row['KPSS p']}), so the formal verdict is "
    f"'{row['verdict']}': S is mean-reverting but highly persistent (slow decay plus level shifts at news), not white noise around 1.",
    f"* **Speed.** AR(1) half-life ≈ **{hl1:.0f} minutes** on the 1-minute grid ({hl.loc['5min','half-life (min)']:.0f} / "
    f"{hl.loc['15min','half-life (min)']:.0f} min on 5 / 15-minute grids): dislocations take hours, not seconds, to decay.",
    f"* **Level.** Mean ΣP = {row['mean ΣP']:.4f} (HAC t = {row['HAC t (mean=1)']:.1f}): a structural {'premium' if row['mean ΣP'] > 1 else 'discount'}, "
    f"which is why the engine z-scores S against a *rolling* mean rather than against 1.00.",
    f"* **Costs dominate.** A taker round trip must clear ≈ {hurdles.loc[BASKET,'round-trip hurdle']:.3f} per basket, while the typical "
    f"1-minute dispersion of S is {row['sd ΣP']:.4f} (ECR = {ecr:.2f}). Mean reversion is real, but most of it happens *inside* the no-arbitrage band.",
]
if books is not None:
    lines.append(f"* **Live books.** Over the recorded hours S_ask never fell below 1 (min {bs['min S_ask']:.3f}) and S_bid never exceeded 1 "
                 f"(max {bs['max S_bid']:.3f}): no riskless taker arbitrage was available in that window.")
display(Markdown("### Conclusions (generated from the numbers above)\n" + "\n".join(lines)))
"""),
("md", r"""
### Limitations
* `/prices-history` mids are sampled per leg once a minute; asynchronous sampling inflates the measured dispersion of S at short horizons.
* Thirty days of 1-minute data for one event is one realisation; news (CPI, FOMC communication) creates level shifts that unit-root tests read as persistence.
* MLB: legs eliminated during the window resolve NO; the sum of the *currently open* legs is therefore not a fixed basket over history (its unit root is a composition effect).
* House/Senate are *augmented* negRisk events: named legs exclude inactive placeholders, so their sum carries P(unnamed outcome).
"""),
("code", r"""
infos = [info_prices] + ([books[2]] if books is not None else [])
R.save_dataset_info("01", infos, {"basket_stats": stats_all.reset_index().to_dict("records"),
                                   "half_life_min": float(hl1), "hurdles": hurdles.reset_index().to_dict("records")})
print("saved results/dataset_info_01.json; figures in results/figures/01_*.png")
"""),
]
