"""Cells of notebooks/01_sum_to_one_eda.ipynb: do the prices of all outcomes add up to $1?"""

CELLS = [
("md", r"""
# 01 · Do the prices add up to $1?

**The idea in one sentence.** On Polymarket, an event like *"What will the Fed decide in October?"* has several outcomes
(cut 50, cut 25, no change, hike 25, hike 50). Exactly one of them will happen, and each "YES" share pays **$1** if its outcome happens.
So if you bought one YES share of *every* outcome you would get exactly $1 back, no matter what. That means the prices of all the
YES shares **should add up to $1.00**.

In practice they don't add up exactly, because traders are human, markets are busy and trading has costs. This notebook looks at real
Polymarket data and asks three simple questions:

1. How far from $1.00 does the total (we call it **S**) actually move?
2. When it moves away, does it come back? (*mean reversion*)
3. Is the gap ever big enough to profit from after trading costs?
"""),
("code", r"""
import sys, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import logging; logging.getLogger("src").setLevel(logging.ERROR)  # the live recorder may be mid-write
import numpy as np, pandas as pd
from IPython.display import Markdown, display
from src import plotting as P, stats_tools as st, research as R
from src.config import load_markets, get_basket
pd.set_option("display.float_format", "{:.4f}".format)
cfg = load_markets()
BASKET = "fed-oct-2026"
ALL = ["fed-oct-2026", "balance-of-power-2026", "house-2026", "senate-2026", "mlb-ws-2026"]
basket = get_basket(BASKET, cfg)
"""),
("md", r"""
## 1. The data

We use **real Polymarket prices**: the price of every outcome, once a minute, for the last 30 days (downloaded with
`python -m src.recorder backfill`). The price used is the *midpoint*, halfway between the best offer to buy and the best offer to sell.
"""),
("code", r"""
prices, info = R.price_panel(BASKET, cfg)
display(Markdown(info.banner_markdown()))
S = prices.sum(axis=1).rename("S")
DK = info.label
display(pd.DataFrame({"outcome": [l.label for l in basket.legs],
                      "average price": prices.mean().to_numpy(),
                      "taker fee rate": [l.fee.rate for l in basket.legs]}).set_index("outcome"))
"""),
("md", r"""
## 2. The single prices move a lot; their total barely moves

The first chart shows each outcome's price. "No change" and "hike 25" swing around as news comes in. The second chart adds them all up:
the total stays in a narrow band close to $1.00. When one outcome gets more likely, the others get cheaper by about the same amount.
"""),
("code", r"""
P.show(P.plot_legs(prices, data_kind=DK, title=f"{basket.title}: price of each outcome"), "01_legs")
P.show(P.plot_sum_band(pd.DataFrame({"s_mid": S}), data_kind=DK, title="sum of all YES prices (S)"), "01_sum_band")
"""),
("code", r"""
q01, q99 = S.quantile(0.01), S.quantile(0.99)
display(Markdown(f"* Average total: **${S.mean():.4f}**, so on average the outcomes are priced about **{(S.mean() - 1) * 100:+.2f} cents** above the fair $1.00.\n"
                 f"* 98% of the time the total stays between **${q01:.3f}** and **${q99:.3f}**.\n"
                 f"* The blueprint guessed it would stay between $0.96 and $1.05: true **{((S >= 0.96) & (S <= 1.05)).mean():.0%}** of the time."))
P.show(P.plot_sum_distribution(S, data_kind=DK, title="how often each value of S occurs"), "01_sum_distribution")
"""),
("md", r"""
## 3. Does the total come back after it moves away? (mean reversion)

Two kinds of series are worth telling apart:

* A **random walk** has no "home". Where it goes next does not depend on where it has been, so it can drift anywhere. A single outcome's
  price looks like this: it follows the news.
* A **mean-reverting** series keeps getting pulled back towards an average. If the total really should be about $1, it should look like this.

The standard check is the **ADF test** (Augmented Dickey–Fuller). It starts from the assumption "this is a random walk" and reports a
**p-value**. A small p-value (below 0.05) means *"it is very unlikely to be a random walk"*, so the series is pulled back towards its average.
"""),
("code", r"""
rows = []
for name, x in [*[(f"outcome: {l.label}", prices[l.leg_id]) for l in basket.legs if prices[l.leg_id].mean() > 0.02], ("TOTAL S", S)]:
    p = st.adf_test(x)["pvalue"]
    rows.append({"series": name, "ADF p-value": p, "verdict": "pulled back to its average" if p < 0.05 else "random walk (drifts)"})
adf = pd.DataFrame(rows).set_index("series")
display(adf)
"""),
("md", r"""
**How fast does it come back?** We measure the *half-life*: after the total jumps away from its average, how long until half of that
gap has closed. (It comes from fitting a simple "today = a × yesterday + noise" model.)
"""),
("code", r"""
fit = st.ar1_fit(S, 60.0)
half_life_min = fit["half_life_s"] / 60
display(Markdown(f"Half-life ≈ **{half_life_min:.0f} minutes**: a gap in the total typically halves in about {half_life_min / 60:.1f} hours."))
lags = [1, 5, 15, 60, 240, 960]
big = [l.leg_id for l in basket.legs if prices[l.leg_id].mean() > 0.02]
curves = {f"outcome {c}": (lags, st.variogram(prices[c].to_numpy(), lags)) for c in big}
curves["TOTAL S"] = (lags, st.variogram(S.to_numpy(), lags))
P.show(P.plot_variogram(curves, data_kind=DK, title="how far each series moves over longer and longer time gaps"), "01_variogram")
display(Markdown("In the chart above, the single outcomes keep moving further the longer you wait (a straight line going up). "
                 "The total flattens out: it does not wander off, it keeps coming back."))
"""),
("md", r"""
## 4. Can you trade it? The cost of a round trip

You can't buy at the midpoint. You buy at the **ask** (a bit higher) and sell at the **bid** (a bit lower), and Polymarket charges a
**taker fee** of about `rate × price × (1 − price)` per share. To trade the total you have to buy **every** outcome and later sell every
outcome, so you pay the gap between bid and ask on all of them, twice, plus fees.

The table below compares the typical size of the total's moves with that round-trip cost for five live events. If the cost is bigger than
the typical move, trading the reversion loses money.
"""),
("code", r"""
h = R.hurdle_table(cfg, ALL)
simple = pd.DataFrame({"typical move of S (std)": h["sd ΣP (1-min)"], "round-trip cost": h["round-trip hurdle"],
                       "move ÷ cost": h["ECR = sd/hurdle"]})
display(simple)
books = R.book_sums(BASKET, cfg, grid="10s")
if books is not None:
    bs = R.band_stats(books[0], basket)
    display(Markdown(books[2].banner_markdown()))
    display(Markdown(R.band_sentence(books[2].hours, bs["min S_ask"], bs["max S_bid"], bs["share S_ask<1"], bs["share S_bid>1"],
                                     share_beyond_fees=bs["share S_ask<1-fees"] + bs["share S_bid>1+fees"])))
"""),
("code", r"""
row = h.loc[BASKET]
display(Markdown("### What we found\n" + "\n".join([
    f"1. **The total really does stay close to $1.** It averages ${S.mean():.4f}, slightly above $1, and 98% of the time it is within "
    f"${q01:.3f} to ${q99:.3f}.",
    f"2. **It is pulled back towards its average** (ADF p-value {R.fmt_p(adf.loc['TOTAL S', 'ADF p-value'])}), while the individual outcomes "
    f"wander. Gaps halve in about {half_life_min / 60:.1f} hours.",
    f"3. **But the moves are small compared with trading costs.** A round trip costs about ${row['round-trip hurdle']:.3f} per basket, while "
    f"the total typically moves about ${row['sd ΣP (1-min)']:.3f}. Notebook 02 tests whether a trading strategy can still make money.",
])))
_ = R.save_dataset_info("01", [info] + ([books[2]] if books is not None else []),
                    {"basket_stats": R.basket_stats_table(cfg, [BASKET]).reset_index().to_dict("records"),
                     "half_life_min": float(half_life_min), "hurdles": h.reset_index().to_dict("records")})
"""),
("md", r"""
### Caveats
* The once-a-minute prices are sampled separately for each outcome, so short spikes in the total can just be timing noise.
* This is one event over 30 days. News days (data releases, Fed speeches) can shift the average level.
"""),
]
