# Future To-Do List — parked ideas & knowledge for when we build

A running list of things we are **NOT building now** but want ready for later. Each
item is self-contained: what it is, in plain English, plus a **Trigger** line saying
exactly when to open and use it. Concept/knowledge notes only unless an item says
otherwise — no code, no backtest numbers baked in.

**How to use this file:** scan the list below. Work an item ONLY when its Trigger
fires. Add a new item by appending a new `## Item N` section and a row to the list.

## The list

1. **CTA / trend-following notes** — what a CTA is and their recipe; ready for the day
   we build a momentum/continuation entry, a volatility-based sizing layer, or a
   trailing "let winners run" exit. *(parked knowledge, not a build)*

<!-- Add future items here: 2., 3., ... each with a matching "## Item N" section below. -->

---

## Item 1 — CTA / Trend-Following notes

**Status: PARKED. Not building this now.** This item exists so that WHEN we build
something new (a momentum / continuation entry, a position-sizing layer, or a
"let winners run" exit), we already have the concept written down in plain English.
It is a knowledge note, not a spec and not a decision.

**Scope of this item: the CTA concept ONLY.** No strategy design, no code, no
numbers from our backtest. Just "what a CTA is and how they think", so a future
chat starts educated instead of from zero.

Everything below is plain English on purpose. Where a claim comes from research,
the source is named at the bottom (all peer-reviewed / academic — nothing for sale).

### 1.1 What a CTA is (in one paragraph)

**CTA = Commodity Trading Advisor.** The name is old and misleading — they do NOT
only trade commodities. A CTA is a fund that trades **many markets as futures**:
currencies, gold, oil, stock indices, government bonds. The style they are famous
for is **trend-following** (also called **"managed futures"** or **"time-series
momentum"**). They have run real money this way for decades, at billions of dollars
of size. Well-known names: **Winton, Man AHL, Aspect, Transtrend, Campbell, Millburn.**

**Why they matter to us:** they are the large-scale, long-track-record PROOF that
trend-following works in **our exact markets — FX and gold.** If we ever move toward
a momentum / continuation entry, this is the world whose ideas we would borrow.

### 1.2 The one-sentence idea they trade

> A market that has been going up tends to keep going up for a while; a market that
> has been going down tends to keep going down. Ride that, on each market judged
> against its own history.

This is **time-series momentum**: each market is judged against **its own past**
(is gold higher than it was N months ago?), NOT against other markets. That is the
cousin of what we do (single-market, event-driven), so it is the relevant one for us.

### 1.3 Their recipe — the 5 parts, plain English

Their secret code is private, but **academic replications copy 75%+ of their
returns with a simple recipe.** The five parts:

1. **Signal — "is this market trending, up or down?"**
   Two common ways: (a) the **sign of the past N-month return** ("up over the last
   12 months → go long"), or (b) a **moving-average crossover** (a fast average of
   price crossing a slow average). They usually **blend a few speeds** (fast, medium,
   slow) so they are not betting on one time-horizon.

2. **Position size — "trade smaller when the market is wild."**
   They size each trade **inversely to its volatility**, so every market contributes
   roughly the **same amount of risk**. A calm market gets a bigger position; a wild
   market gets a smaller one. (Our ATR-based stop sizing is a baby version of this
   same idea.)

3. **Diversify — "spread across dozens of markets."**
   FX, gold, other commodities, bonds, indices — all at once. No single reversal in
   one market can sink the fund. Diversification is a core part of WHY it works, not
   an optional extra.

4. **Exit — "cut losers fast, let winners run."**
   They use a **trailing / volatility-based stop or a signal flip** to get out.
   They do **NOT** use a tight fixed profit target, because the whole style depends
   on a few big winners paying for many small losers.

5. **Speed — "slow."**
   They hold **weeks to months** and rebalance daily or weekly. Turnover is low.

### 1.4 The most important honesty note for us

**The proven CTA edge is SLOW (weeks–months) and DIVERSIFIED (many markets).**
**Our system is FAST (intraday, H1) and FOCUSED (one setup, few pairs).**

So if we build a momentum / continuation entry:
- We **can borrow their ideas** — trend signal, volatility-based sizing, let-winners-run exit.
- We **CANNOT borrow their proof.** Their research is about a monthly hold across
  dozens of markets. It does **not** prove an intraday continuation entry on EURUSD.
  We would have to **re-prove that ourselves** on our own backtest harness, from scratch.

Treat any "CTAs make X% so our version will too" claim as **false by default** until
our own Discovery-block data says otherwise. Different speed = different game.

### 1.5 Two ideas from here worth stealing later (not now)

Written down so they are not forgotten. Neither is a decision.

- **Volatility-based position sizing / "turn it down when wild."** We already compute
  ATR, so the raw material is there. A future sizing layer could scale total exposure
  down in high-volatility regimes. This is a **position-size rule, separate from the
  per-trade stop** — the stop caps one trade; this caps how much we risk overall.
- **"Let winners run" exit.** Our current study uses a FIXED 2R ruler on purpose (a
  constant yardstick to compare ENTRIES). But for a LIVE momentum strategy, a trailing
  / volatility exit is how trend-followers actually make money, because the payoff
  comes from the rare trade that runs far. If we ever test a continuation entry, pair
  it with a trailing-exit test — do NOT judge a momentum entry on a fixed target alone.

### 1.6 Trigger — when to open this item

Open and use this note when we start any of:
- a **momentum / continuation entry** (entering on the break instead of waiting for a retest),
- a **position-sizing / risk-scaling layer** (size down in high volatility),
- a **"let winners run" trailing / volatility exit** (moving off the fixed-2R ruler for live).

Until then: parked. Do nothing with it.

### 1.7 Sources (all rigorous — peer-reviewed / academic, nothing for sale)

- Moskowitz, Ooi & Pedersen — *Time Series Momentum*, Journal of Financial Economics, 2012.
- Hurst, Ooi & Pedersen — *A Century of Evidence on Trend-Following Investing* (AQR).
- Baltas & Kosowski — *Momentum Strategies in Futures Markets and Trend-Following Funds*.
- Menkhoff, Sarno, Schmeling & Schrimpf — *Currency Momentum Strategies*, Journal of Financial Economics, 2012.

(Retail / practitioner explainers exist — e.g. AQR's "Demystifying Managed Futures"
overview, Quantpedia summaries — useful for plain-English intuition but NOT weighted
as evidence. The four above are the rigorous sources.)
