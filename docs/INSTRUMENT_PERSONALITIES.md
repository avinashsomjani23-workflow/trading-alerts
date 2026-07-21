# Instrument Personalities — Reference Baseline

**Purpose:** A frozen description of how each of our 5 live-traded instruments *typically* behaves, based on widely-accepted, textbook FX/commodity market knowledge — NOT on our own backtest data.

**How to use this file:**
- When we backtest or analyze a pair, we compare the data against this baseline.
- **Data agrees with this file →** likely a real behavior, higher confidence.
- **Data disagrees with this file →** a *discussion point*, not a conclusion. Something is off: a detector bug, a thin sample, a timeframe mismatch, OR a genuine regime change worth investigating. Never silently trust data that contradicts widely-accepted behavior without explaining why.
- This file is **background knowledge**, not code truth. It never overrides `TRUTH_LEDGER.md` or live code. It is a sanity-check lens.

**Scope:** Only widely-accepted, mainstream behaviors. No exotic theories, no our-own-findings. If a claim isn't broadly agreed on by professional FX desks / textbooks, it doesn't go here.

**Timeframe note:** We trade H1 only. These personalities describe general behavior; where a trait is timeframe-specific it's noted.

---

## Plain-English glossary (read this first)

These are the finance terms I used before without explaining. Here they are in plain words.

- **Carry / carry-driven:** Every currency has an interest rate set by its central bank. If you hold a currency with a high interest rate and borrow one with a low rate, you earn the *difference* just for holding — that's "carry." When lots of traders do this, it pushes the high-rate currency up and the low-rate one down in long, steady trends. A pair is "carry-driven" when this interest-rate gap is a major force moving it. **Trading effect:** carry pairs tend to *trend* (move in one direction for a long time) rather than chop around.

- **Safe haven:** In a panic (war, crash, banking scare), big money runs to assets seen as "safe" — historically the US dollar, the Swiss franc, the Japanese yen, and gold. When fear hits, these get *bought* regardless of their interest rate. **Trading effect:** safe havens can suddenly jump on fear headlines that have nothing to do with normal economics, which makes them spike unpredictably.

- **Risk-on / risk-off:** "Risk-on" = markets are calm/greedy, money flows into higher-yielding, riskier currencies (like NZD, AUD). "Risk-off" = markets are scared, money flees to safe havens (USD, CHF, JPY, gold). Many pairs are basically a bet on the market's *mood*, not on the two countries' economies.

- **SNB / BoJ / RBNZ / ECB / Fed:** Central banks. **Fed** = USA. **ECB** = Europe (euro). **BoJ** = Japan. **SNB** = Switzerland. **RBNZ** = New Zealand. They set interest rates and sometimes directly intervene in the currency.

- **Intervention:** When a central bank *directly buys or sells its own currency* to force the price where it wants. This is rare but violent — it can move a pair hundreds of pips in minutes and reverse a clean trend instantly. Japan (BoJ) and Switzerland (SNB) are the famous interveners among our pairs.

- **Liquidity / thin / deep:** "Liquidity" = how much money is actively trading a pair. A **deep** (liquid) pair like EURUSD has enormous volume, so it moves smoothly and is hard to push around. A **thin** (illiquid) pair like NZDUSD has less volume, so it jumps around more, spikes harder, and reverses more violently on the same news. **Trading effect:** thin pairs whip; deep pairs glide.

- **Weakest edge / hard to trade:** The more people trade a pair and the more efficient it is, the harder it is to find a repeatable pattern that makes money — because everyone else has already found and traded away the obvious ones. EURUSD is the most-traded pair on earth, so simple edges on it are "weak" (small or already competed away). A less-watched pair can hide a cleaner edge.

- **Mean-reverting vs trending:**
  - **Trending** = price tends to keep going the same direction; breakouts follow through. Good for "break and continue" trades.
  - **Mean-reverting** = price tends to snap back to where it came from; breakouts often fail and reverse. Good for "fade the extreme" trades, bad for breakout trades.
  - This distinction is the single most important personality trait for us, because our SMC logic (BOS/CHoCH/OB) behaves very differently on a trender vs a mean-reverter.

- **Follow-through:** After price breaks a level, does it keep going (good follow-through) or immediately stall and reverse (poor follow-through)? Poor follow-through = lots of *fake* breaks = breakout strategies get chopped up.

- **Sweep / stop hunt:** Price briefly pokes past a level to trigger stop-losses, then reverses. A pair that does this a lot is "sweep-heavy" — it fakes you out before the real move.

---

## Cross-cutting concepts that shape all 5

- **USD is on one side of 4 of our 5 pairs** (EURUSD, USDJPY, NZDUSD, USDCHF all contain USD; GOLD is priced in USD). So **US news (Fed, CPI, NFP) moves ALL of them at once.** This is central to Topic 2 (news mapping): a single USD event hits every pair simultaneously.
- **Sessions matter.** Each currency is most active during its home trading hours:
  - **Asian session** (Tokyo): JPY, NZD most active.
  - **London session:** EUR, CHF, and the biggest overall FX volume.
  - **New York session:** USD, and overlaps with London for the highest-liquidity window of the day.
  - GOLD is most active in London + NY.
- **The "smile" of the US dollar:** the USD tends to strengthen BOTH in extreme fear (safe-haven buying) AND in strong US growth (rate/carry buying), and weaken in the calm middle. This is widely called the "dollar smile."

---

## EUR/USD — "The Benchmark"

**One-line personality:** The deepest, most efficient, most-watched pair in the world — smooth, orderly, but the hardest place to find an easy edge.

**Widely-accepted traits:**
- **Most liquid / deepest FX pair on the planet.** Huge daily volume. This makes its price action relatively *smooth and orderly* — fewer random violent spikes than thinner pairs.
- **Most efficient = weakest simple edges.** Because everyone trades it, obvious patterns get competed away. Clean textbook structure, but small/hard-won edges.
- **Ranges a lot / mean-reverting tendency in normal conditions.** EURUSD spends long stretches grinding in ranges rather than trending hard, especially when Fed and ECB policy are similar. Breakouts often fail back into the range.
- **Driven by Fed-vs-ECB rate expectations.** The big trends come when US and European interest-rate paths diverge.
- **Best behavior during London + NY sessions.** Asian session is often quiet/rangey for EURUSD.
- **Not a safe haven** in the classic sense; it's a "growth/rates" pair between two large developed economies.

**What this implies for our SMC logic:**
- Expect **clean structure but frequent failed breaks** → BOS breakouts may follow through less than on a trending pair. Watch for CHoCH/reversal setups working better than pure continuation.
- Edges here are likely **small and need the tightest statistical validation** — easiest pair to fool ourselves on because the structure looks textbook.

---

## USD/JPY — "The Trend Machine (until Japan steps in)"

**One-line personality:** A strong, carry-driven trender that moves in long clean directions for months — but the Bank of Japan can violently intervene and reverse it without warning.

**Widely-accepted traits:**
- **Carry-driven and strongly trending.** For long periods, the large gap between US and Japanese interest rates pushes USDJPY up in extended, relatively clean trends. It's the textbook "trend" pair among majors.
- **Very sensitive to US interest rates / US bond yields.** When US yields rise, USDJPY usually rises. This makes it react hard to US CPI, NFP, and Fed decisions.
- **Yen is a safe haven.** In global panics, money buys the yen → USDJPY can *drop* sharply on fear even when nothing changed in the US. So it has two modes: calm carry-trend UP, and fear-driven spike DOWN.
- **Intervention risk (BoJ).** Japan has a history of directly intervening to stop the yen getting too weak (USDJPY getting too high). These are rare but *massive and sudden* — hundreds of pips, instantly. Any USDJPY analysis must respect that some big reversals are policy shocks, not structure.
- **Most active in Asian + NY sessions.**

**What this implies for our SMC logic:**
- **Continuation/BOS setups should work relatively well** here because it genuinely trends — this is the pair where "break and continue" is most trustworthy.
- But guard against **outlier reversals from intervention/safe-haven flows** — some losses will be un-model-able policy shocks, not detector failures. Don't tune the model to "fix" an intervention candle.

---

## NZD/USD — "The Whippy Risk Barometer"

**One-line personality:** A thin, jumpy, risk-mood pair that spikes and reverses violently — the most mean-reverting / sweep-prone of our five.

**Widely-accepted traits:**
- **Thin / lower liquidity** vs the majors. Less money trades it, so it moves in sharper, jumpier steps and reverses harder on the same news.
- **A "risk-on/risk-off" barometer.** NZD is a higher-yielding, growth/commodity-linked currency. It rises when markets are greedy (risk-on) and falls when markets are scared (risk-off). Often it's trading global *mood*, not New Zealand's economy.
- **Commodity-linked.** New Zealand's economy leans on commodity/dairy exports and it tracks broader commodity and China-growth sentiment.
- **More mean-reverting / whippy on short timeframes.** Because it's thin and sentiment-driven, it's prone to spikes that snap back — more false breaks and stop-hunts than the deep majors.
- **Reacts to BOTH RBNZ (New Zealand) and US events**, but on a normal day USD news usually dominates the move because the US side is far larger.
- **Most active in Asian session** (NZ/Australia hours) and around US data.

**What this implies for our SMC logic:**
- Expect **more failed breaks, more sweeps, more reversals** → pure continuation/BOS breakouts are riskier here. **Sweep-then-reverse** and **liquidity-grab** setups may fit its personality better.
- Highest whipsaw pair → our stop-outs and "instant death" behavior may cluster here. A signal that works on USDJPY may *invert* on NZDUSD — this is exactly why we analyze per-pair.

---

## USD/CHF — "The Safe-Haven Fake-Out"

**One-line personality:** A safe-haven pair driven by fear and the Swiss central bank, prone to false breaks and sudden policy shocks — often behaves like an inverted EURUSD.

**Widely-accepted traits:**
- **Swiss franc (CHF) is a premier safe haven.** In global fear, money floods into the franc → USDCHF *falls* (franc strengthens) on risk-off, regardless of US data. Like the yen, it has a "fear mode."
- **Heavily influenced by the SNB (Swiss National Bank).** The SNB actively manages the franc and has a history of dramatic policy actions — most famously removing its EUR/CHF floor in Jan 2015, which caused one of the largest single-day FX moves in history. **SNB action = tail-risk violent moves.**
- **Tends to move opposite to EURUSD.** Because both involve European vs USD dynamics, USDCHF and EURUSD are strongly *negatively correlated* — they're often near mirror images. If you understand EURUSD, USDCHF often looks like its flip.
- **Prone to false breaks / poor follow-through** in ranges, partly due to safe-haven flows and SNB management overriding "normal" technical behavior.
- **Most active in London + NY sessions** (European currency).

**What this implies for our SMC logic:**
- Expect **fake breakouts and safe-haven-driven reversals** that don't respect structure → similar caution to EURUSD (its mirror), with added **tail-risk from SNB shocks**.
- Some of its moves are **fear flows, not structure** — like USDJPY's intervention risk, don't tune the model to explain an SNB/panic candle.

---

## GOLD (XAU/USD) — "Its Own Animal"

**One-line personality:** Not a currency pair at all — a fear-and-inflation hedge that trends powerfully, spikes on panic, then snaps back fast. Behaves differently enough that FX intuition often fails on it.

**Widely-accepted traits:**
- **A commodity / store of value, not a currency.** Priced in USD, but it's driven by its own forces: inflation fear, real interest rates, and crisis demand — not by a "gold central bank."
- **Premier safe haven + inflation hedge.** In fear or high inflation, money buys gold → it can rip higher on panic headlines. It's THE classic crisis asset.
- **Inverse to real US interest rates.** When real (inflation-adjusted) US rates rise, gold usually falls (because gold pays no interest, so higher rates make it less attractive); when real rates fall, gold rises. So it reacts hard to Fed/CPI, but often *opposite* to the way USDJPY does.
- **Strong, sustained trends** — gold is known for long directional runs, more than most FX pairs.
- **Sharp V-shaped spikes that reverse fast.** Gold is famous for violent spikes on news that partly retrace quickly (fast "V" moves). *(Note: our own retrace research is consistent with this — but that's our data, not this file's remit; treated as corroboration only.)*
- **Most active in London + NY sessions.**

**What this implies for our SMC logic:**
- **Trend/continuation setups can work well** (it trends), BUT expect **violent spikes and fast retraces** around news that hurt tight stops.
- Gold is the pair **most likely to legitimately behave differently** from the FX four — a feature that's noise on FX can be real signal on gold, and vice versa. Do NOT assume FX conclusions transfer to gold.

---

## Quick-reference matrix

| Pair | Trend vs Range | Liquidity | Safe haven? | Special shock risk | Key driver |
|------|----------------|-----------|-------------|--------------------|------------|
| **EURUSD** | Range/mean-revert | Deepest | No | — | Fed vs ECB rates |
| **USDJPY** | Strong trend | Deep | Yen = haven (down-spikes) | BoJ intervention | US yields / carry |
| **NZDUSD** | Whippy/mean-revert | Thin | No (risk-on) | — | Global risk mood |
| **USDCHF** | Range, fake breaks | Deep | CHF = haven (down-spikes) | SNB policy shock | Fear + EUR mirror |
| **GOLD** | Strong trend + V-spikes | Deep | Yes (premier) | Crisis demand | Real rates / inflation / fear |

---

## Guardrails on using this file

- **This is a lens, not a law.** A widely-accepted trait can still be wrong for a specific window. Use it to *ask questions*, not to force conclusions.
- **Data-vs-methodology rule (from CLAUDE.md):** if our data disagrees with this file, that's a DISCUSSION POINT — name the likely cause (detector bug / thin sample / timeframe / real regime change) before acting. Never score or filter on a disagreement without explaining it.
- **A thin sample never overrides a widely-accepted behavior; weak data doesn't disprove a sound market truth.**
- **This file describes typical/average behavior.** Any specific day can violate it — intervention, SNB shocks, and crises are exactly the exceptions that prove the rule.
