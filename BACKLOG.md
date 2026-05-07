# Trading Alerts System — Backlog & Open Methodology Questions

Append-only notes for ideas, deferred decisions, calibration questions, and
methodology items that came up during development but did not need immediate
action. Reviewed when we have data or capacity to revisit.

Format: dated entry, short title, the question / idea, the reason it was
deferred, and what evidence / data would settle it.

---

## 2026-05-04 — Caution thresholds (3 / 4 / 5) need empirical calibration

The BOS continuation caution thresholds for forex / commodity / index are
trader-set defaults based on general market behaviour. They have not been
tested against historical data from this system.

**Revisit when:** ~1 month of structure event logs are available.
**Evidence required:** distribution of how many BOS events typically print
before a CHoCH on each pair type. If forex routinely makes 4-5 BOS legs,
the threshold should rise; if NAS100 rarely exceeds 3, it should drop.

---

## 2026-05-04 — Pair-aware mitigation rules (3-touch proximal)

Current rule: an OB is mitigated if proximal is touched 3 times OR distal is
closed through. This is intentional but non-standard SMC. Standard SMC
mitigates the moment price closes inside the OB body.

**Risk:** indices and gold often have wider noise around levels. The 3-touch
rule may kill OBs prematurely on those instruments.

**Revisit if:** we see trades skipped on NAS100 / Gold because the OB was
already marked mitigated when price returned to retest.
**Possible solution:** pair-aware touch counts (e.g. forex = 3, gold/index = 5).

---

## 2026-05-04 — Range break: should it use a higher ATR threshold than CHoCH?

Currently proposed: range break uses 0.6× ATR (same as CHoCH). The argument
for parity: structural significance is already established by the level
itself; threshold only filters noise. The argument for stricter (0.8×+):
range walls are the most-defended levels and breaks should require more
conviction.

**Revisit when:** logs show false range-break events that reversed quickly,
or genuine range breaks that scored marginally.

---

## 2026-05-04 — Lookback policy: should range break use lookback=2 anchors?

Currently proposed: range break uses lookback=3 (the dealing range walls).
This is correct for stability of the major range, but may miss situations
where lookback=2 would have caught the reversal earlier.

**Revisit when:** we see CHoCH fire (lookback=2) but range never breaks
(lookback=3) and the trend genuinely reversed — meaning lookback=3 was too
slow at the major level too.

---

## 2026-05-04 — Body minimum threshold (currently proposing 20%)

Approved change: raise OB candle body minimum from 10% to 20% of candle
range. Reasoning: 10% allows near-doji candles which lack directional intent.
20% is a calibration choice — could be 15-25% and still defensible. A vet
does not quantify this with a single number; an automated system must.

**Revisit when:** we see valid OBs being rejected because the last opposing
candle had a body of 16-19%.

---

## 2026-05-04 — Impulsive break: math in email until system is trusted

Body / range ratio with the actual numbers (e.g. "35 pips / 42 pips = 83%")
will appear in alert emails so the user can verify the math is sane.

**Remove math when:** user has reviewed enough alerts to trust the metric.
The flag itself stays; only the working numbers come out.

---

## 2026-05-04 — Logs cleanse cadence (weekly)

Structure events log lives at `logs/structure_events_YYYY-MM.json`,
gitignored. Volume estimate: ~500 KB / week across all pairs, mostly
rejected breaks. To prevent local bloat, a weekly archive / email step
should be added to `weekly_review.py` — emails the current month's log to
the user and resets it.

**Action when ready:** wire archive step into weekly_review.py.

---

## 2026-05-06 — GitHub Actions log persistence (open)

The logs/ directory is gitignored. GitHub Actions runners are ephemeral —
events written during a workflow run are LOST when the runner shuts down.
This means logs only persist when the system runs locally.

**Options to evaluate:**
- (a) Commit logs to git from the workflow (defeats "keep git clean" goal,
      but simplest persistence).
- (b) Upload logs as Actions artifacts (90-day retention, manual download).
- (c) Email logs after every Actions run (high volume, noisy).
- (d) Email weekly digest from a separate workflow scheduled Sunday.

**Recommended path:** (d). One scheduled workflow downloads accumulated
artifacts, emails the digest, no git bloat. Needs design once logging is
stable.

---

## 2026-05-07 — Wick-based swing definition may misidentify stop runs as swings

Swing detection uses candle High/Low (wicks), not Close. A single-candle stop
run (long wick, body near opposite end) registers as a structural swing. Break
detection correctly uses Close, so the level being broken may not have been real
structure — only a wick.

**Impact:** Low in normal conditions. More relevant on Gold and NAS100 around
news spikes and Asia-session wicks.
**Revisit when:** alerts fire against a wick-only extreme that a vet would call
a liquidity grab, not structure.

---

## 2026-05-04 — FVG logging (deferred)

Structure event log currently captures BOS / CHoCH / range break / rejection
only. FVG presence affects OB quality and would benefit from the same
"why was this filtered out" visibility.

**Revisit when:** structure logging has stabilised and we are auditing OB
quality alerts that surprise us.

---

## 2026-05-08 — Phase 2 zone invalidation via state read (revisit during Entry pass)

`smc_detector.check_opposite_bos` was rewritten from a self-contained
walk-forward to a state reader that consumes the dealing_range event ring.
Current behaviour: a zone is killed if the most recent event for the pair
(within `since_ts`) is a BOS or a **Major** CHoCH in the opposite direction.
Minor CHoCH is intentionally ignored (consistent with the locked rule that
Minor does not flip trend).

**Revisit when:** we work on Entry-side logic (limit-order vs market-trigger
nuances). Open questions:
- Should Minor CHoCH ever invalidate a zone (for example, against pristine
  ones with no other confirmation)?
- Should the lookup window (`since_ts`) treat ring overflow specially when
  the ring has been pruned past the alert's emission ts?
- Is "any BOS in opposite direction" too aggressive for short-watch zones
  during news bars where micro-BOS prints in both directions?

**Action when ready:** wire concrete decisions into the Entry pass. Until
then, current behaviour is preserved.

---

## 2026-05-08 — Phase 3 M5 CHoCH consistency with new H1 rules

`smc_detector.detect_ltf_choch` (the M5 entry trigger used by
NAS100 / GOLD `entry_model: ltf_choch`) keeps its own M5-local logic — it has
no premium/discount-zone gate and no Major/Minor distinction. By design it
is intentionally permissive: an M5 CHoCH inside the H1 zone (or within an
M5-ATR grace band) fires the trigger.

**Revisit when:** we work on Phase 3 quality (entry triggers, hit rate).
Open questions:
- Should M5 CHoCH require a reversal from a fraction of the M5-zone's range
  (analog to H1's premium-zone gate)?
- Should M5 distinguish Major vs Minor (lookback=3 vs 2) on M5?
- Currently any swing-low / swing-high inside or near the zone qualifies —
  is that catching too many low-quality M5 reversals?

**Action when ready:** decide whether M5 should mirror H1 zone-gating or
keep its current permissive behaviour as the entry trigger.

---

## 2026-05-08 — Caution thresholds revisit after BOS-counter rebuild

`compute_bos_sequence_count` now reads from the dealing_range event ring
(capped at 20). Counter resets only on Major CHoCH. Caution thresholds
(Forex 3, Commodity 4, Index 5) trigger the "exhausted trend" structure
score (1.0) and were calibrated against the OLD detection (lookback=4
self-contained walk).

**Revisit when:** ~1 month of fresh logs accumulate under the new event
schema. The new detection is stricter (premium-zone gate on CHoCH, no
events on placeholder walls). It will likely produce FEWER CHoCHs and
LONGER BOS sequences before reset → caution thresholds may need to rise.

**Evidence required:** distribution of BOS-sequence-counts at the moment
of CHoCH across pairs, from `logs/structure_events_*.json`.

---

## 2026-05-08 — Strong / Weak high-low classification on OBs (scheduled: 2026-05-09)

LuxAlgo distinguishes "Strong" vs "Weak" swing highs/lows: a high made
during a downtrend is Strong (held the move); a high made during an uptrend
is Weak (likely to break). Mirror for lows. A bullish OB anchored at a
Strong Low is higher quality than one at a Weak Low.

**Status:** approved for implementation 2026-05-09. Will tag each OB with a
strong/weak attribute derived from the prevailing trend at the swing's
formation. Wired into Phase 2 confidence scoring (not a Phase 1 hard gate).

**Note:** see Benchmarking.md section 4 for the source comparison.

---

## 2026-05-08 — Breaker block detection (deferred)

When a bullish OB closes-through to the downside, that zone often flips
and acts as resistance — a "breaker block." Currently we delete the OB on
mitigation (close past distal). Breaker tracking would keep the dead zone
on the chart, flipped, with a separate label.

**Defer until:** Phase 1 + Phase 2 are stable end-to-end. Adding breaker
tracking before the base OB lifecycle is trusted compounds debugging.

**Evidence to revisit:** observed cases where price retests a recently
mitigated OB from the opposite side and reverses — i.e. breaker behaviour
that we currently ignore.

---

## 2026-05-08 — Minor CHoCH OBs use the same "confirmed the X shift" narrative

The fallback narrative (and Gemini prompt) say "{event} confirmed the
{direction} shift." For Minor CHoCH this is misleading: Minor CHoCH does
NOT flip trend (it's a weakening flag inside the prevailing trend). The
sentence is grammatically intact thanks to the new event-label change but
semantically loose for Minor CHoCH OBs.

**Defer until:** narrative copy pass. Low priority — surface label is
correct; only the surrounding template language is off for Minor.

**Possible fix:** branch the narrative template by tier, or rephrase to
"this OB marks the institutional candle behind the {event} print."

---

## 2026-05-07 — Historical backtest harness (replay through Phase 1/2/3)

Goal: feed historical OHLC data through our existing Phase 1, Phase 2, and
Phase 3 modules and record every alert that would have fired. Output: a CSV
per instrument per month with timestamp, OB, score, breakdown. Then compute
a hypothetical win rate per month so we can answer "would this system have
made money in June-Oct 2025?"

**Why we want it:**
- Verify the system on history before risking real money on it
- Catch regressions when methodology changes (alerts shift unexpectedly)
- Establish a per-month / per-pair win-rate benchmark

**Approach (locked):**
- Use the `backtesting.py` library as a thin harness. Inside its
  `Strategy.next()` we feed bars to our actual production modules — no
  reimplementation of detection logic. One concept, one implementation.
- Historical bars: yfinance for FX/Gold/NAS H1/M15/M5 (~2 years available).
- Output: CSV of would-be alerts. Initial grading is mechanical (price hit
  TP / SL / time-exit). Vet-eye grading layered on later via weekly review.

**What this is NOT:**
- Not a replacement for the real weekly review (which does much more).
- Not auto-grading by price alone — that grades luck, not setup quality. The
  CSV is a starting point; a vet still has to look at the borderline ones.

**Trigger:** build only AFTER the live system is stable enough that we
trust its alert generation. Building a backtest harness on top of moving
methodology wastes effort.

**Evidence required to call it a success:** ≥60% mechanical win rate over
6 months of historical data, OR a clearly explainable failure mode that
points at a fixable methodology issue.
