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

## 2026-05-04 — FVG logging (deferred)

Structure event log currently captures BOS / CHoCH / range break / rejection
only. FVG presence affects OB quality and would benefit from the same
"why was this filtered out" visibility.

**Revisit when:** structure logging has stabilised and we are auditing OB
quality alerts that surprise us.
