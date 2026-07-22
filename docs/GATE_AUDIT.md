# Health-Check / Gate Audit (read-only)

Date: 2026-07-21. Question asked: *do the backtest's integrity gates actually fire, or can they pass on nothing — and what silent killers do they NOT catch?*

---

## Headline (plain English)

- **You already have the health-check you asked me to build.** Your 18-year backtest ends with a **10-gate integrity scan**. On any failure it emails you what broke and turns the GitHub run **red** — the run is stamped "not trusted."
- **The gates are real, not paper.** A separate self-test **plants fake bugs and proves each gate catches them** — and that self-test passes today and runs in CI on every push. This is the single most important finding: your safety net is *proven to bite*, not just present.
- **On your real 18-year run, all 10 gates ran and passed.** Verified from that run's own `run_health.json`.
- **So: do NOT build a second health-check.** That would duplicate a working system (against your own "one implementation" rule). Instead we found **two genuine holes** worth a small, cheap patch each.

---

## What each gate protects (plain English)

| Gate | Catches | Proven to bite? |
|---|---|---|
| G1 | P&L headline disagreeing with the raw per-trade numbers or the email | ✅ planted-drift test |
| G2 | Price bars that should've been scanned but weren't — incl. the "blind run produced trades from nothing" trap | ✅ planted-gap test |
| G3 | Scrambled time-order (block→break→alert→fill) / lookahead | ✅ planted-causality test |
| G4 | Any "should never happen" error counter above zero | ✅ planted-anomaly test |
| G5 | Settings silently drifting mid-run | ✅ planted-knob-drift test |
| G6 | "What-if" TP numbers leaking into the real headline | structural (never sums what-if) |
| G7 | Records a fingerprint so two runs can be proven identical | ✅ determinism test |
| G8 | Missing-data creeping in (warn >1%, fail >5% per pair) | threshold check |
| G9 | Records not matching the required shape / bad timestamps | ✅ planted-bad-ts test |
| G10 | Physically impossible per-trade numbers (fake price excursions) | ✅ rule-set test |

**Verdict on the existing system: strong. Leave it running. It directly answers your "scream when a number is wrong" fear.**

---

## The holes (what the 10 gates do NOT cover)

### HOLE 1 — Nothing guards the trades.csv **column set**. (Highest value.)
- Your CANONICAL file is defined as **113 columns**. Your own rule says "wrong count = wrong file = STOP."
- But **no gate checks that a fresh run actually produced those 113 columns.** If a code change silently drops or renames a column, the gates above still pass (they check *values*, not *which columns exist*), and you'd only notice during analysis — exactly the "after 18 years I keep finding bugs" pain.
- `test_schema_version.py` does NOT cover this — it versions the *state files*, not the trades.csv schema.
- **Fix (cheap):** one gate — "the run's trades.csv has exactly the expected column set." Expected set lives in one place. Missing/extra/renamed column → RED. Token cost: near zero (reads one header row).

### HOLE 2 — G7 fingerprints each run but **nothing compares fingerprints across runs**.
- G7 records a hash so re-runs *can* be proven identical — but no step actually *does* the compare. Determinism is testable, not tested, on real runs.
- **Lower priority** — determinism is already proven elsewhere (memory: determinism runs done 2026-07-04). This is a "nice to close later," not a silent killer today.

---

## Recommendation

1. **Keep the 10-gate system as-is.** It works and it bites.
2. **Add ONE new gate: the column-schema gate (Hole 1).** Small, token-cheap, and it plugs the exact gap between "run finished green" and "the file I analyze is the right shape." This is the highest-value single thing in this whole audit.
3. **Park Hole 2** — log it, don't build it now.

The column gate is a **plumbing/correctness** change, not trading logic — so under our rules it's my call to propose and, once you approve, build with a self-test that plants a dropped column and proves the gate goes red.
