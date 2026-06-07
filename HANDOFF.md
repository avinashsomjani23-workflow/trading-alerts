# Handoff — Trend Engine Consolidation + OB Surfacing

## Problem statement

Two H1 trend engines run in parallel on the same swing pool (`dealing_range.detect_swings`, lb-3 + ATR):

- **`structure_engine.py` (`structure_v2` on state)** — reads HH/HL/LH/LL swing structure + the confirmed H4 dealing range (`h4_range.py`). This is the intended model.
- **Legacy `dealing_range.py`** — sets `trend` via a wall-break model (Major/Minor CHoCH) on H1 walls, and additionally produces the dealing-range display walls (H1), the events ring, and the OBs.

Phase 2 gating reads the legacy `trend` (`Phase2_Alert_Engine.py` ~1909 → `smc_detector.compute_bos_sequence_count`). The chart trend banner reads `structure_v2` (`smc_radar.py` ~2659, ~4008). So display and alerts can disagree.

Known defect in `structure_v2`: the premium/discount 25% requirement is a hard gate on the trend flip. If trend is bearish and price reverses upward from mid-range without tagging the bottom 25%, the up-flip never qualifies — trend stays bearish indefinitely even as price breaks every level up (`structure_engine.py:176-182`). A veteran reads structure breaking as the trend changing; the 25% zone is an entry-quality filter, not a lock on trend recognition.

## Pending work (decided / approved)

1. Make `structure_v2` (HH/HL/LH/LL + H4 range) the single source of truth for trend/CHoCH. Repoint Phase 2 gating and the chart banner to it.
2. Remove the 25% premium/discount requirement as a *gate* on the trend flip; keep it as a quality label only. Fixes the dead-state above.
3. Trend must flip on the CHoCH itself — remove `structure_v2`'s transition state (no transition wait, no post-CHoCH BOS wait, no second ATR knob).
4. Before deleting legacy trend logic: fully trace that walls (H1 display range), the events ring, and OB building are repointed/preserved — legacy currently owns them.
5. Rename CHoCH labels to "Up CHoCH" / "Down CHoCH".
6. Add H1 trend/CHoCH state to the Phase 1 table (↑ / ↓ / U) and to the zone card/chart.
7. Threshold replay: compare `CHOCH_ATR_MULT` 0.6 vs 0.85 on logged scans, measuring true-flip rate (CHoCH led to a real reversal vs price reclaimed and resumed), then pick the value.
8. Remove dead `active_unmitigated_obs` key (real persisted key is `active_zones`).
9. Phase 1 email redesign — reorganize/clarify, drop non-useful fields.

## Open decisions for next chat

- Q2: display walls H1 vs H4. The H4 dealing range exists in `h4_range.py`; legacy display walls are H1. Decide which feeds the displayed dealing range.
- Item 7: define the follow-through metric precisely (next-opposite-swing break, or ≥N ATR move before reclaim — and N).
- Whether to push the OB changes (below) now or bundle with the engine swap.

## Done this session (working tree, NOT pushed)

OB selection rewritten in `smc_radar.py`:
- Single proximity window — both OBs gated at `OB_PROXIMITY_ATR = 5.0` (replaces the OB1 inner 4 / OB2 outer 12 ring).
- Selection: nearest two within window, pristine (touches == 0) preferred on ties, no direction/trend gating, hard cap `OB_MAX_KEEP = 2`.
- Build-time proximity gate and slate-drift `out_of_proximity` both moved to `OB_PROXIMITY_ATR`.
- `OB1_INNER_LIMIT_ATR` / `OB2_OUTER_LIMIT_ATR` constants removed; `_split_primary_alternative` rewritten; `h1_trend` arg dropped.
- `smc_radar.py` parses clean. No stale references to removed constants.

## Confirmed facts (verified in code this session, do not re-derive)

- `structure_v2` consumers found are display-only (chart caption `smc_radar.py:2659-2669`, banner note ~4008). No gating consumer found. (Full reference sweep not yet exhaustive — item 4.)
- `structure_engine.compute_structure(df, h4_range)` reads the confirmed H4 range (`smc_radar.py:3368`).
- Legacy `trend = event_direction` flips on the CHoCH candle (`dealing_range.py:1557`) — legacy already flips on CHoCH, no transition wait. The transition wait exists only in `structure_v2`.
- Legacy CHoCH has a Major/Minor split (`dealing_range.py` `_pick_choch_pivot`); Minor CHoCH does not flip trend / move walls. `structure_v2` has one CHoCH definition (no tiers).
- OB build loop processes every event incl. CHoCH (`smc_radar.py` ~617, CHoCH handled ~693). The NAS100 05-Jun bearish CHoCH OB built; it was hidden by the old OB1/OB2 ring dead-band (now fixed by the single-window change).
