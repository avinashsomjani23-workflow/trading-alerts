# Scan Log Schema (schema_version 1.0.0)

Ground truth for the four artifacts written under
`backtest/out/scanlog/<run_id>/`. The conformance test
(`test_scanlog_self.py::test_schema_conformance`) validates real records
against this document; the first invalid record aborts (gate G9).

---

## Live-parity: the honest mapping

SPEC Â§5 asks the backtest heartbeat to reuse the LIVE scan-log field names so a
Tier-B matcher can join the two through one parser. The live file is
`phase2_scan_log.jsonl`. Its real per-record shape (read from the repo, not
assumed) is:

```
ts_ist, pair, zones_in_active_obs, current_price, h1_trend,
zone_outcomes[], final_action, trend_alignment
```

**Reality check (stated plainly, no overselling):** the live record and the
spec's proposed heartbeat share *few* field names. Most spec fields do not
exist live. So the mapping is partial, and documented here truthfully rather
than pretending the schemas already align.

| backtest scan field | live field            | relationship |
|---------------------|-----------------------|--------------|
| `pair`              | `pair`                | identical |
| `current_price`     | `current_price`       | identical meaning |
| `trend`             | `h1_trend`            | renamed; same value domain (`bullish`/`bearish`/null) |
| `n_active_zones`    | `zones_in_active_obs` | renamed; same count |
| `final_action`*     | `final_action`        | live-only label, copied through when available |
| `ts`                | (`ts_ist` is IST)     | **NOT** the same: backtest `ts` is the tz-aware UTC bar boundary; live `ts_ist` is a wall-clock IST stamp. Different timezone, different meaning. Do not join on it. |

Everything else in the backtest heartbeat is a **backtest-only extension**,
namespaced with `bt_` where it has no live counterpart, per SPEC Â§2.2.

`*` `final_action` is emitted only when the backtest can derive an equivalent
label; otherwise omitted (absence is allowed for this optional field).

---

## 1. `manifest.json` (object, written first)

| field | type | notes |
|-------|------|-------|
| `run_id` | str | matches the run directory name |
| `schema_version` | str | `1.0.0` |
| `generated_utc` | str (ISO) | the only timestamped element |
| `git_sha` | str | short SHA of the repo at run time |
| `risk_usd` | number | risk per trade |
| `min_warmup_bars` | int | MIN_WARMUP used by the walk |
| `fetch_pad_days` | int | history pad before the walk window |
| `pairs` | array | per-pair served record (below) |
| `knobs` | object | live-read snapshot of every Â§3 knob |
| `versions` | object | `python`, `pandas`, `yfinance` |

Per-pair served record inside `pairs[]`:

| field | type | notes |
|-------|------|-------|
| `name` | str | pair name |
| `symbol` | str | yfinance symbol |
| `requested_start` / `requested_end` | str (ISO) | what the run asked for |
| `served_start` / `served_end` | str (ISO) | what the data actually covered |
| `n_bars` | int | bars served in `[requested window]` |
| `fingerprint` | str | 16-hex content hash of the OHLC frame |
| `prox_cap_atr` | number | the `atr_multiplier` in force for THIS pair |

---

## 2. `scan_log.jsonl` (one record per pair-bar)

Required fields on every record:

| field | type | notes |
|-------|------|-------|
| `ts` | str (ISO, tz-aware UTC) | the wall-clock bar boundary; must be in `df.index` |
| `pair` | str | |
| `outcome` | enum | one of the Â§ outcome set below |
| `bt_slice_mode` | str | always `"B"` (literal, asserted) |

Optional / context fields (present when meaningful):

| field | type | notes |
|-------|------|-------|
| `just_closed_ts` | str (ISO) | `slice.index[-1]` |
| `n_bars_in_slice` | int | |
| `atr` | number \| null | **null when NaN - never a fake number** |
| `trend` | str \| null | from `compute_pair_walls` (maps live `h1_trend`) |
| `n_active_zones` | int | maps live `zones_in_active_obs` |
| `nearest_zone` | object \| null | `{ob_timestamp, direction, distance_atr}` or null |
| `prox_cap_atr` | number | cap in force this bar |
| `bt_conditions` | array[str] | zero or more condition codes raised on this bar |

`outcome` enum: `NO_ZONE`, `OUT_OF_RANGE`, `ALERT`, `RE_ARM_WAIT`,
`WARMUP_SKIP`, `NAN_ATR_SKIP`, `DEGENERATE_SKIP`.

---

## 3. `events.jsonl` (one record per causal event)

Every record has `kind` and `emit_ts`. Kind-specific fields:

| kind | key fields |
|------|-----------|
| `ob_seen` | `pair`, `ob_timestamp`, `bos_timestamp`, `direction`, `proximal`, `distal` |
| `alert` | `pair`, `alert_ts`, `ob_timestamp`, `bos_timestamp`, `direction`, `alert_seq`, `trend`, `trend_alignment` |
| `alert_suppressed_dedup` | `pair`, `ob_timestamp`, `direction` |
| `alert_lookahead_blocked` | `pair`, `h1_ts`, `bos_ts`, `ob_ts` |
| `ob_mitigated` | `pair`, `ob_timestamp`, `reason`, `ts` |
| `ob_aged_out` | `pair`, `ob_timestamp`, `age_days` |
| `re_arm` | `pair`, `ob_timestamp`, `ts` |
| `trade_fill` | `pair`, `alert_ts`, `fill_ts`, `entry`, `entry_zone` |
| `trade_exit` | `pair`, `alert_ts`, `exit_ts`, `exit_reason`, `r_realised`, `pnl_usd` |

Causality chain fields (`ob_timestamp`, `bos_timestamp`, `alert_ts`,
`fill_ts`, `exit_ts`) are carried wherever applicable, tz-aware.

`r_if_exit_tp1` / `r_if_exit_tp2` may appear but are tagged
`hypothetical: true` and are excluded from every aggregate by the health layer
(gate G6).

---

## 4. `run_health.json` (written by gates.finalize)

| field | type | notes |
|-------|------|-------|
| `run_id` | str | |
| `overall` | enum | `PASS` \| `FAIL` |
| `exit_code` | int | 0 on PASS, 1 on any FAIL gate |
| `warnings_present` | bool | true if any WARN condition/gate fired |
| `condition_counts` | object | code -> count |
| `event_counts` | object | kind -> count |
| `outcome_counts` | object | outcome -> count |
| `gates` | array | each `{id, description, threshold, observed, verdict}` |
| `content_hash` | str | hash of scan_log + events (determinism stamp, G7) |
| `headline_pnl_usd` | number | recomputed from `r_realised` only (G1/G6) |
