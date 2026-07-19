"""TRUTH LEDGER GATE (TRUTH_FIXES_SPEC.md T3).

Every trades.csv column must have a row in TRUTH_LEDGER.md (source file:line,
when stamped, population, status). This test makes the CLAUDE.md rule
mechanical: a new column added to the CSV writer without a ledger entry turns
CI red instead of shipping unaudited.

Rides `front_cols` in backtest/h1_only_reporting.py — the writer's own
canonical column order. Columns outside front_cols land via the `rest`
catch-all; those are governed by the alert-time-view classification comment in
h1_only_simulator (accepted residual, documented in the ledger).
"""

import csv
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# Columns that were VOIDED during the corrupted era and un-voided by the
# 2026-07-10 audit (Batch 0) once the 07-09 canonical baseline proved they carry
# real values. This guard makes "un-voided" mechanical: if the canonical CSV ever
# regresses these back to all-null or a single constant, CI goes red instead of a
# stale VOID silently becoming a false "verified (population)". Detection-sourced
# break_* columns are deliberately NOT here — their VALUES are stale until the
# post-gate-removal fresh run (Batch 1), so a variety check would be meaningless.
_EX_CORRUPTED_COLS = [
    "fvg_pts",
    "freshness_pts",
    "score",
    "bos_verdict",
    "confluences_present",
]
# setup_badge is intentionally sparse (blank on non-badge rows) — assert only that
# SOME rows carry a badge, not that most do.
_SPARSE_EX_CORRUPTED_COLS = ["setup_badge", "setup_badge_kind"]


def _canonical_csv() -> Path:
    """Resolve the ONE canonical trades.csv from CANONICAL.md (never glob)."""
    doc = (_ROOT / "backtest" / "results" / "CANONICAL.md").read_text(encoding="utf-8")
    m = re.search(r"^\s{2,}(backtest/results/\S+/trades\.csv)\s*$", doc, re.M)
    assert m, "canonical trades.csv path not found in CANONICAL.md — format changed"
    p = _ROOT / m.group(1)
    assert p.exists(), f"canonical CSV named in CANONICAL.md does not exist: {p}"
    return p


def _build_row_line_map():
    """Map each trades.csv column -> the line in _build_row's return dict that emits it."""
    src = (_ROOT / "backtest" / "h1_only_simulator.py").read_text(encoding="utf-8").splitlines()
    def_i = next(i for i, l in enumerate(src) if l.startswith("def _build_row"))
    in_ret = False
    out = {}
    for i, line in enumerate(src[def_i:], start=def_i + 1):
        s = line.strip()
        if s.startswith("return {"):
            in_ret = True
            continue
        if in_ret:
            m = re.match(r'"([a-z0-9_]+)":', s)
            if m:
                out[m.group(1)] = i
            if s == "}":
                break
    return out


def test_row_build_ledger_line_refs_point_at_the_column():
    """Every row-build ledger `src` line-ref must actually emit that column.

    Kills the drift bug class the 2026-07-10 audit found: the _build_row signature
    grew, every column's emit line shifted ~+300, and the ledger's `:NNNN` refs all
    pointed at the wrong (or blank) line. This asserts each row-build column whose
    src cell carries a `:NNNN` has that exact line define `"<col>":` in the code.
    """
    ledger = (_ROOT / "TRUTH_LEDGER.md").read_text(encoding="utf-8").splitlines()
    src = (_ROOT / "backtest" / "h1_only_simulator.py").read_text(encoding="utf-8").splitlines()
    real = _build_row_line_map()

    wrong = []
    for line in ledger:
        if not (line.startswith("| ") and line.count("|") >= 5):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        col, srccell = cells[0], cells[1]
        # multi-column rows like "tp1 / tp2" name the first key; only single-name
        # rows whose name is an actual emitted column are checked here.
        if col not in real:
            continue
        m = re.search(r":(\d+)", srccell)
        if not m:
            continue  # prose-only ref (no line number) — not line-ref-guarded
        cited = int(m.group(1))
        code_line = src[cited - 1] if 0 < cited <= len(src) else ""
        if f'"{col}":' not in code_line:
            wrong.append((col, cited, real[col]))

    assert not wrong, (
        "TRUTH_LEDGER row-build line-refs are stale (col, cited_line, real_line):\n"
        + "\n".join(f"  {c}: ledger says :{cit}, code emits at :{rl}" for c, cit, rl in wrong)
    )


def _def_name_at(src_lines, lineno):
    """Return the def/class name on `lineno` (1-indexed), or None if not a def line."""
    if not (0 < lineno <= len(src_lines)):
        return None
    m = re.match(r"\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", src_lines[lineno - 1])
    return m.group(1) if m else None


def _def_line_map(rel_path):
    """Map def-name -> its (first) definition line for a source file."""
    src = (_ROOT / rel_path).read_text(encoding="utf-8").splitlines()
    out = {}
    for i, line in enumerate(src, start=1):
        m = re.match(r"\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if m and m.group(1) not in out:
            out[m.group(1)] = i
    return out, src


# Files whose Insights-section ledger rows carry `def`-anchored line-refs.
_INSIGHT_SRC_FILES = ("insights.py", "h1_only_reporting.py")


def test_insight_ledger_line_refs_point_at_their_function():
    """Every Insights-section ledger ref `<file>.py:NNNN` that names a function must
    cite that function's real def line.

    Same drift class as the row-build guard, one layer up: insights.py /
    h1_only_reporting.py grew, every `def` shifted a few lines, and the Insights
    table's `:NNNN` refs pointed just above/below the real def. For each ref whose
    src cell also spells a function name that exists in that file, assert the cited
    line is exactly that function's def. Refs that don't name a resolvable function
    (prose anchors, data-line refs) are skipped — this guards the def-anchored ones,
    which are the ones that silently rot.
    """
    ledger = (_ROOT / "TRUTH_LEDGER.md").read_text(encoding="utf-8").splitlines()

    # Only scan rows inside the Insights section (has its own header) so row-build
    # refs (already guarded above) and the layer-map prose don't double-count.
    in_section = False
    defmaps = {f: _def_line_map(f"backtest/{f}") for f in _INSIGHT_SRC_FILES}

    wrong = []
    checked = 0
    for line in ledger:
        if line.startswith("## Insights"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section:
            continue
        if not (line.startswith("| ") and line.count("|") >= 3):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        namecell = cells[0] if len(cells) > 0 else ""
        srccell = cells[1] if len(cells) > 1 else ""
        for fname in _INSIGHT_SRC_FILES:
            defmap, src = defmaps[fname]
            # A ref is DEF-ANCHORED (and therefore guardable) only when it explicitly
            # names its function, one of two ways:
            #   1. parenthetical right after the ref:  "<file>:NNNN (fnname)"
            #   2. the ref is a "<file>:NNNN..." with NO parenthetical, and the ROW's
            #      name cell (col 0) is itself a function in this file.
            # Bare comment/section anchors like ":1411 (reference tabs §10)" name no
            # real def and are correctly skipped.
            for m in re.finditer(re.escape(fname) + r":(\d+)(?:-\d+)?\s*(?:\(([^)]*)\))?", srccell):
                cited = int(m.group(1))
                paren = m.group(2) or ""
                # function this specific ref claims to point at:
                paren_fns = [fn for fn in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", paren) if fn in defmap]
                if paren_fns:
                    named = paren_fns
                else:
                    # only fall back to the name cell if the src cell names NO def at
                    # all (single-function rows like ob_freshness_comparison:533-559).
                    src_fns = [fn for fn in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", srccell) if fn in defmap]
                    if src_fns:
                        continue  # some other ref in this cell owns the name; skip
                    named = [fn for fn in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", namecell) if fn in defmap]
                if not named:
                    continue  # not a def-anchored ref — skip (prose/data line)
                checked += 1
                if not any(defmap[fn] == cited for fn in named):
                    wrong.append((fname, cited, _def_name_at(src, cited), named,
                                  {fn: defmap[fn] for fn in named}))

    assert checked > 0, "insight line-ref guard scanned nothing — Insights section moved?"
    assert not wrong, (
        "TRUTH_LEDGER Insights line-refs are stale — cited line is not the named "
        "function's def:\n" + "\n".join(
            f"  {f}:{cit} -> def there is {at!r}; row names {names}, real defs {real}"
            for f, cit, at, names, real in wrong)
    )


def test_baseline_ex_corrupted_columns_are_real():
    """The un-voided columns must be populated AND varied in the canonical run.

    Failure mode this kills: a future canonical run drops one of these columns to
    all-null or a single constant, but the ledger still reads 'verified
    (population)' — an insight built on it would look trustworthy while being dead.
    """
    path = _canonical_csv()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for col in _EX_CORRUPTED_COLS + _SPARSE_EX_CORRUPTED_COLS:
            assert col in header, f"un-voided column '{col}' missing from canonical CSV header"

        seen = {c: set() for c in _EX_CORRUPTED_COLS}
        sparse_nonblank = {c: 0 for c in _SPARSE_EX_CORRUPTED_COLS}
        rows = 0
        for row in reader:
            rows += 1
            for c in _EX_CORRUPTED_COLS:
                v = row[c]
                if v not in ("", "None"):
                    seen[c].add(v)
            for c in _SPARSE_EX_CORRUPTED_COLS:
                if row[c] not in ("", "None"):
                    sparse_nonblank[c] += 1

    assert rows > 1000, f"canonical CSV suspiciously small ({rows} rows)"
    for c in _EX_CORRUPTED_COLS:
        assert len(seen[c]) >= 2, (
            f"un-voided column '{c}' has <2 distinct non-null values in the "
            f"canonical baseline ({sorted(seen[c])!r}) — corruption/regression; "
            "its ledger status must NOT read 'verified (population)'"
        )
    for c in _SPARSE_EX_CORRUPTED_COLS:
        assert sparse_nonblank[c] > 0, (
            f"sparse un-voided column '{c}' is entirely blank in the canonical "
            "baseline — badge insight is dead, re-void it"
        )


# ── AREA C — derivation recompute (Deep Value Pass, 2026-07-10) ─────────────
# Every column that is a PURE function of other (already-frozen) csv columns is
# recomputed here from its inputs, using the EXACT live formula (file:line in
# each rule), and asserted to reproduce the CSV value for the FULL canonical
# population. Batch 2 did this for only 2 columns; this does it for all 15 that
# are CSV-recomputable. Columns needing row-build internals not in the CSV
# (frozen DR object -> pd_zone/pd_pct, df H1 index -> ob_age_h1_bars/
# bars_break_to_pullback, exit-engine TP replay -> r_if_exit_tp1/2) are NOT here
# — their FORMULAS are unit-tested on synthetic OBs in
# backtest/test_h1_only.py::test_edge_lab_columns (drives live _build_row);
# population recompute needs a re-run and is out of this pass.
#
# sl_distance_atr is special: it divides by the FULL-precision _h1_atr
# (h1_only_simulator.py:1407) but the CSV only stores atr_at_ob rounded to 6dp
# (:1396), so a bit-exact recompute is impossible near a rounding boundary. We
# assert the CSV value is REACHABLE by some true atr inside the stored 6dp
# round-band — proving the formula is right, not that the float round-trips.

import pandas as pd  # noqa: E402

_NY_TZ = "America/New_York"
_EXCLUDE_REASONS = {"never_filled", "timeout", "window_end"}


def _c_utc(v):
    if v in (None, ""):
        return None
    try:
        ts = pd.Timestamp(v)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    except Exception:
        return None


def _c_hour_ny(v):
    ts = _c_utc(v)
    return None if ts is None else int(ts.tz_convert(_NY_TZ).hour)


def _c_session_from_ny(h):
    if h is None:
        return "unknown"
    if 2 <= h < 8:
        return "London"
    if 8 <= h < 16:
        return "NY"
    if h >= 19 or h < 2:
        return "Asia"
    return "Other"


def _c_fill_session(fill_ts, alert_ts):
    h = _c_hour_ny(fill_ts) if fill_ts not in (None, "") else None
    if h is None:
        h = _c_hour_ny(alert_ts)
    return _c_session_from_ny(h) if h is not None else "unknown"


def _c_bias(r):
    return "LONG" if r["direction"] == "bullish" else "SHORT"


def _c_pd_alignment(r):
    bias = _c_bias(r)
    z = r["pd_zone"]
    if z in ("", "unknown"):
        return "unknown"
    if bias == "LONG":
        return "aligned" if z == "discount" else "counter"
    return "aligned" if z == "premium" else "counter"


def _c_float(v):
    return float(v) if v not in (None, "") else None


# col -> (fn(row) -> expected csv string). SKIP sentinel for edge-only rows.
_SKIP = object()


def _area_c_rules():
    def bias(r):
        return _c_bias(r)

    def alert_utc_hour(r):
        ts = _c_utc(r["alert_ts"])
        return _SKIP if ts is None else str(int(ts.hour))

    def session(r):                       # :1555 _fill_session(alert, alert)
        return _c_fill_session(r["alert_ts"], r["alert_ts"])

    def ob_session(r):                    # :1613 / :123-127
        h = _c_hour_ny(r["ob_timestamp"]) if r["ob_timestamp"] not in (None, "") else None
        return _c_session_from_ny(h) if h is not None else "unknown"

    def fill_session(r):                  # :1614
        return _c_fill_session(r["fill_ts"], r["alert_ts"])

    def pd_alignment(r):                  # :241-258
        return _c_pd_alignment(r)

    def event(r):                         # :276-288
        tag = r["bos_tag"] or "BOS"
        tier = r["bos_tier"] or "Major"
        return "Confirmation BOS" if (tag == "BOS" and tier == "Confirm") else f"{tier} {tag}"

    def reversed_from_extreme(r):         # :1346-1350
        rp = r["reversal_pct"]
        if "CHoCH" not in str(r["bos_tag"]) or rp in (None, ""):
            return ""
        return str(bool(float(rp) >= 1.0))

    def trend_pd_agree(r):                # :1424-1432
        h1t = r["h1_trend"] or None
        pda = _c_pd_alignment(r)
        if h1t in (None, "") or pda == "unknown":
            return ""
        wt = ((r["direction"] == "bullish" and h1t == "bullish")
              or (r["direction"] == "bearish" and h1t == "bearish"))
        return str(bool(wt and pda == "aligned"))

    def r_capture_ratio(r):               # :1416-1417
        mfe = _c_float(r["mfe_r"]); rr = _c_float(r["r_realised"])
        if mfe is None or mfe <= 0 or rr is None:
            return ""
        return str(round(rr / mfe, 3))

    def ob_to_fill_hours(r):              # :1068-1075
        ob = _c_utc(r["ob_timestamp"]); fl = _c_utc(r["fill_ts"])
        if ob is None or fl is None:
            return ""
        return str(round((fl - ob).total_seconds() / 3600.0, 2))

    def killzone_alignment(r):            # :198-217
        if r["fill_ts"] in (None, ""):
            return "never_filled"
        ob_kz = r["ob_in_killzone"] == "True"
        fl_kz = r["fill_in_killzone"] == "True"
        if ob_kz and fl_kz:
            return "Both"
        if ob_kz:
            return "OB only"
        if fl_kz:
            return "Fill only"
        return "Neither"

    def headline_exclusion(r):            # :142-173
        er = r["exit_reason"]
        if er in _EXCLUDE_REASONS:
            return f"unresolved:{er}" if er in ("timeout", "window_end") else str(er)
        if r["ist_blocked"] == "True":
            return "ist_blocked"
        if r["weekend_blocked"] == "True":
            return "weekend_blocked"
        return ""

    def eligible_for_headline(r):         # :1230
        return str(headline_exclusion(r) == "")

    return {
        "bias": bias, "alert_utc_hour": alert_utc_hour, "session": session,
        "ob_session": ob_session, "fill_session": fill_session,
        "pd_alignment": pd_alignment, "event": event,
        "reversed_from_extreme": reversed_from_extreme,
        "trend_pd_agree": trend_pd_agree, "r_capture_ratio": r_capture_ratio,
        "ob_to_fill_hours": ob_to_fill_hours,
        "killzone_alignment": killzone_alignment,
        "headline_exclusion": headline_exclusion,
        "eligible_for_headline": eligible_for_headline,
    }


def test_area_c_derivations_recompute_on_canonical_csv():
    """Recompute every CSV-derivable column from its inputs; 0 mismatches over
    the full canonical population. Bites if any live derivation formula drifts
    from what produced the file (or the file is rebuilt with a broken formula)."""
    path = _canonical_csv()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 1000, f"canonical CSV suspiciously small ({len(rows)})"

    rules = _area_c_rules()
    failures = {}
    for col, fn in rules.items():
        bad = []
        for i, r in enumerate(rows):
            exp = fn(r)
            if exp is _SKIP:
                continue
            if str(exp) != str(r.get(col, "")):
                bad.append((i, r.get(col), exp))
                if len(bad) >= 5:
                    break
        if bad:
            failures[col] = bad
    assert not failures, f"Area C derivation mismatches: {failures}"


def test_area_c_sl_distance_atr_reachable_via_atr_rounding_band():
    """sl_distance_atr = |entry - sl_initial| / _h1_atr, but the CSV stores
    atr_at_ob rounded to 6dp — so recomputing from the CSV cannot be bit-exact
    near a rounding boundary. Assert every value is reachable by SOME true atr
    inside the stored 6dp round-band (formula correct, precision-limited).
    A truly wrong value (e.g. dividing by the wrong quantity) would land far
    outside that ~1-ULP band and be flagged UNREACHABLE."""
    import numpy as np
    path = _canonical_csv()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    unreachable = []
    for i, r in enumerate(rows):
        entry = _c_float(r["entry"]); sl = _c_float(r["sl_initial"])
        atr = _c_float(r["atr_at_ob"]); csvv = r["sl_distance_atr"]
        if not atr or entry is None or sl is None or csvv in ("", None):
            continue
        if str(round(abs(entry - sl) / atr, 3)) == str(csvv):
            continue
        num = abs(entry - sl)
        grid = np.linspace(atr - 0.5e-6, atr + 0.5e-6, 2001)
        if not any(str(round(num / a, 3)) == str(csvv) for a in grid):
            unreachable.append((i, csvv, atr))
            if len(unreachable) >= 5:
                break
    assert not unreachable, (
        "sl_distance_atr values NOT reachable within the 6dp atr round-band "
        f"(would be a real formula bug): {unreachable}")


# ── AREA B — ATR normalisation (Deep Value Pass, 2026-07-10) ────────────────
# Every *_atr column divides by the ONE frozen ob['h1_atr'] (stored as atr_at_ob
# at 6dp). sl_distance_atr is the only one whose numerator is fully in the CSV
# (|entry - sl_initial|), so it is the CANONICAL-DATA anchor for "the denominator
# is atr_at_ob and nothing else". The live-code proof that all SIX *_atr columns
# share that denominator (scaling test + source tripwire) lives in
# backtest/test_h1_only.py::test_area_b_*. This test proves the anchor on the file
# AND bites a wrong denominator.


def _reachable_via_atr_band(num, atr, csvv, half_width=0.5e-6, steps=201):
    """True if round(num / a, 3) == csvv for SOME true atr in the 6dp round-band
    [atr - 0.5ulp, atr + 0.5ulp]. The CSV stores atr_at_ob at 6dp but the live
    divide uses full-precision _h1_atr, so an exact match is not guaranteed near a
    rounding boundary — reachability inside the band is the correct proof."""
    import numpy as np
    if str(round(num / atr, 3)) == str(csvv):
        return True
    grid = np.linspace(atr - half_width, atr + half_width, steps)
    return any(str(round(num / a, 3)) == str(csvv) for a in grid)


def test_area_b_sl_distance_atr_divides_atr_at_ob_on_canonical():
    """AREA B anchor: on the canonical file, sl_distance_atr must be reproducible
    as |entry - sl_initial| / atr_at_ob (within the 6dp round-band) — proving it
    divides by atr_at_ob. Bite: dividing the SAME numerator by a WRONG atr
    (atr_at_ob * 1.5) must make the CSV value unreachable for essentially every
    row, so a real denominator swap turns this red.

    Sampled (seeded) for speed — the grid search is O(rows * steps); a 600-row
    seeded sample is a stable, representative slice per CLAUDE.md smart-sampling.
    """
    import random
    path = _canonical_csv()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    usable = [r for r in rows
              if r["sl_distance_atr"] not in ("", None)
              and r["atr_at_ob"] not in ("", None)
              and r["entry"] not in ("", None)
              and r["sl_initial"] not in ("", None)]
    assert len(usable) > 1000, f"too few usable sl_distance_atr rows ({len(usable)})"
    random.seed(0)
    sample = random.sample(usable, 600)

    correct_unreachable = []
    wrong_reachable = 0
    for i, r in enumerate(sample):
        num = abs(_c_float(r["entry"]) - _c_float(r["sl_initial"]))
        atr = _c_float(r["atr_at_ob"])
        csvv = r["sl_distance_atr"]
        # CORRECT denominator: must be reachable within the atr round-band.
        if not _reachable_via_atr_band(num, atr, csvv):
            correct_unreachable.append((i, csvv, atr))
        # WRONG denominator: atr * 1.5 must NOT reach the CSV value (bite).
        if _reachable_via_atr_band(num, atr * 1.5, csvv):
            wrong_reachable += 1

    assert not correct_unreachable, (
        "sl_distance_atr NOT reachable dividing by atr_at_ob (denominator drifted "
        f"from ob['h1_atr']): {correct_unreachable[:5]}")
    # A correct-denominator file makes the wrong (1.5x) denominator essentially
    # always unreachable; allow a tiny tail for small-value 3dp rounding collisions.
    assert wrong_reachable <= 6, (
        f"{wrong_reachable}/600 rows still reachable under a 1.5x wrong denominator "
        "— the anchor is not actually pinning atr_at_ob as the divisor (bite too weak)")


def test_every_csv_column_has_a_ledger_row():
    ledger = (_ROOT / "TRUTH_LEDGER.md").read_text(encoding="utf-8")
    src = (_ROOT / "backtest" / "h1_only_reporting.py").read_text(encoding="utf-8")

    m = re.search(r"front_cols = \[(.*?)\]", src, re.S)
    assert m, "front_cols list not found in h1_only_reporting.py — writer moved; update this test"

    cols = re.findall(r'"([a-z0-9_]+)"', m.group(1))
    assert len(cols) > 40, f"front_cols parse looks wrong (only {len(cols)} names)"

    missing = [c for c in cols if c not in ledger]
    assert not missing, (
        "trades.csv columns missing a TRUTH_LEDGER.md row: "
        f"{missing}. Rule (CLAUDE.md, Logging): no new metric ships without a "
        "ledger entry + a class-killing guard."
    )


if __name__ == "__main__":
    test_every_csv_column_has_a_ledger_row()
    test_row_build_ledger_line_refs_point_at_the_column()
    test_baseline_ex_corrupted_columns_are_real()
    print("truth-ledger gate: OK")
