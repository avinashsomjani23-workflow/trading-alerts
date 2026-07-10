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
