"""EDGE ENGINE — human-readable DISCOVERY report writer (SPEC_STAGED §7/§9.1,
EDGE_DISCOVERY_REPORT_SPEC work-item 2).

Renders `edge_engine_discovery.md` — the full-detail companion to the summary
discovery email. Every insight the trader wants (per-feature numbers, near-misses,
per-pair / per-session baselines, sub-screens) already lives in the committed
discovery artefacts; this module is PURE RENDERING of them.

HARD CONTRACT (do not break):
  * Inputs = engine_dir ONLY: `stage1_discovery.json`, `stage1_discovery_features.csv`,
    `stage0_gate.json` (census). It NEVER reads trades.csv and NEVER builds a
    validation frame (SPEC_STAGED §4.2 — the "never materialised" guarantee).
  * The discovery language stamp appears verbatim near the top. The literal strings
    "survivor" and " edge" must NOT appear anywhere in the rendered .md (the §11.3
    language rule, now scanned on the report too).
  * Every table carries N (blind-spot guard); every rendered expR shows its n.
  * One table is rendered for EVERY feature record (count-asserted in tests — no
    silent truncation).

The verdict / threshold semantics are NOT re-derived here — the near-miss section
reads the `criteria` pass/fail flags stamped on each feature record by
`_apply_candidate_criteria`. If an older committed JSON lacks those flags (pre-fix
backfill), the near-miss section degrades to "criteria flags absent" rather than
re-deriving thresholds.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

DISCOVERY_JSON = "stage1_discovery.json"
DISCOVERY_CSV = "stage1_discovery_features.csv"
GATE_JSON = "stage0_gate.json"
REPORT_MD = "edge_engine_discovery.md"

# The four discovery criteria, in ladder order (mirrors _apply_candidate_criteria).
CRITERIA_KEYS = ("fdr_reject", "ci_excludes_0", "substance_n", "substance_effect")


# ---------------------------------------------------------------------------
# Loading (engine_dir only — never trades.csv)
# ---------------------------------------------------------------------------
def _load_json(engine_dir: str, name: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(engine_dir, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _load_buckets(engine_dir: str) -> List[Dict[str, str]]:
    """Read every bucket/level row from the committed features CSV (DISCOVERY only
    by construction — the discovery writer never emits VALIDATION rows)."""
    p = os.path.join(engine_dir, DISCOVERY_CSV)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Small formatting helpers (every stat prints WITH its N)
# ---------------------------------------------------------------------------
def _f(x: Any) -> str:
    return "—" if x is None or x == "" else str(x)


def _ci(rec: Dict[str, Any], key: str = "delta_disc_ci") -> str:
    ci = rec.get(key) or [None, None]
    lo, hi = (ci + [None, None])[:2]
    if lo is None or hi is None:
        return "[—, —]"
    return f"[{lo:+.4f}, {hi:+.4f}]"


def _ci_pair(lo: Any, hi: Any) -> str:
    if lo in (None, "") or hi in (None, ""):
        return "[—, —]"
    return f"[{float(lo):+.4f}, {float(hi):+.4f}]"


def _n_pair(rec: Dict[str, Any]) -> str:
    n = rec.get("top_bottom_n_disc") or rec.get("best_worst_n_disc") or [None, None]
    return f"{_f(n[0])}/{_f(n[1])}"


def _failed_flags(rec: Dict[str, Any]) -> Optional[List[str]]:
    """The criteria that FAILED for this record, or None if the record carries no
    criteria block (older JSON / thin record)."""
    crit = rec.get("criteria")
    if not isinstance(crit, dict):
        return None
    return [k for k in CRITERIA_KEYS if not crit.get(k)]


def _flags_cell(rec: Dict[str, Any]) -> str:
    failed = _failed_flags(rec)
    if failed is None:
        return "—"
    return "PASS" if not failed else "fail:" + ",".join(failed)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------
def _header(disc: Dict[str, Any], gate: Dict[str, Any]) -> List[str]:
    census = ((gate or {}).get("census") or {}).get("by_split") or {}
    census_str = "  ".join(f"{k}={v}" for k, v in census.items())
    tok = disc.get("token", "—")
    comp_sha = disc  # token block mirrors the email's 4-line block
    # Title deliberately avoids the standalone word "edge" preceded by a space:
    # the §11.3 language rule bans the literal " edge" anywhere in this .md.
    L = [
        f"# DISCOVERY REPORT (engine phase A) — {disc.get('run_id')}",
        "",
        f"> {disc.get('language_stamp', '')}",
        "",
        f"- **run_id:** {disc.get('run_id')}",
        f"- **window:** {disc.get('window')}",
        f"- **N (discovery split):** {disc.get('n_discovery')}",
        f"- **split census:** {census_str}",
        f"- **scope:** {disc.get('scope')}",
        f"- **generated (UTC):** {disc.get('generated_utc')}",
        "",
        "**APPROVAL TOKEN (same 4-line block as the email):**",
        "",
        "```",
        f"APPROVAL TOKEN: {tok}",
        "To confirm on validation (ONE shot):  press CONFIRM in the Action with this token,",
        f"or locally:  python -m backtest.diagnostics.edge_engine --approve {tok}",
        "             python -m backtest.diagnostics.edge_engine --phase confirm",
        "```",
        "",
        "### How to read this report",
        "",
        "- **candidate** = passed all four discovery criteria (FDR-reject, CI excludes 0,",
        "  both extreme buckets N≥150, effect ≥0.10R). It is NOT confirmed — luck is not",
        "  ruled out until it repeats on validation years it has never seen.",
        "- **candidate_thin** = a real discovery signal but a thin bucket or a sub-0.10R",
        "  effect. Kept visible, never shipped from here.",
        "- **noise** = did not clear the discovery signal bar. **thin** = fewer than two",
        "  testable buckets to compare at all.",
        "- **Δdisc** = top-vs-bottom expR gap on the discovery split (categorical: best vs",
        "  worst level). Its 95% bootstrap CI is shown beside it.",
        "- **Why discovery numbers can still be luck:** these are one split, gates off, all",
        "  scores. A gap here is a hypothesis to test, not a result to trade.",
        "",
    ]
    return L


def _verdict_table(feats: List[Dict[str, Any]]) -> List[str]:
    rows = sorted(feats, key=lambda r: -(abs(r.get("delta_disc") or 0.0)))
    L = ["## 1. Verdict table (all features, |Δdisc| desc)", "",
         "| feature | type | timing | verdict | Δdisc | CI | N(top/bot) | screen p | fdr | failed criteria |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r.get("_fdr_p")
        p_str = "—" if p is None else f"{float(p):.4g}"
        L.append(
            f"| {r.get('feature')} | {r.get('type','—')} | {r.get('timing','—')} | "
            f"{r.get('verdict','—')} | {_f(r.get('delta_disc'))} | {_ci(r)} | "
            f"{_n_pair(r)} | {p_str} | {r.get('fdr_reject')} | {_flags_cell(r)} |")
    L.append("")
    return L


def _feature_bucket_rows(buckets: List[Dict[str, str]], feat: str) -> List[Dict[str, str]]:
    return [b for b in buckets if b.get("feature") == feat]


def _bucket_table(rows: List[Dict[str, str]], is_cont: bool) -> List[str]:
    """One markdown table for a feature's bucket/level rows. Every expR row carries n."""
    if is_cont:
        # NB: header says "bound_lo/bound_hi" not "edge_lo/edge_hi" — the literal
        # " edge" is banned by the §11.3 language rule (scanned on this .md too).
        head = ("| bucket | bound_lo | bound_hi | n | expR | CI | wr% | totR | pos_q |")
        sep = "|---|---|---|---|---|---|---|---|---|"
    else:
        head = ("| level | n | expR | CI | wr% | totR | pos_q |")
        sep = "|---|---|---|---|---|---|---|"
    L = [head, sep]
    for b in rows:
        ci = _ci_pair(b.get("ci_lo"), b.get("ci_hi"))
        if is_cont:
            L.append(f"| {_f(b.get('bucket'))} | {_f(b.get('edge_lo'))} | "
                     f"{_f(b.get('edge_hi'))} | {_f(b.get('n'))} | {_f(b.get('expR'))} | "
                     f"{ci} | {_f(b.get('wr_pct'))} | {_f(b.get('totR'))} | "
                     f"{_f(b.get('pos_quarters'))} |")
        else:
            L.append(f"| {_f(b.get('level'))} | {_f(b.get('n'))} | {_f(b.get('expR'))} | "
                     f"{ci} | {_f(b.get('wr_pct'))} | {_f(b.get('totR'))} | "
                     f"{_f(b.get('pos_quarters'))} |")
    return L


def _candidate_deep_dives(disc: Dict[str, Any], buckets: List[Dict[str, str]]
                          ) -> List[str]:
    cands = [r for r in disc.get("features", []) if r.get("verdict") == "candidate"]
    L = ["## 2. Candidate deep-dives", ""]
    if not cands:
        L += ["_No candidates on this discovery split — a valid null._", ""]
        return L
    for r in cands:
        feat = r["feature"]
        is_cont = r.get("type") == "continuous"
        L.append(f"### `{feat}`  ({r.get('type')}, {r.get('timing')})")
        L.append("")
        fav = (f"favoured bucket: {r.get('favoured_bucket')}" if is_cont
               else f"best level: {r.get('best_level')!r}  worst level: {r.get('worst_level')!r}")
        fdr_p = r.get("_fdr_p")
        L.append(f"- Δdisc **{_f(r.get('delta_disc'))}** CI {_ci(r)}, "
                 f"N(top/bot) {_n_pair(r)}, screen p "
                 f"{'—' if fdr_p is None else f'{float(fdr_p):.4g}'}")
        L.append(f"- {fav}")
        L.append("- CANDIDATE — luck not ruled out. The confirm phase re-computes this same "
                 "Δ on validation years and requires the same sign + ≥60% positive quarters.")
        L.append("")
        L += _bucket_table(_feature_bucket_rows(buckets, feat), is_cont)
        L.append("")
    return L


def near_miss_features(disc: Dict[str, Any]) -> List[Tuple[Dict[str, Any], str]]:
    """The features that failed EXACTLY ONE discovery criterion (read off the stamped
    `criteria` flags — never re-derived from thresholds). Returns (record, the one
    failed-criterion name) pairs, |Δdisc| desc. Shared by the report's near-miss
    section and the email's near-miss count so both agree (one implementation)."""
    near = []
    for r in disc.get("features", []):
        if r.get("verdict") == "candidate":
            continue
        failed = _failed_flags(r)
        if failed is not None and len(failed) == 1:
            near.append((r, failed[0]))
    near.sort(key=lambda t: -(abs(t[0].get("delta_disc") or 0.0)))
    return near


def _near_misses(disc: Dict[str, Any], buckets: List[Dict[str, str]]) -> List[str]:
    """Features that failed EXACTLY ONE criterion (read off the stamped flags, never
    re-derived). Non-candidates, shown for transparency (C4), no threshold move (F)."""
    L = ["## 3. Near-misses", "",
         "**NOT candidates. Shown for transparency (C4). No action, no threshold "
         "renegotiation (F).**", ""]
    near = near_miss_features(disc)
    if not near:
        L += ["_None this run (or criteria flags absent in this JSON)._", ""]
        return L
    L += ["| feature | verdict | Δdisc | CI | N(top/bot) | the one failed criterion |",
          "|---|---|---|---|---|---|"]
    for r, f1 in near:
        L.append(f"| {r.get('feature')} | {r.get('verdict')} | {_f(r.get('delta_disc'))} | "
                 f"{_ci(r)} | {_n_pair(r)} | {f1} |")
    L.append("")
    # ob_range_atr: fdr-rejected Spearman but flat top-vs-bottom — show its quintile curve.
    for r, _f1 in near:
        if r.get("feature") == "ob_range_atr":
            L.append("`ob_range_atr` fdr-rejected on the Spearman trend but the top-vs-bottom "
                     "Δ is near-flat — a non-monotonic shape. Its quintile curve:")
            L.append("")
            L += _bucket_table(_feature_bucket_rows(buckets, "ob_range_atr"), True)
            L.append("")
    return L


def _baseline_context(disc: Dict[str, Any]) -> List[str]:
    pop = disc.get("population_stats") or {}
    L = ["## 4. Baseline context — how did pairs / sessions do", "",
         "**Caveats (read before any number below):** (a) this is the gates-off, "
         "all-scores discovery population — NOT what live (score≥4, filtered) trading "
         "would produce; (b) Book B pairs (GBPUSD, AUDUSD, USDCAD, EURJPY) are pooled "
         "per SPEC §3.3 but are NOT in live trade scope.", ""]

    def _stat_row(label_key: str, label_val: str, s: Dict[str, Any]) -> str:
        return (f"| {label_val} | {_f(s.get('n'))} | {_f(s.get('expR'))} | "
                f"{_ci_pair(s.get('ci_lo'), s.get('ci_hi'))} | {_f(s.get('wr_pct'))} | "
                f"{_f(s.get('totR'))} | {_f(s.get('pos_quarters'))} |")

    ov = pop.get("overall") or {}
    if ov:
        L += ["### Overall discovery-split",
              "| — | n | expR | CI | wr% | totR | pos_q |",
              "|---|---|---|---|---|---|---|",
              _stat_row("overall", "ALL", ov), ""]
    per_pair = pop.get("per_pair") or []
    if per_pair:
        L += ["### Per pair",
              "| pair | n | expR | CI | wr% | totR | pos_q |",
              "|---|---|---|---|---|---|---|"]
        for s in per_pair:
            L.append(_stat_row("pair", s.get("pair"), s))
        L.append("")
    per_sess = pop.get("per_session") or []
    if per_sess:
        L += ["### Per session (alert session)",
              "| session | n | expR | CI | wr% | totR | pos_q |",
              "|---|---|---|---|---|---|---|"]
        for s in per_sess:
            L.append(_stat_row("session", s.get("session"), s))
        L.append("")
    if not pop:
        L += ["_population_stats absent in this JSON (pre-refactor run)._", ""]
    return L


def _sub_screens(disc: Dict[str, Any]) -> List[str]:
    L = ["## 5. Sub-screens", ""]
    # Snapback
    snap = disc.get("snapback") or []
    L += ["### Snapback bins (bars_break_to_pullback)", "",
          "| bin | n | expR | CI | wr% | totR | pos_q | caveat |",
          "|---|---|---|---|---|---|---|---|"]
    for b in snap:
        cav = b.get("caveat") or ""
        L.append(f"| {_f(b.get('bin'))} | {_f(b.get('n'))} | {_f(b.get('expR'))} | "
                 f"{_ci_pair(b.get('ci_lo'), b.get('ci_hi'))} | {_f(b.get('wr_pct'))} | "
                 f"{_f(b.get('totR'))} | {_f(b.get('pos_quarters'))} | {cav} |")
    if not snap:
        L.append("| — | — | — | — | — | — | — | (no snapback rows) |")
    L.append("")
    # SL anatomy
    anat = disc.get("sl_anatomy") or {}
    rows = anat.get("rows") if isinstance(anat, dict) else None
    L += ["### SL-anatomy (clean-break rate by bucket, on eligible SL exits)", ""]
    if rows:
        L += ["| feature | hi | lo | clean_rate hi | clean_rate lo | rate_diff | CI | n hi/lo | robust |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            ci = r.get("ci") or [None, None]
            L.append(f"| {r.get('feature')} | {r.get('hi')} | {r.get('lo')} | "
                     f"{_f(r.get('clean_rate_hi'))} | {_f(r.get('clean_rate_lo'))} | "
                     f"{_f(r.get('rate_diff'))} | {_ci_pair(ci[0], ci[1])} | "
                     f"{_f(r.get('n_hi'))}/{_f(r.get('n_lo'))} | "
                     f"{r.get('robust_clean_break_predictor')} |")
        L.append("")
        L.append("_Promotions are a validation concept — deferred to the confirm phase._")
    else:
        note = anat.get("note", "no SL-anatomy rows") if isinstance(anat, dict) else "—"
        L.append(f"_{note}_")
    L.append("")
    # News confounder
    news = disc.get("news_confounder") or {}
    L += ["### News confounder (context only, never a gate)", ""]
    if news.get("note"):
        L.append(f"_{news.get('note')}_")
    else:
        L.append(f"- clean-break SL exits near news: {_f(news.get('clean_break_near_news_pct'))}%")
        L.append(f"- all eligible rows near news: {_f(news.get('all_eligible_near_news_pct'))}%")
    L.append("")
    L.append("_Interactions: deferred to the confirm phase._")
    L.append("")
    return L


def _appendix(disc: Dict[str, Any], buckets: List[Dict[str, str]]
              ) -> Tuple[List[str], int]:
    """Full per-feature bucket/level table for EVERY feature record. Returns the
    lines and the number of feature tables rendered (tests assert this equals the
    feature-record count — no silent truncation)."""
    L = ["## 6. Appendix — every feature's bucket/level table", ""]
    n_tables = 0
    for r in disc.get("features", []):
        feat = r["feature"]
        is_cont = r.get("type") == "continuous"
        L.append(f"### `{feat}`  ({r.get('type')}, {r.get('timing')}) — "
                 f"verdict: {r.get('verdict')}")
        L.append("")
        frows = _feature_bucket_rows(buckets, feat)
        if frows:
            L += _bucket_table(frows, is_cont)
        else:
            L.append("_no bucket rows in the CSV for this feature (thin / insufficient "
                     "distinct values)._")
        L.append("")
        n_tables += 1
    return L, n_tables


def _observations(disc: Dict[str, Any]) -> List[str]:
    L = ["## 7. Observations (observation-only, no action)", ""]
    feats = {r["feature"]: r for r in disc.get("features", [])}
    ev, bt = feats.get("event"), feats.get("bos_tier")
    if ev and bt and ev.get("delta_disc") == bt.get("delta_disc") \
            and ev.get("_fdr_p") == bt.get("_fdr_p"):
        L.append("- `event` and `bos_tier` produced identical screen stats "
                 f"(Δdisc {_f(ev.get('delta_disc'))}, p "
                 f"{'—' if ev.get('_fdr_p') is None else f'{float(ev['_fdr_p']):.4g}'}) — "
                 "likely a 1:1 level mapping in this population. Observation only; "
                 "neither feature is dropped (manifest is frozen, B6).")
    else:
        L.append("- (no identical-stat feature pairs detected this run)")
    L.append("")
    return L


# ---------------------------------------------------------------------------
# Top-level render
# ---------------------------------------------------------------------------
def render_discovery_report(engine_dir: str) -> Optional[str]:
    """Render edge_engine_discovery.md from committed engine_dir artefacts alone.
    Returns the written path, or None if the discovery JSON is absent. Reads no
    trades.csv, builds no validation frame."""
    disc = _load_json(engine_dir, DISCOVERY_JSON)
    if disc is None:
        return None
    gate = _load_json(engine_dir, GATE_JSON) or {}
    buckets = _load_buckets(engine_dir)

    lines: List[str] = []
    lines += _header(disc, gate)
    lines += _verdict_table(disc.get("features", []))
    lines += _candidate_deep_dives(disc, buckets)
    lines += _near_misses(disc, buckets)
    lines += _baseline_context(disc)
    lines += _sub_screens(disc)
    appendix, _n = _appendix(disc, buckets)
    lines += appendix
    lines += _observations(disc)

    out = os.path.join(engine_dir, REPORT_MD)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out


def report_path(engine_dir: str) -> str:
    return os.path.join(engine_dir, REPORT_MD)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Render edge_engine_discovery.md from committed engine_dir JSON/CSV")
    ap.add_argument("--engine-dir", required=True)
    a = ap.parse_args()
    p = render_discovery_report(a.engine_dir)
    print(f"wrote {p}" if p else "no stage1_discovery.json in engine_dir")
