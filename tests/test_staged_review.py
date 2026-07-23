"""STAGED HUMAN REVIEW guard (EDGE_ENGINE_SPEC.md §18 / SPEC_STAGED §11).

Kills the bug class "validation spent without a ledger line / stamp", and pins
every hard constraint of the staged-review design:

  1. No-leak         — discovery output has no *_val keys / VALIDATION rows.
  2. Candidate ladder — _apply_candidate_criteria hits each verdict; survivor/
     hypothesis/inverted are impossible outputs.
  3. Language         — discovery report/email carry the stamp, never "survivor"/"edge".
  4. Gate refusal     — verdict scope + no approval => stage1 refuses, writes nothing.
  5. Token binding    — a mutated discovery JSON invalidates an issued token.
  6. Phase A/B agree  — delta_disc identical across discovery and confirm.
  7. Exploratory      — no gate; --phase discovery errors out.
  8. Single-use+ledger — token consumed once; re-run refuses; burn stamps N=2.
  9. Force            — --force does not open the gate.

Run:  python tests/test_staged_review.py   (or via pytest)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from backtest.diagnostics import edge_engine as ee
from backtest.diagnostics import edge_email


# ---------------------------------------------------------------------------
# Synthetic run builder — one trades.csv, two splits, engineered features.
# ---------------------------------------------------------------------------
def _build_trades(seed: int = 42) -> pd.DataFrame:
    """A synthetic eligible EURUSD (book A) population spanning DISCOVERY (2010)
    and VALIDATION (2019), each > MIN_SPLIT_N. `edge_feat` carries a strong
    monotonic effect (a CANDIDATE); `noise_feat` is pure noise; the rest are
    constants filling the manifest so the screens run."""
    rng = np.random.default_rng(seed)
    n_per = 900  # > MIN_SPLIT_N per split
    rows = []
    for split_label, year in (("DISCOVERY", 2010), ("VALIDATION", 2019),
                              ("HOLDOUT", 2023)):
        for i in range(n_per):
            ts = pd.Timestamp(f"{year}-06-01", tz="UTC") + pd.Timedelta(hours=i)
            x = rng.uniform(0, 1)  # the designed feature value
            # strong monotone: high x => higher r (Δ top-vs-bottom well over 0.10R)
            r = (x - 0.5) * 2.0 + rng.normal(0, 0.3)
            rows.append({
                "setup_id": f"{split_label}_{i}",
                "alert_ts": ts.isoformat(),
                "fill_ts": (ts + pd.Timedelta(hours=1)).isoformat(),
                "exit_ts": (ts + pd.Timedelta(hours=5)).isoformat(),
                "pair": "EURUSD",
                "entry_zone": "proximal",
                "eligible_for_headline": True,
                "r_realised": round(float(r), 4),
                "exit_reason": "tp1" if r > 0 else "sl",
                "entry": 1.1000, "sl_initial": 1.0950, "tp1": 1.1100,
                "bias": "LONG", "direction": "bullish",
                # the CANDIDATE feature
                "break_close_atr": round(float(x), 4),
                # a NOISE continuous feature
                "impulse_leg_to_extreme_atr": round(float(rng.uniform(0, 1)), 4),
                # score: gates-off proof needs >= MIN_BELOW_FLOOR_N below floor
                "score": 2 if i < ee.MIN_BELOW_FLOOR_N else 6,
            })
    df = pd.DataFrame(rows)
    # Fill the rest of the manifest with constants so every screen has its column.
    const_cont = {c: 1.0 for c in ee.CONTINUOUS_FEATURES
                  if c not in df.columns}
    const_cat = {c: "x" for c in ee.CATEGORICAL_FEATURES if c not in df.columns}
    for c, v in {**const_cont, **const_cat}.items():
        df[c] = v
    df["pair"] = "EURUSD"
    # SL-anatomy + news columns absent on purpose (disable those sub-screens).
    return df


def _write_run(tmp: str, df: pd.DataFrame) -> str:
    run_dir = os.path.join(tmp, "h1only_20100101_20251231")
    os.makedirs(os.path.join(run_dir, "edge_engine"), exist_ok=True)
    df.to_csv(os.path.join(run_dir, "trades.csv"), index=False)
    return run_dir


def _stub_gate(run_dir: str, scope: str) -> None:
    """Write a minimal stage0_gate.json with the given scope. The staged Stage-1
    functions read only scope / news_usable / sl_anatomy_usable from it; the real
    stage0 baseline self-check needs the frozen MT5 bar cache (absent for a
    synthetic pair), so we stub the gate rather than replay bars."""
    ed = os.path.join(run_dir, "edge_engine")
    ee._write_json(ee._stage_path(ed, 0),
                   {"stage": 0, "pass": True, "scope": scope,
                    "news_usable": False, "sl_anatomy_usable": False,
                    "census": {"by_split": {"DISCOVERY": 900, "VALIDATION": 900,
                                            "HOLDOUT": 900}}})


_FAILS = []


def _ok(m): print(f"  OK:   {m}")
def _bad(m):
    # RAISE, don't just collect: CI runs these via `pytest tests/ -q`, which never
    # calls main(). A print-and-append _bad is invisible to pytest (Deep Value A).
    _FAILS.append(m); print(f"  FAIL: {m}"); raise AssertionError(m)
def _eq(a, b, m):
    (_ok if a == b else _bad)(f"{m}  ({a!r} == {b!r})")
def _true(c, m):
    (_ok if c else _bad)(m)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _step_candidate_ladder():
    """§11.2 — _apply_candidate_criteria hits each verdict; survivor/hypothesis/
    inverted are impossible outputs."""
    # candidate: signal + N + effect.
    cand = {"delta_disc": 0.5, "delta_disc_ci": [0.2, 0.8],
            "top_bottom_n_disc": [300, 300]}
    _eq(ee._apply_candidate_criteria(cand, True), "candidate", "strong => candidate")
    # candidate_thin by effect: signal + N, |Δ| < MIN_EFFECT_R.
    ct = {"delta_disc": 0.05, "delta_disc_ci": [0.01, 0.09],
          "top_bottom_n_disc": [300, 300]}
    _eq(ee._apply_candidate_criteria(ct, True), "candidate_thin", "small effect => candidate_thin")
    # candidate_thin by N: signal but thin buckets.
    ctn = {"delta_disc": 0.5, "delta_disc_ci": [0.2, 0.8],
           "top_bottom_n_disc": [10, 10]}
    _eq(ee._apply_candidate_criteria(ctn, True), "candidate_thin", "thin N + signal => candidate_thin")
    # noise: no FDR reject.
    noise = {"delta_disc": 0.5, "delta_disc_ci": [-0.2, 0.8],
             "top_bottom_n_disc": [300, 300]}
    _eq(ee._apply_candidate_criteria(noise, False), "noise", "no signal => noise")
    # thin: pre-flagged.
    _eq(ee._apply_candidate_criteria({"verdict": "thin"}, True), "thin", "pre-thin => thin")
    _eq(ee._apply_candidate_criteria({"delta_disc": None}, True), "thin", "None Δ => thin")
    # impossibility of the three forbidden strings across a fuzz of inputs.
    rng = np.random.default_rng(1)
    forbidden = {"survivor", "hypothesis", "inverted"}
    for _ in range(400):
        rec = {"delta_disc": float(rng.normal(0, 0.5)),
               "delta_disc_ci": sorted([float(rng.normal(0, 0.3)),
                                        float(rng.normal(0, 0.3))]),
               "top_bottom_n_disc": [int(rng.integers(0, 500)),
                                     int(rng.integers(0, 500))]}
        v = ee._apply_candidate_criteria(rec, bool(rng.integers(0, 2)))
        if v in forbidden:
            _bad(f"forbidden verdict {v} produced"); break
    else:
        _ok("survivor/hypothesis/inverted impossible over 400 fuzzed inputs")


def _dtype_frame(level_a, level_b, dtype: str, n: int = 200, seed: int = 7):
    """Two-level categorical frame with a STRONG expR separation. `level_a` carries
    high r (~+0.5), `level_b` low r (~-0.5). Column dtype is forced to the target
    so the screen sees exactly the native dtype the real trades.csv has."""
    rng = np.random.default_rng(seed)
    rows = []
    for lvl, mu in ((level_a, 0.5), (level_b, -0.5)):
        for i in range(n):
            ts = pd.Timestamp("2010-06-01", tz="UTC") + pd.Timedelta(hours=len(rows))
            rows.append({"feat": lvl, "alert_ts": ts.isoformat(),
                         "r_realised": round(float(rng.normal(mu, 0.3)), 4)})
    df = pd.DataFrame(rows)
    if dtype == "bool":
        df["feat"] = df["feat"].astype(bool)
    elif dtype == "int64":
        df["feat"] = df["feat"].astype("int64")
    elif dtype == "mixed":
        df["feat"] = df["feat"].astype(object)  # bools held in an object column
    else:  # str
        df["feat"] = df["feat"].astype(str)
    return df


def _step_dtype_regression():
    """EDGE_DISCOVERY_REPORT_SPEC §3 — the categorical dtype bug (F3) is dead across
    every level dtype. Before the fix, bool/int64/mixed columns matched ZERO rows on
    the str level key → CI (None, None) → verdict capped at noise. This asserts a
    real CI that excludes 0 in BOTH discovery and confirm mode, and a non-zero
    val-side N for the native (non-str) dtypes."""
    cases = {
        "bool": (True, False),
        "int64": (1, 0),
        "mixed": (True, False),
        "str": ("hi", "lo"),
    }
    for dtype, (a, b) in cases.items():
        disc = _dtype_frame(a, b, dtype, seed=11)
        val = _dtype_frame(a, b, dtype, seed=99)
        # discovery-only mode
        buckets: list = []
        rec_d = ee._categorical_screen(disc, None, "feat", buckets)
        lo, hi = rec_d.get("delta_disc_ci", [None, None])
        _true(lo is not None and hi is not None,
              f"[{dtype}] discovery CI is non-null (was (None,None) pre-fix)")
        _true(ee._ci_excludes_zero(lo, hi),
              f"[{dtype}] discovery CI excludes 0 (strong separation)")
        _true(min(rec_d.get("best_worst_n_disc", [0, 0])) >= ee.MIN_BUCKET_N,
              f"[{dtype}] both discovery levels N>=150")
        # confirm mode (val is not None) — val-side stats must be on real rows
        buckets2: list = []
        rec_c = ee._categorical_screen(disc, val, "feat", buckets2)
        _true(min(rec_c.get("best_worst_n_val", [0, 0])) > 0,
              f"[{dtype}] val-side N non-zero (native compare matched rows)")
        _true(rec_c.get("delta_val") is not None,
              f"[{dtype}] delta_val computed (not None)")
        # the confirm-mode CSV must carry both DISCOVERY and VALIDATION level rows
        splits = {row["split"] for row in buckets2}
        _true("VALIDATION" in splits, f"[{dtype}] VALIDATION rows emitted in confirm mode")

    # Ladder sees a real CI from a bool feature -> candidate (not noise).
    disc = _dtype_frame(True, False, "bool", seed=3)
    rec = ee._categorical_screen(disc, None, "feat", [])
    v = ee._apply_candidate_criteria(rec, fdr_reject=True)
    _eq(v, "candidate", "bool feature with real CI + FDR reject => candidate (not noise)")
    _true(rec.get("criteria", {}).get("ci_excludes_0") is True,
          "criteria flags stamped: ci_excludes_0 True for the bool candidate")


def _step_no_leak(run_dir):
    """§11.1 — discovery JSON has no *_val keys, CSV has no VALIDATION rows,
    validation_untouched is true."""
    ed = os.path.join(run_dir, "edge_engine")
    _stub_gate(run_dir, "verdict")
    res = ee.stage1_discovery(run_dir, ed, False)
    _true(res["validation_untouched"], "validation_untouched: true")
    leaked = [k for f in res["features"] for k in f if k.endswith("_val")
              or k.startswith("delta_val") or k == "val_favoured_pos_quarters"]
    _eq(leaked, [], "no *_val keys in any feature record")
    csv = pd.read_csv(os.path.join(ed, "stage1_discovery_features.csv"))
    _eq(int((csv["split"] == "VALIDATION").sum()), 0, "no VALIDATION rows in CSV")
    _eq(res["interactions"], "deferred_to_confirm", "interactions deferred")
    return res


def _step_language(run_dir):
    """§11.3 — discovery email body AND the rendered .md carry the stamp and never
    the banned strings 'survivor'/' edge'. Also asserts the report structure: one
    table per feature record (no silent truncation) and every rendered expR cell
    carries its n (blind-spot guard)."""
    from backtest.diagnostics import edge_report
    ed = os.path.join(run_dir, "edge_engine")
    res = ee._read_json(ee._discovery_path(ed))
    body = edge_email.build_discovery_body(res).lower()
    _true(ee.DISCOVERY_LANGUAGE_STAMP.lower() in body, "language stamp present in email")
    _true("survivor" not in body, "'survivor' absent from discovery email")
    _true(" edge" not in body, "' edge' absent from discovery email")
    # the email carries the new summary lines (§5).
    _true("near-misses:" in body, "email carries near-miss count line")
    _true(edge_report.REPORT_MD in body, "email points at the committed report")

    # render + scan the .md (F7 / §4).
    p = edge_report.render_discovery_report(ed)
    _true(p is not None, "report rendered")
    md = open(p, encoding="utf-8").read()
    mlow = md.lower()
    _true(ee.DISCOVERY_LANGUAGE_STAMP.lower() in mlow, "language stamp present in .md")
    _true("survivor" not in mlow, "'survivor' absent from report .md")
    _true(" edge" not in mlow, "' edge' absent from report .md")

    # ── report structure (§7): the APPENDIX renders one section per feature
    # record, and every expR table in it carries an n column (blind-spot guard).
    appendix = md.split("## 6. Appendix")[-1]
    feats = res.get("features", [])
    # every appendix feature heading is "### `<feat>`  (...)".
    appendix_headings = [ln for ln in appendix.splitlines() if ln.startswith("### `")]
    _eq(len(appendix_headings), len(feats),
        "appendix has exactly one table section per feature record (no truncation)")
    # every appendix table that shows an expR column also shows an n column.
    naked = [ln for ln in appendix.splitlines()
             if ln.startswith("| ") and "expR" in ln and "| n |" not in ln]
    _eq(naked, [], "every appendix expR table header also carries an n column")


def _step_gate_refusal(run_dir):
    """§11.4 — verdict scope, no approval => stage1 refuses, writes no
    stage1_features.json and no ledger line."""
    ed = os.path.join(run_dir, "edge_engine")
    canon = ee._stage_path(ed, 1)
    if os.path.exists(canon):
        os.remove(canon)
    res = ee.stage1(run_dir, ed, False)
    _eq(res.get("pass"), False, "refused stage1 pass=false")
    _eq(res.get("refused"), "approval_gate", "refused: approval_gate")
    _true(not os.path.exists(canon), "no stage1_features.json written")
    _true(not os.path.exists(ee._ledger_path(ed)), "no ledger line written")


def _step_token_binding_and_single_use(run_dir):
    """§11.5 + §11.8 — approve arms one confirm; token binds to the exact
    discovery bytes; second confirm refuses; burn re-opens with N=2 + caveat."""
    ed = os.path.join(run_dir, "edge_engine")
    disc = ee._read_json(ee._discovery_path(ed))
    token = disc["token"]

    # mutate one byte of discovery => the issued token no longer approves.
    disc2 = dict(disc); disc2["n_discovery"] = disc["n_discovery"] + 1
    ee._write_json(ee._discovery_path(ed), disc2)
    _eq(ee.approve(ed, token).get("approved"), False, "mutated discovery => approve refused")
    # restore the real discovery (recompute a fresh token on the true bytes).
    ee.stage1_discovery(run_dir, ed, False)
    fresh = ee._compute_token(ed)["token"]

    ap = ee.approve(ed, fresh)
    _true(ap["approved"], "fresh token approves")
    # first confirm consumes the token: ledger N=1, not burned.
    r1 = ee.stage1(run_dir, ed, False)
    _eq(r1.get("pass"), True, "first confirm passes the gate")
    _eq(r1.get("validation_runs"), 1, "ledger N=1 after first confirm")
    _eq(r1.get("validation_burned"), False, "not burned at N=1")
    _true(json.load(open(ee._approval_path(ed)))["consumed"], "token consumed")

    # second confirm WITHOUT a new approval refuses.
    r2 = ee.stage1(run_dir, ed, False)
    _eq(r2.get("refused"), "approval_gate", "second confirm refuses (consumed)")

    # burn re-opens: ledger N=2, burned true, and the stage-4 caveat appears.
    r3 = ee.stage1(run_dir, ed, False, "auditing a suspected detector bug")
    _eq(r3.get("validation_runs"), 2, "ledger N=2 after burn")
    _eq(r3.get("validation_burned"), True, "burned true at N=2")
    cav = ee._collect_caveats([], False, {}, ee._ledger_summary(ed))
    _true(any("VALIDATION RE-OPENED" in c for c in cav), "burn caveat present for stage 4")


def _step_force_is_not_a_bypass(run_dir):
    """§11.9 — --force (forced=True) does not open the gate."""
    ed = os.path.join(run_dir, "edge_engine")
    # consume any live approval so the gate is genuinely closed.
    ap = ee._read_json(ee._approval_path(ed)) or {}
    ap["consumed"] = True
    ee._write_json(ee._approval_path(ed), ap)
    res = ee.stage1(run_dir, ed, True)  # forced=True
    _eq(res.get("refused"), "approval_gate", "forced run still refused by gate")


def _step_phase_ab_agreement(run_dir):
    """§11.6 — delta_disc per feature is identical in discovery and confirm
    (same function, same discovery frame => no drift)."""
    ed = os.path.join(run_dir, "edge_engine")
    disc = ee._read_json(ee._discovery_path(ed))
    conf = ee._read_json(ee._stage_path(ed, 1))
    dmap = {f["feature"]: f.get("delta_disc") for f in disc["features"]}
    cmap = {f["feature"]: f.get("delta_disc") for f in conf["features"]}
    common = set(dmap) & set(cmap)
    mismatches = [f for f in common if dmap[f] != cmap[f]]
    _eq(mismatches, [], "delta_disc identical across discovery and confirm")


def _step_exploratory_bypass():
    """§11.7 — exploratory scope has no gate; stage1 runs the legacy loop with no
    approval; stage1_discovery errors out."""
    df = _build_trades()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _write_run(tmp, df)
        ed = os.path.join(run_dir, "edge_engine")
        _stub_gate(run_dir, "exploratory")
        # no gate: stage1 runs and passes with NO approval file present.
        _true(not os.path.exists(ee._approval_path(ed)), "no approval file")
        res = ee.stage1(run_dir, ed, False)
        _eq(res.get("pass"), True, "exploratory stage1 runs with no gate")
        _true("validation_runs" not in res, "no ledger stamps in exploratory")
        # --phase discovery errors out in exploratory scope.
        try:
            ee.stage1_discovery(run_dir, ed, False)
            _bad("stage1_discovery should raise in exploratory scope")
        except SystemExit:
            _ok("stage1_discovery raises SystemExit in exploratory scope")


def _run_sequence():
    """The 6 stateful steps share ONE run_dir in a fixed order: discovery is
    written first (_step_no_leak), read by later steps, and _step_token_binding
    runs the confirm that writes the stage1_features _step_phase_ab reads. This
    ordering is deliberate — the steps are a sequence, not independent tests,
    which is why they are `_step_*` (invisible to pytest collection) and run
    through this one wrapper instead."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = _write_run(tmp, _build_trades())
        _step_no_leak(run_dir)
        _step_language(run_dir)
        _step_gate_refusal(run_dir)
        _step_token_binding_and_single_use(run_dir)
        _step_phase_ab_agreement(run_dir)
        _step_force_is_not_a_bypass(run_dir)


def _run_all_steps():
    """Every check, in the one order that satisfies the sequence's state deps.
    Shared by the script entry (run_all) and the pytest entry so the two runners
    exercise byte-identical logic — no path is green in one and untested in the
    other."""
    _step_candidate_ladder()
    _step_dtype_regression()
    _step_exploratory_bypass()
    _run_sequence()


def test_staged_review():
    """Sole pytest entry point. The `_step_*` helpers are hidden from pytest
    collection on purpose: they take a shared `run_dir` (which pytest would
    mis-read as a fixture) and they are order-dependent. This one test drives the
    whole ordered suite and asserts on the failure ledger, so a `_bad()` in any
    step (which only records, never raises) still turns pytest red."""
    before = len(_FAILS)
    _run_all_steps()
    assert len(_FAILS) == before, f"{len(_FAILS) - before} failure(s): {_FAILS[before:]}"


def run_all():
    _run_all_steps()
    if _FAILS:
        print(f"\n{len(_FAILS)} FAILURE(S)")
        sys.exit(1)
    print("\nstaged-review guard: OK")


if __name__ == "__main__":
    run_all()
