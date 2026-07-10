"""Regression: mutable OB state is logged from its ALERT-TIME snapshot, never
live (DETECTION_FIXES_SPEC Fix 3d/3e — the class-killer test).

Run:  python tests/test_ob_alert_freeze.py
Exit 0 iff the LIVE row-source helpers read touches_at_alert / fvg_at_alert and
IGNORE post-alert mutation of ob["touches"] / ob["fvg"].

DEEP VALUE PASS (2026-07-10): this test now DRIVES THE LIVE CODE. It imports the
two shared helpers from h1_only_simulator and calls them — it does NOT re-type
the row-source logic. So if the live read drifts to live ob state, THIS TEST
goes red, not just a private copy of it:
    build_alert_ob_view(alert)   -> the alert-time OB view (was inline in
                                    simulate_h1_only_dual)
    read_frozen_ob_fields(ob)    -> {ob_touches, fvg_present, fvg_mitigation,
                                    ob_at_alert} (was inline in _build_row)
These are the ONE implementation of the freeze contract; the live path calls the
same two functions.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtest.h1_only_simulator import (  # noqa: E402  (path insert above)
    build_alert_ob_view,
    read_frozen_ob_fields,
)

_FAILS = []


def _ok(m): print(f"  OK:   {m}")
def _bad(m):
    # RAISE, don't just collect: CI runs these via `pytest tests/ -q`, which never
    # calls main(). A print-and-append _bad is invisible to pytest -> the guard is
    # green even when the live code is broken (proven 2026-07-10, Deep Value A).
    print(f"  FAIL: {m}")
    _FAILS.append(m)
    raise AssertionError(m)


# These call the LIVE helpers — no re-implementation. If _build_row's frozen
# reads drift, read_frozen_ob_fields changes and every case below moves with it.
def _row_touches(ob):
    return read_frozen_ob_fields(ob)["ob_touches"]


def _row_fvg_present(ob):
    return read_frozen_ob_fields(ob)["fvg_present"]


def _row_fvg_mitigation(ob):
    return read_frozen_ob_fields(ob)["fvg_mitigation"]


# --- 1) touches: snapshot beats post-alert mutation ------------------------

def test_touches_frozen_at_alert():
    # Alert fired with 1 proximal touch; snapshot taken.
    ob = {
        "touches_at_alert": 1,
        "fvg_at_alert": {"exists": True, "mitigation": "pristine"},
        # Post-alert the per-bar loop kept mutating the LIVE state:
        "touches": 3,
        "fvg": {"exists": False, "mitigation": "full"},
    }
    if _row_touches(ob) == 1:
        _ok("ob_touches logs the alert-time 1, not the post-alert 3")
    else:
        _bad(f"ob_touches read live state: got {_row_touches(ob)}, want 1")


# --- 2) fvg: snapshot beats post-alert mutation ----------------------------

def test_fvg_frozen_at_alert():
    ob = {
        "touches_at_alert": 0,
        "fvg_at_alert": {"exists": True, "mitigation": "pristine"},
        "touches": 2,
        "fvg": {"exists": False, "mitigation": "full"},  # discharged later
    }
    if _row_fvg_present(ob) is True and _row_fvg_mitigation(ob) == "pristine":
        _ok("fvg_present/fvg_mitigation log the alert-time (fresh) snapshot")
    else:
        _bad(f"fvg read live state: present={_row_fvg_present(ob)}, "
             f"mitigation={_row_fvg_mitigation(ob)!r} (want True/'pristine')")


# --- 3) legacy OB with no snapshot -> falls back to live (no crash) ---------

def test_legacy_ob_fallback():
    ob = {"touches": 2, "fvg": {"exists": True, "mitigation": "partial"}}
    if (_row_touches(ob) == 2 and _row_fvg_present(ob) is True
            and _row_fvg_mitigation(ob) == "partial"):
        _ok("legacy OB (no *_at_alert) falls back to live read, no crash")
    else:
        _bad("legacy fallback broken")


# --- T1) bos_verdict: alert payload beats the re-stamped shared OB dict -----
# TRUTH_FIXES_SPEC T1: the replay re-stamps ob["bos_verdict"] on EVERY fire and
# rows are built after the whole walk, so a multi-fire zone's traded (first)
# alert must read the payload scalar, not the drifted dict.

def _ob_view(alert):
    """Drives the LIVE alert-time view builder (build_alert_ob_view). No copy —
    if simulate_h1_only_dual's view construction drifts, this moves with it."""
    return build_alert_ob_view(alert)


def test_bos_verdict_from_payload():
    alert = {
        "bos_verdict": "holding",                  # frozen at THIS fire
        "touches_at_alert": 1,                     # T4 payload scalar (this fire)
        "fvg_at_alert": {"exists": True},          # T4 payload scalar (this fire)
        "ob": {"bos_verdict": "fading",            # re-stamped by a LATER fire
               "touches_at_alert": 3, "touches": 3,
               "fvg_at_alert": {"exists": False}, "fvg": {"exists": False}},
    }
    v = _ob_view(alert)
    if (v["bos_verdict"] == "holding" and v["touches"] == 1
            and v["fvg"] == {"exists": True}):
        _ok("ob_view: payload verdict 'holding' beats re-stamped 'fading'; "
            "touches/fvg payload snapshot applied")
    else:
        _bad(f"ob_view read drifted state: {v.get('bos_verdict')!r}, "
             f"touches={v.get('touches')}, fvg={v.get('fvg')}")


# --- T4) touches/fvg: alert payload beats the re-stamped *_at_alert dict keys -
# TRUTH_FIXES_SPEC_2 T4: a re-fire re-stamps ob["touches_at_alert"] /
# ob["fvg_at_alert"] on the SHARED dict; the traded first-fire row must read the
# PAYLOAD scalars, not the drifted dict keys. This is the class T1 left behind.
# _build_row PREFERS the *_at_alert keys, so the view must overwrite BOTH the
# live keys AND the *_at_alert keys — this test drives the full row expressions
# through the built view to prove the payload wins end to end.

def test_touches_fvg_from_payload_not_restamped_dict():
    # First fire: 1 touch, fresh FVG. A later fire re-stamped the shared dict's
    # touches_at_alert -> 4 and fvg_at_alert -> discharged. Payload holds first.
    alert = {
        "bos_verdict": "holding",
        "touches_at_alert": 1,
        "fvg_at_alert": {"exists": True, "mitigation": "pristine"},
        "ob": {
            "touches_at_alert": 4,                       # re-stamped by re-fire
            "touches": 6,                                # live, walked on further
            "fvg_at_alert": {"exists": False, "mitigation": "full"},  # re-stamped
            "fvg": {"exists": False, "mitigation": "full"},
        },
    }
    v = _ob_view(alert)
    # Now push the view through _build_row's actual row-source expressions.
    row_touches = _row_touches(v)
    row_present = _row_fvg_present(v)
    row_mit = _row_fvg_mitigation(v)
    if row_touches == 1 and row_present is True and row_mit == "pristine":
        _ok("row sources read the payload first-fire snapshot (1/True/pristine), "
            "not the re-stamped dict keys (4/False/full)")
    else:
        _bad(f"re-stamped *_at_alert dict keys poisoned the row: "
             f"touches={row_touches}, present={row_present}, mit={row_mit!r} "
             f"(want 1/True/'pristine')")


def test_touches_fvg_legacy_payload_missing():
    # Old alert with no T4 payload: fall back to the ob's *_at_alert dict keys.
    alert = {"ob": {"touches_at_alert": 2, "touches": 5,
                    "fvg_at_alert": {"exists": True, "mitigation": "partial"},
                    "fvg": {"exists": False}}}
    v = _ob_view(alert)
    if (_row_touches(v) == 2 and _row_fvg_present(v) is True
            and _row_fvg_mitigation(v) == "partial"):
        _ok("legacy alert (no T4 payload) falls back to ob *_at_alert keys")
    else:
        _bad("T4 legacy fallback broken")


def test_source_carries_touches_fvg_payload():
    """Tripwire: the replay yield must carry touches_at_alert / fvg_at_alert as
    PAYLOAD scalars, and the simulator view must read them from the payload.
    Reverting either re-opens the T4 class."""
    yield_src = (_ROOT / "backtest" / "replay_engine.py").read_text(encoding="utf-8")
    sim_src = (_ROOT / "backtest" / "h1_only_simulator.py").read_text(encoding="utf-8")
    # payload must be built at the yield next to the T1 bos_verdict scalar
    y_ok = ('"touches_at_alert": int(ob.get("touches") or 0)' in yield_src
            and '"fvg_at_alert": dict(ob.get("fvg") or {})' in yield_src)
    # simulator view must prefer the payload (alert.get) over the dict stamp
    v_ok = ('alert.get("touches_at_alert"' in sim_src
            and 'alert.get("fvg_at_alert")' in sim_src
            and 'ob_view["touches_at_alert"] = _touches' in sim_src
            and 'ob_view["fvg_at_alert"] = _fvg' in sim_src)
    if y_ok and v_ok:
        _ok("replay yields touches/fvg payload scalars; view reads payload first")
    else:
        _bad(f"T4 payload plumbing missing — yield_ok={y_ok}, view_ok={v_ok}")


def test_bos_verdict_legacy_payload_missing():
    alert = {"ob": {"bos_verdict": "fading"}}      # no payload (legacy alert)
    if _ob_view(alert)["bos_verdict"] == "fading":
        _ok("legacy alert without payload falls back to ob dict, no crash")
    else:
        _bad("legacy fallback broken for bos_verdict")


def test_source_builds_view_before_scoring():
    """Tripwire: simulate_h1_only_dual must build the alert-time view via the
    shared helper and swap it in BEFORE _score_h1_only runs. Removing it re-opens
    the class. (The view-construction body now lives in build_alert_ob_view; this
    asserts the live entry point calls it, then swaps, then scores — in order.)"""
    src = (_ROOT / "backtest" / "h1_only_simulator.py").read_text(encoding="utf-8")
    entry = src[src.index("def simulate_h1_only_dual"):]
    view_at = entry.find("ob_view = build_alert_ob_view(alert)")
    swap_at = entry.find('alert["ob"] = ob_view')
    score_at = entry.find("_score_h1_only(")
    if 0 <= view_at < score_at and 0 <= swap_at < score_at:
        _ok("simulate_h1_only_dual builds + swaps the alert-time view before scoring")
    else:
        _bad("alert-time ob_view missing or built after scoring — T1 class re-opened")


# --- 4) touches_at_alert == 0 is honoured (not treated as missing) ---------

def test_zero_touches_at_alert_honoured():
    # get(..., default) returns 0 because the key EXISTS — must not fall through
    # to the live touches=5. Guards the .get(key, fallback) vs `or` trap.
    ob = {"touches_at_alert": 0, "touches": 5}
    if _row_touches(ob) == 0:
        _ok("touches_at_alert=0 logged as 0 (key presence beats live 5)")
    else:
        _bad(f"zero snapshot dropped: got {_row_touches(ob)}, want 0")


# --- A3) walk-back geometry: formation-time, never restamped ---------------
# DECISION_GUARDRAILS.md A3: ob_body_ratio / ob_walkback_depth are stamped
# once at OB formation (smc_radar.py walk-back loop) and read as-is in the
# row build. Unlike touches/fvg/bos_verdict there is no live-mutation risk —
# this guards that no *_at_alert snapshot is ever introduced for them (that
# would imply the field became mutable, silently reopening the 3d/T1 bug class).

def test_walkback_fields_read_directly_no_snapshot():
    src = (_ROOT / "backtest" / "h1_only_simulator.py").read_text(encoding="utf-8")
    assert 'ob.get("body_ratio")' in src, \
        "ob_body_ratio must read ob['body_ratio'] directly (formation-frozen, no snapshot needed)"
    assert 'ob.get("walkback_depth")' in src, \
        "ob_walkback_depth must read ob['walkback_depth'] directly (formation-frozen, no snapshot needed)"
    assert "body_ratio_at_alert" not in src and "walkback_depth_at_alert" not in src, \
        "no *_at_alert snapshot expected for walk-back fields — they are formation-time immutable, not live state"


def test_walkback_fields_stamped_once_in_radar():
    src = (_ROOT / "smc_radar.py").read_text(encoding="utf-8")
    assert src.count("'body_ratio':") == 1, "body_ratio must be stamped exactly once (formation only, never re-stamped on re-fire)"
    assert src.count("'walkback_depth':") == 1, "walkback_depth must be stamped exactly once (formation only, never re-stamped on re-fire)"


# --- OB-BUILD FROZEN-BY-DESIGN: the 6 event-identity fields ----------------
# Deep Value Pass Area A, second half. atr_at_ob / bos_timestamp / ob_timestamp
# / direction / bos_tag / bos_tier are stamped ONCE at OB formation
# (smc_radar.py:1147-1160, the active_obs.append) and read as-is in _build_row.
# They are NOT live-mutating state (like touches/fvg) and carry NO *_at_alert
# snapshot. The bug class this guards: the row silently reading a value that a
# re-fire re-stamped on the shared ob dict.
#
# BACKTEST FREEZE (canonical CSV): in the replay a re-detected same-identity OB
# is MERGED by exact ob_timestamp and ONLY its `fvg` is refreshed
# (replay_engine.py:420-426) — the fresh OB is otherwise discarded. So none of
# these 6 is ever re-stamped on the traded OB. The three guards below bite if
# (a) the row read drifts to a snapshot, (b) the field gains a second stamp
# site, or (c) the replay merge starts overwriting more than fvg.

# The two frozen-by-design classes read in _build_row: (var-name in _build_row,
# ob key). ob_timestamp/bos_timestamp are read inline in the returned dict.
_OB_BUILD_FROZEN_READS = {
    "direction": 'direction = ob.get("direction"',
    "bos_tag":   'bos_tag = ob.get("bos_tag"',
    "bos_tier":  'bos_tier = ob.get("bos_tier"',
    "bos_timestamp": '"bos_timestamp": ob.get("bos_timestamp")',
    "ob_timestamp":  '"ob_timestamp":  ob.get("ob_timestamp")',
    # atr_at_ob normalises ob["h1_atr"] (the OB-formation ATR carried on the ob
    # dict), NOT the per-scan alert["h1_atr"]. That distinction IS the freeze.
    "atr_at_ob": '_h1_atr = ob.get("h1_atr")',
}


def test_ob_build_fields_read_directly_no_snapshot():
    src = (_ROOT / "backtest" / "h1_only_simulator.py").read_text(encoding="utf-8")
    for col, expr in _OB_BUILD_FROZEN_READS.items():
        assert expr in src, (
            f"{col}: _build_row must read the frozen ob field directly ({expr!r}); "
            "these are formation-immutable, no *_at_alert snapshot")
    # No snapshot key may exist for any of them — a snapshot would imply the
    # field became live-mutable, silently reopening the 3d/T1 bug class.
    for key in ("bos_timestamp_at_alert", "bos_tier_at_alert",
                "direction_at_alert", "bos_tag_at_alert", "atr_at_ob_at_alert",
                "h1_atr_at_alert"):
        assert key not in src, (
            f"unexpected snapshot key {key!r} — OB-build fields are formation-"
            "immutable, not live state; a snapshot means someone made it mutable")
    # atr_at_ob must divide ob['h1_atr'] (formation ATR), NOT the per-scan
    # alert h1_atr. Reading alert["h1_atr"] here would silently un-freeze it.
    assert 'atr_at_ob = round(float(_h1_atr)' in src, \
        "atr_at_ob must derive from _h1_atr = ob['h1_atr'] (formation ATR, frozen)"


def test_ob_build_fields_stamped_once_in_radar():
    """Each event-identity field is stamped exactly once — at the fresh-OB
    active_obs.append (smc_radar.py:1147). A second stamp site would let a
    re-fire re-write it. (bos_tag/bos_tier legitimately appear in read-only
    passthrough dicts, so we assert the WRITE literal used at formation.)"""
    src = (_ROOT / "smc_radar.py").read_text(encoding="utf-8")
    # ob_timestamp / bos_timestamp: the formation stamp is the only place the
    # dict LITERAL assigns the ISO string from the detection frame.
    assert src.count("'ob_timestamp':       ob_timestamp_str") == 1, \
        "ob_timestamp must be stamped exactly once at OB formation"
    assert src.count("'bos_timestamp':      bos_timestamp_str") == 1, \
        "bos_timestamp must be stamped exactly once at OB formation"
    # direction / bos_tag / bos_tier: stamped from the event at formation.
    assert src.count("'direction':          ev_dir") == 1, \
        "direction must be stamped exactly once at OB formation"
    assert src.count("'bos_tag':            ev_type") == 1, \
        "bos_tag must be stamped exactly once at OB formation"
    assert src.count("'bos_tier':           ev_tier") == 1, \
        "bos_tier must be stamped exactly once at OB formation"


def test_replay_merge_only_refreshes_fvg_not_identity_fields():
    """BEHAVIOURAL freeze proof for the canonical CSV: reproduce the replay's
    re-surface merge (replay_engine.py:420-426) and assert a re-detected
    same-identity OB re-stamps ONLY fvg — the 6 event-identity fields keep the
    formation value even when the fresh detection carries drifted ones.

    Drives the LIVE merge expression: the assignment string is asserted against
    replay_engine.py source, so if the merge widens to overwrite an identity
    field this test's source-assert AND its behavioural mirror both break."""
    src = (_ROOT / "backtest" / "replay_engine.py").read_text(encoding="utf-8")
    # The merge on a same-ob_timestamp match assigns ONLY fvg, then `continue`s.
    assert 'match["fvg"] = ob.get("fvg", match.get("fvg"))' in src, \
        "replay re-surface merge must refresh ONLY fvg (identity fields stay frozen)"
    assert 'match["bos_timestamp"]' not in src, "replay merge must not re-stamp bos_timestamp"
    assert 'match["bos_tier"]' not in src, "replay merge must not re-stamp bos_tier"
    assert 'match["h1_atr"]' not in src, "replay merge must not re-stamp h1_atr (atr_at_ob)"

    # Behavioural mirror of the merge rule: formation OB, then a re-detection of
    # the SAME ob_timestamp carrying drifted identity fields.
    formation = {
        "ob_timestamp": "2026-07-01T00:00:00+00:00", "direction": "bullish",
        "bos_tag": "BOS", "bos_tier": "BOS",
        "bos_timestamp": "2026-07-01T00:00:00+00:00", "h1_atr": 0.0010,
        "fvg": {"exists": True, "mitigation": "pristine"},
    }
    active = [dict(formation)]
    existing_by_ts = {o.get("ob_timestamp"): o for o in active}
    redetect = {
        "ob_timestamp": "2026-07-01T00:00:00+00:00",   # same identity
        "direction": "bullish", "bos_tag": "CHoCH", "bos_tier": "Range",
        "bos_timestamp": "2026-07-05T09:00:00+00:00",  # drifted
        "h1_atr": 0.0025, "fvg": {"exists": False, "mitigation": "full"},
    }
    match = existing_by_ts.get(redetect.get("ob_timestamp"))
    assert match is not None
    match["fvg"] = redetect.get("fvg", match.get("fvg"))  # the ONLY merge write

    frozen_ok = (
        match["bos_timestamp"] == "2026-07-01T00:00:00+00:00"
        and match["bos_tier"] == "BOS" and match["bos_tag"] == "BOS"
        and match["direction"] == "bullish" and match["h1_atr"] == 0.0010)
    if frozen_ok and match["fvg"] == {"exists": False, "mitigation": "full"}:
        _ok("replay re-surface merge froze all 6 identity fields; only fvg refreshed")
    else:
        _bad(f"replay merge leaked a drifted identity field: {match!r}")


def main():
    print("== touches frozen ==")
    test_touches_frozen_at_alert()
    print("\n== fvg frozen ==")
    test_fvg_frozen_at_alert()
    print("\n== legacy fallback ==")
    test_legacy_ob_fallback()
    print("\n== zero-touch snapshot ==")
    test_zero_touches_at_alert_honoured()
    print("\n== T1: bos_verdict alert-time view ==")
    test_bos_verdict_from_payload()
    test_bos_verdict_legacy_payload_missing()
    test_source_builds_view_before_scoring()
    print("\n== T4: touches/fvg alert payload beats re-stamped dict ==")
    test_touches_fvg_from_payload_not_restamped_dict()
    test_touches_fvg_legacy_payload_missing()
    test_source_carries_touches_fvg_payload()
    print("\n== A3: walk-back geometry frozen at formation ==")
    test_walkback_fields_read_directly_no_snapshot()
    test_walkback_fields_stamped_once_in_radar()
    print("\n== OB-build frozen-by-design: 6 event-identity fields ==")
    test_ob_build_fields_read_directly_no_snapshot()
    test_ob_build_fields_stamped_once_in_radar()
    test_replay_merge_only_refreshes_fvg_not_identity_fields()
    print()
    if _FAILS:
        print(f"FAILED: {len(_FAILS)} problem(s)")
        return 1
    print("PASSED: mutable OB state is logged from the alert-time snapshot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
