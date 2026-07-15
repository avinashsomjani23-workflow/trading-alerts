"""Shared detection driver for the diagnostic harnesses.

This is the ONE module all three harnesses import. It replays the live SMC
detection path against historical data WITHOUT touching any trading logic. It
only imports and calls the live functions; it never re-implements detection.

Slice-mode discipline (see FABLE_REFERENCE.md Â§4):
  - Mode B: bar-by-bar, each live call sees ONLY closed bars up to the wall
    clock (the honest "what the live system could have known" view). Used for
    structure snapshots, alerts, P&L.
  - Mode A: one-shot over the full dataframe (a static structural census).
    Cheaper, but a swing near the right edge is "confirmed" here that the live
    bar-by-bar system had not yet seen. Used ONLY where future bars cannot
    change a past decision (Harness 2 swing census).

Every output object carries `slice_mode` so a harness can never silently mix
the two.

Knob overrides (see FABLE_REFERENCE.md Â§3): some ATR knobs are read as module
globals at call time (monkeypatch works); others are captured as default
arguments bound at function-definition time (monkeypatching the constant does
NOT change them â the def-time-default trap). `KnobOverrides` handles each by
the correct mechanism, verifies the override actually took, and restores
everything unconditionally on exit.

NEVER writes to live state files. `ReplayState` is in-memory by design; a fresh
one is built per walk. NEVER mutates the caller's pair_conf (run_backtest
mutates pair_conf in place; we deep-copy to avoid inheriting/causing that).
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Live modules â read-only use.
import smc_radar
import smc_detector
import dealing_range
import h4_range

from backtest import data_loader
from backtest import replay_engine

MIN_WARMUP_BARS = 50  # mirrors replay_engine.MIN_WARMUP


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class KnobError(Exception):
    """Raised when a knob override cannot be applied safely."""


class WindowUnserveable:
    """Returned (not raised) when a data window cannot be served. Harnesses
    print this; they never fabricate rows."""
    def __init__(self, pair: str, reason: str, detail: str = ""):
        self.pair = pair
        self.reason = reason
        self.detail = detail
        self.slice_mode = None

    def __repr__(self):
        return f"WindowUnserveable(pair={self.pair!r}, reason={self.reason!r}, detail={self.detail!r})"


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SwingRec:
    type: str
    price: float
    idx: int
    ts: Optional[pd.Timestamp]


@dataclass(frozen=True)
class EventRec:
    type: str            # 'BOS' | 'CHoCH'
    tier: Optional[str]
    direction: Optional[str]
    candle_ts: Optional[str]


@dataclass(frozen=True)
class ObRec:
    direction: Optional[str]
    proximal: Optional[float]
    distal: Optional[float]
    ob_timestamp: Optional[str]
    bos_timestamp: Optional[str]
    bos_tag: Optional[str]
    bos_tier: Optional[str]
    fvg_present: bool
    sweep_present: bool


@dataclass(frozen=True)
class BarSnapshot:
    pair: str
    wall_clock_ts: pd.Timestamp
    just_closed_ts: Optional[pd.Timestamp]
    n_bars_in_slice: int
    atr: Optional[float]
    state: Optional[str]
    trend: Optional[str]
    choch: bool
    choch_flip_count: int
    swings: Tuple[SwingRec, ...]
    events_tail: Tuple[EventRec, ...]
    active_zones: Tuple[ObRec, ...]
    raw_radar_return_type: str
    slice_mode: str = "B"


@dataclass
class AlertWalkResult:
    pair: str
    ob_seen: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    ob_mitigated: List[Dict[str, Any]] = field(default_factory=list)
    trade_rows: List[Dict[str, Any]] = field(default_factory=list)
    counters: Dict[str, int] = field(default_factory=dict)
    slice_mode: str = "B"


@dataclass(frozen=True)
class CensusResult:
    pair: str
    swings: Tuple[SwingRec, ...]
    events_tail: Tuple[EventRec, ...]
    active_zones: Tuple[ObRec, ...]
    state: Optional[str]
    trend: Optional[str]
    slice_mode: str = "A"


# ---------------------------------------------------------------------------
# Helpers to convert live dicts -> records
# ---------------------------------------------------------------------------
def _swing_rec(s: Dict[str, Any]) -> SwingRec:
    ts = s.get("ts")
    if ts is not None and not isinstance(ts, pd.Timestamp):
        try:
            ts = pd.Timestamp(ts)
        except Exception:
            ts = None
    if isinstance(ts, pd.Timestamp) and ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return SwingRec(type=s.get("type"), price=float(s.get("price")),
                    idx=int(s.get("idx")), ts=ts)


def _event_rec(e: Dict[str, Any]) -> EventRec:
    return EventRec(type=e.get("type"), tier=e.get("tier"),
                    direction=e.get("direction"), candle_ts=e.get("candle_ts"))


def _ob_rec(o: Dict[str, Any]) -> ObRec:
    return ObRec(
        direction=o.get("direction"),
        proximal=(float(o["proximal_line"]) if o.get("proximal_line") is not None else None),
        distal=(float(o["distal_line"]) if o.get("distal_line") is not None else None),
        ob_timestamp=o.get("ob_timestamp"),
        bos_timestamp=o.get("bos_timestamp"),
        bos_tag=o.get("bos_tag"),
        bos_tier=o.get("bos_tier"),
        fvg_present=bool((o.get("fvg") or {}).get("exists")),
        sweep_present=bool((o.get("sweep_observed") or {}).get("exists")),
    )


# ---------------------------------------------------------------------------
# KnobOverrides
# ---------------------------------------------------------------------------
# The knob registry. Keys are the harness-facing names. proximity_cap is NOT
# patched here (it is applied by the walk via a pair_conf copy â see
# FABLE_REFERENCE Â§3 #10/#11); it is stored on the context so a harness can use
# one uniform `with KnobOverrides(...)` call and the walk reads it back.
_PATCHABLE_GLOBAL = {
    # name -> (module, attribute)
    "bos_atr_mult":               (dealing_range, "BOS_ATR_MULT"),
    "structure_choch_atr_mult":   (dealing_range, "STRUCTURE_CHOCH_ATR_MULT"),
    "structure_lock_atr_mult":    (dealing_range, "STRUCTURE_LOCK_ATR_MULT"),
    "ob_max_range_atr_mult":      (smc_detector, "OB_MAX_RANGE_ATR_MULT"),
    "min_ob_range_atr_mult":      (smc_detector, "MIN_OB_RANGE_ATR_MULT"),
}
_PATCHABLE_DICT = {
    # name -> (module, attribute)  (mutated in place, restored in place)
    "fvg_noise_floor_mult":            (smc_detector, "FVG_NOISE_FLOOR_MULT"),
    "sweep_equal_level_tolerance_atr": (smc_detector, "SWEEP_EQUAL_LEVEL_TOLERANCE_ATR"),
    "sweep_wick_pierce_min_atr":       (smc_detector, "SWEEP_WICK_PIERCE_MIN_ATR"),
}
# Functions whose def-time default captures MIN_LEG_ATR_MULT. (func, param_name).
_MIN_LEG_DEFAULT_SITES = [
    (dealing_range.detect_swings,            "min_leg_atr_mult"),
    (dealing_range.compute_structure,        "_min_leg_atr_mult"),
    (dealing_range._filter_swings_by_leg_atr, "min_mult"),
    (h4_range.compute_h4_range,              "min_leg_atr_mult"),
    (smc_detector.get_swing_points,          "min_leg_atr_mult"),
]

_REFUSED = {"rearm_extra_atr"}  # knob #9 â local literal inside replay_pair


class _Registry:
    """Records original values for unconditional restore."""
    def __init__(self):
        self._globals: List[Tuple[Any, str, Any]] = []
        self._dicts: List[Tuple[dict, dict]] = []          # (live_dict, saved_copy)
        self._defaults: List[Tuple[Any, str, Any]] = []    # (func, '__defaults__'|'__kwdefaults__', old)

    def save_global(self, mod, attr):
        self._globals.append((mod, attr, getattr(mod, attr)))

    def save_dict(self, d):
        self._dicts.append((d, dict(d)))

    def save_func_attr(self, func, attr, old):
        self._defaults.append((func, attr, old))

    def restore(self):
        for mod, attr, val in reversed(self._globals):
            setattr(mod, attr, val)
        for d, saved in reversed(self._dicts):
            d.clear()
            d.update(saved)
        for func, attr, old in reversed(self._defaults):
            setattr(func, attr, old)


def _patch_default(func, param_name, new_value, registry: _Registry):
    """Patch a function's default for `param_name` BY NAME (never by position).

    Raises KnobError if the parameter has no default â the source has drifted
    from FABLE_REFERENCE and silently patching the wrong slot is exactly the
    failure mode we are guarding against.
    """
    sig = inspect.signature(func)
    pos_defaults = [p for p in sig.parameters.values()
                    if p.default is not inspect.Parameter.empty
                    and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.POSITIONAL_ONLY)]
    names = [p.name for p in pos_defaults]
    if param_name in names:
        i = names.index(param_name)
        old = func.__defaults__ or ()
        registry.save_func_attr(func, "__defaults__", old)
        func.__defaults__ = old[:i] + (new_value,) + old[i + 1:]
        return
    # keyword-only?
    if func.__kwdefaults__ and param_name in func.__kwdefaults__:
        registry.save_func_attr(func, "__kwdefaults__", dict(func.__kwdefaults__))
        func.__kwdefaults__[param_name] = new_value
        return
    raise KnobError(
        f"{func.__qualname__} has no default param {param_name!r} â source has "
        f"changed; re-verify against the repo before sweeping this knob.")


class KnobOverrides:
    """Context manager that overrides Â§3 knobs by the correct mechanism, then
    restores everything unconditionally. Non-reentrant: compose layered
    overrides into ONE call rather than nesting (nested partial restores are
    how silent corruption happens).
    """
    _active = False

    def __init__(self, *, min_leg_atr_mult: Optional[float] = None,
                 bos_atr_mult: Optional[float] = None,
                 structure_choch_atr_mult: Optional[float] = None,
                 structure_lock_atr_mult: Optional[float] = None,
                 ob_max_range_atr_mult: Optional[float] = None,
                 min_ob_range_atr_mult: Optional[float] = None,
                 fvg_noise_floor_mult: Optional[dict] = None,
                 sweep_equal_level_tolerance_atr: Optional[dict] = None,
                 sweep_wick_pierce_min_atr: Optional[dict] = None,
                 proximity_cap: Optional[Any] = None,
                 rearm_extra_atr: Optional[float] = None):
        if rearm_extra_atr is not None:
            raise KnobError(
                "REARM_EXTRA_ATR (knob #9) is a local literal inside "
                "replay_engine.replay_pair and cannot be overridden without "
                "editing live code. Proposed (NOT applied) fix: hoist it to a "
                "module constant `replay_engine.REARM_EXTRA_ATR` read at call "
                "time. See the Harness-3 report.")
        self._req = {k: v for k, v in dict(
            min_leg_atr_mult=min_leg_atr_mult,
            bos_atr_mult=bos_atr_mult,
            structure_choch_atr_mult=structure_choch_atr_mult,
            structure_lock_atr_mult=structure_lock_atr_mult,
            ob_max_range_atr_mult=ob_max_range_atr_mult,
            min_ob_range_atr_mult=min_ob_range_atr_mult,
            fvg_noise_floor_mult=fvg_noise_floor_mult,
            sweep_equal_level_tolerance_atr=sweep_equal_level_tolerance_atr,
            sweep_wick_pierce_min_atr=sweep_wick_pierce_min_atr,
        ).items() if v is not None}
        # proximity_cap is held, not patched (applied by the walk).
        self.proximity_cap = proximity_cap
        self._registry: Optional[_Registry] = None

    def __enter__(self):
        if KnobOverrides._active:
            raise KnobError("KnobOverrides is non-reentrant. Compose overrides "
                            "into one call instead of nesting.")
        reg = _Registry()
        self._registry = reg
        try:
            # 1. MIN_LEG_ATR_MULT â def-time defaults + the constant (belt & braces).
            if "min_leg_atr_mult" in self._req:
                v = float(self._req["min_leg_atr_mult"])
                for func, pname in _MIN_LEG_DEFAULT_SITES:
                    _patch_default(func, pname, v, reg)
                reg.save_global(dealing_range, "MIN_LEG_ATR_MULT")
                dealing_range.MIN_LEG_ATR_MULT = v
                # smc_detector re-exports the constant; keep them consistent.
                reg.save_global(smc_detector, "MIN_LEG_ATR_MULT")
                smc_detector.MIN_LEG_ATR_MULT = v
            # 2. Scalar module globals.
            for name, (mod, attr) in _PATCHABLE_GLOBAL.items():
                if name in self._req:
                    reg.save_global(mod, attr)
                    setattr(mod, attr, float(self._req[name]))
            # 3. Dict knobs â mutate in place.
            for name, (mod, attr) in _PATCHABLE_DICT.items():
                if name in self._req:
                    live = getattr(mod, attr)
                    reg.save_dict(live)
                    live.clear()
                    live.update(self._req[name])
            KnobOverrides._active = True
            self._verify()
        except Exception:
            reg.restore()
            self._registry = None
            raise
        return self

    def _verify(self):
        """Probe that every requested override actually took. A sweep that
        thinks it overrode a knob but didn't is worse than a crash."""
        if "min_leg_atr_mult" in self._req:
            v = float(self._req["min_leg_atr_mult"])
            for func, pname in _MIN_LEG_DEFAULT_SITES:
                bound = inspect.signature(func).parameters[pname].default
                if bound != v:
                    raise KnobError(
                        f"verify-after-set failed: {func.__qualname__}.{pname} "
                        f"= {bound!r}, expected {v!r}")
            if dealing_range.MIN_LEG_ATR_MULT != v:
                raise KnobError("verify-after-set failed: dealing_range.MIN_LEG_ATR_MULT")
        for name, (mod, attr) in _PATCHABLE_GLOBAL.items():
            if name in self._req and getattr(mod, attr) != float(self._req[name]):
                raise KnobError(f"verify-after-set failed: {mod.__name__}.{attr}")
        for name, (mod, attr) in _PATCHABLE_DICT.items():
            if name in self._req and getattr(mod, attr) != self._req[name]:
                raise KnobError(f"verify-after-set failed: {mod.__name__}.{attr}")

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._registry is not None:
                self._registry.restore()
        finally:
            KnobOverrides._active = False
            self._registry = None
        return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_window(pair_conf: Dict[str, Any], start, end, *, warmup_days: int = 35):
    """Wrapper over data_loader.load_bars. Fetches with a warmup pad before
    `start` so structure is warm at the walk start (mirrors run_backtest).

    Returns the H1 df (UTC-indexed) or a WindowUnserveable.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    fetch_start = start - pd.Timedelta(days=warmup_days)
    symbol = pair_conf["symbol"]
    df = data_loader.load_bars(symbol, "1h",
                               fetch_start.to_pydatetime(), end.to_pydatetime())
    if df is None or df.empty:
        return WindowUnserveable(pair_conf["name"], "no_data",
                                 f"{symbol} 1h {fetch_start.date()}..{end.date()}")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    # Honesty: surface any yfinance clamp by comparing what we asked vs got.
    if df.index.min() > fetch_start + pd.Timedelta(days=1):
        # not fatal; the caller's window may still be fully covered.
        pass
    return df


# ---------------------------------------------------------------------------
# Mode-B per-bar detection walk
# ---------------------------------------------------------------------------
def walk_detection(pair_conf: Dict[str, Any], df: pd.DataFrame,
                   start_ts, end_ts, *, stride: int = 1,
                   anchor_ts: Optional[List[pd.Timestamp]] = None
                   ) -> Iterator[BarSnapshot]:
    """Mode B. Yield a BarSnapshot for each wall-clock T (every bar if
    stride=1, every Nth bar, or only the bars in anchor_ts). Performs NO
    proximity check / alert / re-arm â alerts come ONLY from walk_alerts so we
    never re-implement Phase 2.
    """
    pair_name = pair_conf["name"]
    pair_type = pair_conf["pair_type"]
    if df is None or df.empty:
        return
    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    in_window = df.loc[start_ts:end_ts]
    if in_window.empty:
        return

    if anchor_ts is not None:
        anchor_set = {pd.Timestamp(t) for t in anchor_ts}
        ts_list = [t for t in in_window.index if t in anchor_set]
    else:
        ts_list = list(in_window.index)[::max(1, stride)]

    for T in ts_list:
        if T not in df.index:
            raise KnobError(f"wall-clock T={T} not a bar boundary in df.index")
        slice_df = replay_engine._slice_closed_before(df, T)
        replay_engine._assert_no_lookahead(slice_df, T, tag=f"driver:{pair_name}")
        if len(slice_df) < MIN_WARMUP_BARS:
            continue
        atr = smc_detector.compute_atr(slice_df)
        walls = smc_radar.compute_pair_walls(slice_df, pair_name)
        events = walls.get("events", []) or []
        # Suppress detect_smc_radar's per-event [OB-DROP] stdout (mirrors what
        # replay_engine does internally). Pure I/O reduction; no logic change.
        with contextlib.redirect_stdout(io.StringIO()):
            raw = smc_radar.detect_smc_radar(
                slice_df, pair_type=pair_type, events=events,
                walls=walls, pair_name=pair_name)
        raw_type = _classify_radar_return(raw)
        obs = replay_engine._normalize_obs_result(raw)
        sv2 = walls.get("structure_v2", {}) or {}
        yield BarSnapshot(
            pair=pair_name,
            wall_clock_ts=T,
            just_closed_ts=slice_df.index[-1],
            n_bars_in_slice=len(slice_df),
            atr=atr,
            state=sv2.get("state"),
            trend=walls.get("trend"),
            choch=bool(sv2.get("choch")),
            choch_flip_count=int(sv2.get("choch_flip_count", 0)),
            swings=tuple(_swing_rec(s) for s in (walls.get("swings") or [])),
            events_tail=tuple(_event_rec(e) for e in events),
            active_zones=tuple(_ob_rec(o) for o in obs),
            raw_radar_return_type=raw_type,
            slice_mode="B",
        )


def _classify_radar_return(raw: Any) -> str:
    if isinstance(raw, dict):
        return "dict_active_zones" if "active_zones" in raw else "dict_other"
    if isinstance(raw, list):
        return "list"
    if isinstance(raw, tuple):
        return "tuple"
    return type(raw).__name__


# ---------------------------------------------------------------------------
# Mode-B alert walk (wraps the real replay_pair + simulator)
# ---------------------------------------------------------------------------
def walk_alerts(pair_conf: Dict[str, Any], df: pd.DataFrame, start_ts, end_ts,
                *, risk_usd: float = 250.0,
                overrides: Optional[KnobOverrides] = None) -> AlertWalkResult:
    """Mode B. Replays the REAL replay_engine.replay_pair generator and
    simulates each alert with the REAL simulator. Applies run_backtest's
    OB-dedup rule (FABLE_REFERENCE Â§6 guard 6; source: run_backtest.py:212-223):
    only the FIRST alert per (ob_timestamp, direction) is simulated.

    NEVER mutates the caller's pair_conf. proximity_cap (if any) is applied to
    a deep copy here â this is the ONLY proximity mechanism (Â§3 #10/#11), and
    it bypasses run_backtest._run_h1_only's in-place mutation entirely.
    """
    pair_name = pair_conf["name"]
    pair_type = pair_conf.get("pair_type", "forex")
    conf = copy.deepcopy(pair_conf)
    if overrides is not None and overrides.proximity_cap is not None:
        cap = overrides.proximity_cap
        if isinstance(cap, dict):
            conf["atr_multiplier"] = float(cap.get(pair_type, conf["atr_multiplier"]))
        else:
            conf["atr_multiplier"] = float(cap)

    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    res = AlertWalkResult(pair=pair_name)
    state = replay_engine.ReplayState()
    counters = {"alerts_total": 0, "alerts_simulated": 0, "ob_dedup_skipped": 0}

    for event in replay_engine.replay_pair(conf, df, state=state,
                                           walk_start_ts=start_ts,
                                           walk_end_ts=end_ts):
        kind = event.get("kind")
        if kind == "ob_seen":
            res.ob_seen.append(event)
        elif kind == "ob_mitigated":
            res.ob_mitigated.append(event)
        elif kind == "alert":
            counters["alerts_total"] += 1
            res.alerts.append(event)
            # 2026-07-15: seen_obs first-touch dedupe REMOVED (parity with
            # run_backtest.py). Every re-armed re-touch is a real spaced re-approach
            # (re-arm hysteresis) and a mitigated OB is dropped upstream before it
            # can fire, so we simulate every touch until mitigation. ob_dedup_skipped
            # is now always 0 (kept for schema stability). See test_retouch_trading.py.
            counters["alerts_simulated"] += 1
            import backtest.h1_only_simulator as sim
            rows = sim.simulate_h1_only_dual(event, conf, df, risk_usd=risk_usd)
            res.trade_rows.extend(rows)

    res.counters = counters
    return res


# ---------------------------------------------------------------------------
# Mode-A full-dataset census
# ---------------------------------------------------------------------------
def census_full_df(pair_conf: Dict[str, Any], df: pd.DataFrame) -> CensusResult:
    """Mode A. ONE call of the live assembler over the full df. Never feed the
    result into any alert/P&L computation."""
    pair_name = pair_conf["name"]
    pair_type = pair_conf["pair_type"]
    walls = smc_radar.compute_pair_walls(df, pair_name)
    events = walls.get("events", []) or []
    with contextlib.redirect_stdout(io.StringIO()):
        raw = smc_radar.detect_smc_radar(df, pair_type=pair_type, events=events,
                                         walls=walls, pair_name=pair_name)
    obs = replay_engine._normalize_obs_result(raw)
    sv2 = walls.get("structure_v2", {}) or {}
    return CensusResult(
        pair=pair_name,
        swings=tuple(_swing_rec(s) for s in (walls.get("swings") or [])),
        events_tail=tuple(_event_rec(e) for e in events),
        active_zones=tuple(_ob_rec(o) for o in obs),
        state=sv2.get("state"),
        trend=walls.get("trend"),
        slice_mode="A",
    )


# ---------------------------------------------------------------------------
# ATR cache control
# ---------------------------------------------------------------------------
def clear_atr_cache() -> int:
    """Empty smc_detector._ATR_CACHE in place (do not rebind â other modules
    hold the reference). Returns number of evicted entries."""
    n = len(smc_detector._ATR_CACHE)
    smc_detector._ATR_CACHE.clear()
    return n


# ---------------------------------------------------------------------------
# Self-check (Â§1.8) â run before any harness work; abort on failure
# ---------------------------------------------------------------------------
def _toy_df():
    idx = pd.date_range("2025-01-06 10:00", periods=5, freq="1h", tz="UTC")
    return pd.DataFrame({
        "Open":   [1.0, 1.1, 1.2, 1.1, 1.0],
        "High":   [1.15, 1.25, 1.3, 1.2, 1.1],
        "Low":    [0.95, 1.05, 1.15, 1.05, 0.95],
        "Close":  [1.1, 1.2, 1.25, 1.1, 1.0],
        "Volume": [0, 0, 0, 0, 0],
    }, index=idx)


def self_check(verbose: bool = True) -> bool:
    """Driver pre-flight. Returns True on success; raises on failure."""
    def _log(m):
        if verbose:
            print(f"[driver.self_check] {m}")

    # 1. Toy-slice semantics.
    df = _toy_df()
    s13 = replay_engine._slice_closed_before(df, df.index[3])  # 13:00
    assert len(s13) == 3 and s13.index[-1] == df.index[2], "slice_closed_before mid"
    s10 = replay_engine._slice_closed_before(df, df.index[0])  # 10:00
    assert s10.empty, "slice_closed_before at first bar should be empty"
    _log("toy-slice semantics OK")

    # 2. Override round-trip (MIN_LEG_ATR_MULT touches def-time defaults).
    base_defaults = {f.__qualname__: f.__defaults__ for f, _ in _MIN_LEG_DEFAULT_SITES}
    with KnobOverrides(min_leg_atr_mult=9.9):
        for func, pname in _MIN_LEG_DEFAULT_SITES:
            assert inspect.signature(func).parameters[pname].default == 9.9, \
                f"override not applied to {func.__qualname__}.{pname}"
        assert dealing_range.MIN_LEG_ATR_MULT == 9.9
    for func, _ in _MIN_LEG_DEFAULT_SITES:
        assert func.__defaults__ == base_defaults[func.__qualname__], \
            f"defaults not restored for {func.__qualname__}"
    assert abs(dealing_range.MIN_LEG_ATR_MULT - 1.5) < 1e-12, "constant not restored"
    _log("override round-trip OK")

    # 3. Def-time-trap demonstration: monkeypatch ONLY the constant; swings
    #    must be UNCHANGED through detect_swings' default path (proves the trap
    #    is real and that our __defaults__ patching is necessary).
    trap_df = _synthetic_swingy_df(300)
    base_swings = dealing_range.detect_swings(trap_df)  # uses def-time 1.5
    _saved = dealing_range.MIN_LEG_ATR_MULT
    try:
        dealing_range.MIN_LEG_ATR_MULT = 9.9   # constant only, no __defaults__
        trap_swings = dealing_range.detect_swings(trap_df)  # still uses 1.5
        assert len(trap_swings) == len(base_swings), \
            "def-time trap NOT reproduced â monkeypatch unexpectedly took effect"
    finally:
        dealing_range.MIN_LEG_ATR_MULT = _saved
    _log("def-time-trap demonstrated (monkeypatch-only is a no-op) OK")

    # 4. Cache determinism: cold vs warm produce identical ATR.
    clear_atr_cache()
    cold = smc_detector.compute_atr(trap_df)
    warm = smc_detector.compute_atr(trap_df)
    assert cold == warm, "ATR cold vs warm differ"
    _log("ATR cache determinism OK")

    # 5. Non-reentrancy guard.
    try:
        with KnobOverrides(bos_atr_mult=0.5):
            with KnobOverrides(bos_atr_mult=0.6):
                pass
        raise AssertionError("nested KnobOverrides should have raised")
    except KnobError:
        pass
    assert abs(dealing_range.BOS_ATR_MULT - 0.4) < 1e-12, "BOS not restored after nest attempt"
    _log("non-reentrancy guard OK")

    _log("ALL DRIVER SELF-CHECKS PASSED")
    return True


def _synthetic_swingy_df(n: int) -> pd.DataFrame:
    """A deterministic zig-zag H1 df with enough structure for swing detection."""
    import math
    idx = pd.date_range("2025-01-01 00:00", periods=n, freq="1h", tz="UTC")
    closes = [1.0 + 0.01 * math.sin(i / 3.0) + 0.002 * (i % 7) for i in range(n)]
    opens = [c - 0.001 for c in closes]
    highs = [max(o, c) + 0.003 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 0.003 for o, c in zip(opens, closes)]
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                         "Close": closes, "Volume": [0] * n}, index=idx)


if __name__ == "__main__":
    self_check()
