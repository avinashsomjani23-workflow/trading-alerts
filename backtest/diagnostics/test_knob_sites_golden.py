"""Golden gate for the knob-override sites — protects "use live, never copy".

The sweep overrides MIN_LEG_ATR_MULT by patching the def-time DEFAULT of every
live function that captures it (the "def-time trap": monkeypatching the module
constant does NOT change a value already bound as a default — see
driver.self_check step 3). The list of those functions, `_MIN_LEG_DEFAULT_SITES`
in driver.py, is a HAND-MAINTAINED mirror of the live code.

The blind spot: if someone later adds a NEW live function that takes a
`min_leg_atr_mult` (or `_min_leg_atr_mult` / `min_mult`) default, the sweep will
SILENTLY fail to override it there, and driver._verify() will not catch it
(it only checks the sites it already knows). A two-year corpus would then be
quietly wrong for that knob.

This test scans the live modules at import time and asserts the set of
default-capturing sites EQUALS the registry. CI goes red the day live drifts —
the same golden-gate pattern the structure baseline already uses. This is the
load-bearing guarantee behind "the sweep just uses the live system".

Run:  python -m backtest.diagnostics.test_knob_sites_golden
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import dealing_range
import smc_detector
import h4_range

from backtest.diagnostics import driver

# Live modules the sweep claims to cover for the MIN_LEG knob. Scanning these
# four (not the whole repo) keeps the test fast and matches where the constant
# and its consumers live.
_SCANNED_MODULES = [dealing_range, smc_detector, h4_range]

# Parameter names that carry the MIN_LEG knob as a def-time default. The three
# spellings the live code actually uses (see _MIN_LEG_DEFAULT_SITES in driver).
_MIN_LEG_PARAM_NAMES = {"min_leg_atr_mult", "_min_leg_atr_mult", "min_mult"}


def _has_min_leg_default(func) -> bool:
    """True if `func` has one of the MIN_LEG param names WITH a default value
    (not just a parameter — a default is what the trap captures)."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return False
    for name, p in sig.parameters.items():
        if name in _MIN_LEG_PARAM_NAMES and p.default is not inspect.Parameter.empty:
            return True
    return False


def discover_live_min_leg_sites():
    """Return the set of (module_name, func_qualname) for every live function
    that captures a MIN_LEG default. This is the GROUND TRUTH the registry must
    match. `min_mult` is generic, so we only count it when the function also
    looks leg/swing-related to avoid false positives from unrelated helpers."""
    found = set()
    for mod in _SCANNED_MODULES:
        for _name, obj in vars(mod).items():
            if not (inspect.isfunction(obj) and obj.__module__ == mod.__name__):
                continue
            if not _has_min_leg_default(obj):
                continue
            params = set(inspect.signature(obj).parameters)
            # `min_mult` alone is ambiguous; require a leg/swing fingerprint.
            if params & {"min_leg_atr_mult", "_min_leg_atr_mult"} or \
               ("min_mult" in params and ("swing" in obj.__name__.lower()
                                          or "leg" in obj.__name__.lower())):
                found.add((mod.__name__, obj.__qualname__))
    return found


def registry_sites():
    """The sites driver.py claims to override, as (module_name, func_qualname)."""
    return {(f.__module__, f.__qualname__) for f, _pname in driver._MIN_LEG_DEFAULT_SITES}


def test_min_leg_sites_match_live():
    live = discover_live_min_leg_sites()
    registry = registry_sites()
    missing_from_registry = live - registry   # live drifted; sweep would miss it
    stale_in_registry = registry - live       # registry points at a gone func
    assert not missing_from_registry, (
        "KNOB-SITE DRIFT: live functions take a MIN_LEG default but the sweep "
        f"does NOT override them: {sorted(missing_from_registry)}. Add them to "
        "driver._MIN_LEG_DEFAULT_SITES or the sweep silently ignores the knob "
        "there — a two-year-corpus-corrupting blind spot.")
    assert not stale_in_registry, (
        "STALE KNOB SITE: driver._MIN_LEG_DEFAULT_SITES references a function "
        f"that no longer takes a MIN_LEG default: {sorted(stale_in_registry)}. "
        "driver._verify() would raise at run time; fix the registry.")


def main():
    test_min_leg_sites_match_live()
    print("[knob-sites-golden] OK — registry matches live MIN_LEG default sites:")
    for m, q in sorted(registry_sites()):
        print(f"    {m}.{q}")


if __name__ == "__main__":
    main()
