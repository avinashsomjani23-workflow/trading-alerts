"""
h4_range.py — H4 dealing range, built from H1, mapped onto H1.

WHAT THIS IS (plain English)
  The dealing range (ceiling / floor / equilibrium) is now defined on H4, not
  H1. We do NOT fetch an H4 feed (none is available) — we build H4 candles in
  memory from the H1 data Phase 1 already pulls, find the most recent confirmed
  H4 swing high and swing low, and expose those PRICE LEVELS so they can be
  drawn as horizontal walls on the H1 timeline.

THE RULE (single, simple — no six movers, no placeholders, no ratchet)
  Every scan:
    ceiling = price of the MOST RECENT CONFIRMED H4 swing high
    floor   = price of the MOST RECENT CONFIRMED H4 swing low
  picked INDEPENDENTLY (the high and the low may be from different dates).
  Recomputed from scratch each scan — a newer confirmed swing always replaces
  the old one, higher or lower, so a wall can never go stale.

  Broken-wall live tracking: if the live price has CLOSED beyond a wall, that
  wall rides the live extreme (max High since the broken swing for a ceiling;
  min Low for a floor) until the next confirmed H4 swing of that type forms.
  Price is therefore always contained within [floor, ceiling].

TWO RANGES, ON PURPOSE
  - LIVE range (ceiling/floor above, with broken-wall tracking): for display,
    the digest, containment, the trader's eye.
  - LAST FULLY-CONFIRMED range (both walls are confirmed swings, NO live
    tracking): the STABLE reference the CHoCH premium/discount 25% gate reads.
    A gate computed off a moving range jitters and fires false CHoCHs, so the
    gate must read a frozen range. Exposed separately as `confirmed_*`.

ATR
  All ATR in this system is H1 ATR (locked decision). The H4 swing leg-size
  filter therefore measures H4 swing legs against the H4 bars' own True Range
  via dealing_range._filter_swings_by_leg_atr — that is the H4 leg's volatility,
  which is the correct denominator for an H4 swing. (This is NOT an ATR-source
  contradiction: "H1 ATR only" governs the H1 BOS / CHoCH displacement gates in
  dealing_range. Swing significance on H4 must be measured on H4 bars or the
  filter is inert.) If you want the H4 swing filter off, pass
  min_leg_atr_mult=None.

COUPLING
  Reads the SHARED swing definition (dealing_range.detect_swings: lb-3 + ATR
  leg filter). Writes no state. Imported by Phase 1 to attach an H4 range block
  to the per-pair state, and by compute_pd_position consumers to read walls.
"""

from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

import dealing_range as _dr  # shared swing definition only; no state I/O


# H4 grid anchor (UTC hour). Derived from the live H4 close cadence (18:30 IST
# close => 13:00 UTC), which puts H4 opens on 01/05/09/13/17/21 UTC. Only the
# PHASE matters (which hours are bucket boundaries), not the absolute number.
H4_ANCHOR_HOUR_UTC = 1

SWING_LOOKBACK   = _dr.SWING_LOOKBACK         # 3
MIN_LEG_ATR_MULT = _dr.MIN_LEG_ATR_MULT       # 1.5

# Minimum H1 bars in the window to even attempt an H4 range. Need enough H4
# candles to confirm a swing on each side (lb*2+1) plus headroom.
_MIN_H1_BARS = 80


def _extract_utc_index(df) -> Optional[pd.DatetimeIndex]:
    """Return a tz-aware UTC DatetimeIndex for df, or None.

    Phase 1's live df comes from fetch_data, which does reset_index() — so the
    timestamps live in a 'Datetime' (or 'Date') COLUMN, not the index. The
    probe / cold-start path may hand us a DatetimeIndex instead. Handle both.
    Naive timestamps are assumed UTC (yfinance returns UTC).
    """
    idx = None
    if 'Datetime' in getattr(df, 'columns', []):
        idx = pd.DatetimeIndex(pd.to_datetime(df['Datetime']))
    elif 'Date' in getattr(df, 'columns', []):
        idx = pd.DatetimeIndex(pd.to_datetime(df['Date']))
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        return None
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    else:
        idx = idx.tz_convert('UTC')
    return idx


def build_h4(df) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build H4 OHLC from H1, gap-aware (never spans a market-closed gap).

    Each H1 bar is assigned to a 4h grid bucket (phase-anchored to the H4 grid).
    A new H4 candle starts when the bucket changes OR when there is a >1h gap
    between consecutive H1 bars (market closed). This guarantees no candle spans
    a weekend / session break — matching how H1 itself omits closed-market bars.

    Returns (h4_df, stats). h4_df columns: Open/High/Low/Close/_nbars, indexed
    by the first H1 timestamp of each H4 candle (tz-aware UTC). Empty df if the
    input is unusable.
    """
    empty = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', '_nbars'])
    if df is None or len(df) == 0:
        return empty, {}
    idx = _extract_utc_index(df)
    if idx is None or len(idx) != len(df):
        return empty, {'error': 'no_timestamp_source'}

    for col in ('Open', 'High', 'Low', 'Close'):
        if col not in df.columns:
            return empty, {'error': f'missing_{col}'}

    O = df['Open'].to_numpy(dtype=float)
    H = df['High'].to_numpy(dtype=float)
    L = df['Low'].to_numpy(dtype=float)
    C = df['Close'].to_numpy(dtype=float)

    epoch0 = pd.Timestamp('1970-01-01', tz='UTC')
    epoch_h = ((idx - epoch0) // pd.Timedelta(hours=1)).astype('int64').to_numpy()
    bucket = (epoch_h - H4_ANCHOR_HOUR_UTC) // 4

    rows: List[Tuple] = []
    stats = {'gaps_split': 0, 'full4': 0, 'short': 0, 'bars_per_candle': {}}
    cur = None
    prev_ts = None
    prev_bucket = None

    def flush():
        nonlocal cur
        if cur is None:
            return
        rows.append((cur['ts'], cur['o'], cur['h'], cur['l'], cur['c'], cur['n']))
        if cur['n'] == 4:
            stats['full4'] += 1
        else:
            stats['short'] += 1
        stats['bars_per_candle'][cur['n']] = stats['bars_per_candle'].get(cur['n'], 0) + 1
        cur = None

    for k in range(len(idx)):
        ts = idx[k]
        b = int(bucket[k])
        gap = prev_ts is not None and (ts - prev_ts) > pd.Timedelta(hours=1)
        new_bucket = b != prev_bucket
        if cur is None:
            cur = {'ts': ts, 'o': O[k], 'h': H[k], 'l': L[k], 'c': C[k], 'n': 1}
        elif new_bucket or gap:
            if gap and not new_bucket:
                stats['gaps_split'] += 1
            flush()
            cur = {'ts': ts, 'o': O[k], 'h': H[k], 'l': L[k], 'c': C[k], 'n': 1}
        else:
            cur['h'] = max(cur['h'], H[k])
            cur['l'] = min(cur['l'], L[k])
            cur['c'] = C[k]
            cur['n'] += 1
        prev_ts = ts
        prev_bucket = b
    flush()

    h4 = pd.DataFrame(rows, columns=['ts', 'Open', 'High', 'Low', 'Close', '_nbars']).set_index('ts')
    return h4, stats


def _iso(ts) -> Optional[str]:
    try:
        return ts.isoformat() if hasattr(ts, 'isoformat') else (str(ts) if ts is not None else None)
    except Exception:
        return None


def compute_h4_range(df, min_leg_atr_mult: Optional[float] = MIN_LEG_ATR_MULT) -> Dict[str, Any]:
    """Compute the H4 dealing range from an H1 df. Pure; writes no state.

    Returns a dict (always — never raises on thin data):
      {
        'valid':            bool,         # both LIVE walls present + range non-degenerate
        'ceiling':          float|None,   # LIVE ceiling (broken-wall live tracking applied)
        'floor':            float|None,
        'equilibrium':      float|None,
        'ceiling_ts':       str|None,     # confirmed-swing ts (None when riding live)
        'floor_ts':         str|None,
        'ceiling_broken':   bool,         # True => ceiling is riding the live high
        'floor_broken':     bool,
        # Last FULLY-CONFIRMED range (no live tracking) — the stable reference
        # for the CHoCH 25% premium/discount gate. valid only when BOTH walls
        # are confirmed swings.
        'confirmed_valid':  bool,
        'confirmed_ceiling':float|None,
        'confirmed_floor':  float|None,
        'confirmed_eq':     float|None,
        'confirmed_ceiling_ts': str|None,
        'confirmed_floor_ts':   str|None,
        'n_confirmed_highs':int,
        'n_confirmed_lows': int,
        'current_price':    float|None,
        'contained':        bool|None,
        'source':           str,
        'stats':            dict,         # gap/build stats for diagnostics
      }
    """
    out: Dict[str, Any] = {
        'valid': False, 'ceiling': None, 'floor': None, 'equilibrium': None,
        'ceiling_ts': None, 'floor_ts': None,
        'ceiling_broken': False, 'floor_broken': False,
        'confirmed_valid': False, 'confirmed_ceiling': None, 'confirmed_floor': None,
        'confirmed_eq': None, 'confirmed_ceiling_ts': None, 'confirmed_floor_ts': None,
        'n_confirmed_highs': 0, 'n_confirmed_lows': 0,
        'current_price': None, 'contained': None,
        'source': 'h4', 'stats': {},
    }
    if df is None or len(df) < _MIN_H1_BARS:
        out['source'] = 'insufficient_h1'
        return out

    h4, stats = build_h4(df)
    out['stats'] = stats
    if len(h4) < SWING_LOOKBACK * 2 + 1:
        out['source'] = 'insufficient_h4'
        return out

    # Shared swing definition on the H4 bars (lb-3 + ATR leg filter on H4 TR).
    swings = _dr.detect_swings(h4, lookback=SWING_LOOKBACK, min_leg_atr_mult=min_leg_atr_mult)
    n = len(h4)
    # A swing at H4 idx i is CONFIRMED only after SWING_LOOKBACK further H4 bars.
    confirmed = [s for s in swings if s['idx'] + SWING_LOOKBACK < n]
    highs = [s for s in confirmed if s['type'] == 'high']
    lows = [s for s in confirmed if s['type'] == 'low']
    out['n_confirmed_highs'] = len(highs)
    out['n_confirmed_lows'] = len(lows)

    # Live price = last H1 close.
    try:
        price = float(df['Close'].to_numpy(dtype=float)[-1])
    except Exception:
        price = None
    out['current_price'] = price

    if not highs or not lows:
        out['source'] = 'no_confirmed_pair'
        return out

    ceil_s = max(highs, key=lambda s: s['idx'])   # most recent confirmed high
    floor_s = max(lows, key=lambda s: s['idx'])    # most recent confirmed low
    conf_ceiling = float(ceil_s['price'])
    conf_floor = float(floor_s['price'])

    # --- Last fully-confirmed range (frozen reference for the CHoCH gate) ----
    if conf_ceiling > conf_floor:
        out['confirmed_valid'] = True
        out['confirmed_ceiling'] = conf_ceiling
        out['confirmed_floor'] = conf_floor
        out['confirmed_eq'] = (conf_ceiling + conf_floor) / 2.0
        out['confirmed_ceiling_ts'] = _iso(ceil_s.get('ts'))
        out['confirmed_floor_ts'] = _iso(floor_s.get('ts'))

    # --- Live range (broken-wall live tracking) -----------------------------
    ceiling = conf_ceiling
    floor = conf_floor
    ceiling_ts = _iso(ceil_s.get('ts'))
    floor_ts = _iso(floor_s.get('ts'))
    ceiling_broken = floor_broken = False

    if price is not None:
        Hh = h4['High'].to_numpy(dtype=float)
        Ll = h4['Low'].to_numpy(dtype=float)
        if price > ceiling:
            live_hi = float(Hh[ceil_s['idx'] + 1:].max()) if ceil_s['idx'] + 1 < n else price
            ceiling = max(ceiling, live_hi, price)
            ceiling_ts = None          # riding live extreme, not a confirmed swing
            ceiling_broken = True
        if price < floor:
            live_lo = float(Ll[floor_s['idx'] + 1:].min()) if floor_s['idx'] + 1 < n else price
            floor = min(floor, live_lo, price)
            floor_ts = None
            floor_broken = True

    if ceiling <= floor:
        out['source'] = 'degenerate_range'
        return out

    out['valid'] = True
    out['ceiling'] = ceiling
    out['floor'] = floor
    out['equilibrium'] = (ceiling + floor) / 2.0
    out['ceiling_ts'] = ceiling_ts
    out['floor_ts'] = floor_ts
    out['ceiling_broken'] = ceiling_broken
    out['floor_broken'] = floor_broken
    out['contained'] = (floor <= price <= ceiling) if price is not None else None
    return out
