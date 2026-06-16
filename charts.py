"""Shared H1 chart STYLE engine (Wave 2 item 2C).

Why this exists
---------------
There were TWO H1 chart renderers — `smc_radar.generate_h1_chart` (Phase 1
scout digest) and `Phase2_Alert_Engine.generate_h1_chart` / `_zoomed_chart`
(Phase 2 trade email) — with NO shared code. Their styles had DRIFTED: the
same market rendered with different candle proportions (Phase 1 thin-body /
fat-wick / tall figure; Phase 2 fat-body / thin-wick / short figure) and the
colour palette was duplicated by hand in both files.

Decision (owner, "Option 3"): the two charts keep their DIFFERENT CONTENT
(scout = full structure incl. dealing-range walls; trade = entry/SL/TP focus),
but they share ONE style engine so they can never visually drift again. This
module is that single source: colours, candle geometry, the base canvas, the
BOS/CHoCH colour rule, and the PNG encode. Both renderers call these; neither
redefines a colour or a candle.

Canonical look = Phase 1's (the owner's reference scout chart): thin bodies,
fat wicks, butt-capped wicks. Phase 2 adopts it, so candles now match across
both emails. Figure HEIGHT stays a per-call policy (Phase 1 is adaptive-tall,
Phase 2 is fixed) because the two serve different framing jobs — but it comes
through one function so the seam is explicit.

Behaviour-neutral by construction: every constant below is copied verbatim from
the value that was already live in the two renderers (verified 2026-06-16). The
golden baseline images live in debug/ (gitignored) from 2C step-zero.
"""

from __future__ import annotations

from io import BytesIO
import base64

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — both phases run in CI/Actions with no display
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# --- Palette (was hand-duplicated in both renderers) ------------------------
BG          = "#131722"   # chart background (TradingView dark)
SPINE       = "#2a2a3e"   # axis spine colour
CANDLE_UP   = "#26a69a"   # bullish candle (close >= open)
CANDLE_DOWN = "#ef5350"   # bearish candle

# Zone (Order Block) band
ZONE_FACE   = "#9b59b6"
ZONE_EDGE   = "#bb8fce"
OB_OUTLINE  = "#d7bde2"   # primary OB candle outline
GREY_FACE   = "#666666"   # invalidated zone fill
GREY_EDGE   = "#888888"   # invalidated zone edge / OB outline

# FVG
FVG_PRISTINE_FACE = "#27ae60"
FVG_PRISTINE_EDGE = "#2ecc71"
FVG_PARTIAL_FACE  = "#f4d03f"   # amber — partial mitigation (caution)
FVG_PARTIAL_EDGE  = "#f1c40f"

# Structural-event line palette (v2 has only BOS / Range BOS / CHoCH —
# no Major/Minor). Matches the email legend in both phases.
BOS_COLOR       = "#00bcd4"   # cyan  — internal swing break
RANGE_BOS_COLOR = "#00897b"   # teal  — H4 dealing-range wall break
CHOCH_COLOR     = "#ff9800"   # orange — trend flip

# Dealing-range walls / equilibrium
DR_FACE = "#3498db"
DR_EDGE = "#5dade2"

# Levels (Phase 2 trade chart)
ENTRY_COLOR = "#e67e22"
SL_COLOR    = "#e74c3c"

# Swing markers (already identical in both renderers — kept here as the source)
SWING_COLOR       = "#d4a017"
SETUP_BREAK_COLOR = "#ffffff"   # bold X on the swing the current setup broke

PRICE_WHITE = "#ffffff"

# --- Candle geometry (CANONICAL = Phase 1's thin-body / fat-wick look) -------
# Phase 2 used body 0.8 / wick 1.2 / alpha 0.9 (squat). We standardise on
# Phase 1's so candles match across both emails.
BODY_W      = 0.55
WICK_W      = 1.5
BODY_ALPHA  = 0.95
MIN_BODY_FR = 0.02   # doji floor: a zero-range body still draws this fraction

# Figure
FIG_WIDTH       = 12
FIG_HEIGHT_BASE = 7.5    # Phase 1 base; grows with DR/zone span up to 11
FIG_HEIGHT_MAX  = 11.0
FIG_HEIGHT_P2   = 5.0    # Phase 2 fixed
SAVE_DPI        = 150


FIG_WIDTH_ZOOM  = 11     # zoomed entry chart is slightly narrower
FIG_HEIGHT_ZOOM = 5.2


def base_canvas(fig_height: float = FIG_HEIGHT_BASE, fig_width: float = FIG_WIDTH):
    """One canvas for every H1 chart. Returns (fig, ax) themed dark.

    Width/height are parameters because the three charts frame differently
    (scout tall-adaptive, context fixed, zoom slightly narrower) — but the dark
    theme (background + spine colours) is unified here so it can never drift.
    """
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), facecolor=BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
    return fig, ax


def adaptive_height(required_range: float, candle_range: float) -> float:
    """Phase 1's adaptive figure height: taller as DR/zone forces a wider y span.

    Kept verbatim from smc_radar.generate_h1_chart so the scout chart's framing
    is unchanged.
    """
    candle_range = max(candle_range, 1e-9)
    ratio = required_range / candle_range
    h = FIG_HEIGHT_BASE
    if ratio > 1.5:
        h = min(FIG_HEIGHT_MAX, FIG_HEIGHT_BASE + (ratio - 1.5) * 1.7)
    return h


# Zoomed-chart candle geometry — INTENTIONALLY wider bodies for the close-up
# entry view (verified design choice in Phase 2's zoomed chart, not drift). The
# COLOURS are shared; only the geometry differs, and only for the zoom.
BODY_W_ZOOM  = 0.70
WICK_W_ZOOM  = 1.6
BODY_ALPHA_ZOOM = 0.92


def draw_candles(ax, O, H, L, C, *, body_w=BODY_W, wick_w=WICK_W,
                 body_alpha=BODY_ALPHA, butt_cap=True):
    """The single candle primitive — thin body, fat wick, butt-capped.

    O/H/L/C are 1-D sequences (numpy arrays or lists) of equal length, already
    sliced to the plot window. x is the positional index 0..n-1 (both renderers
    plot on positional x, not timestamps).

    Geometry defaults to the CANONICAL look (Phase 1 context). The zoomed entry
    chart passes the wider BODY_W_ZOOM/WICK_W_ZOOM on purpose — that close-up is
    meant to show fatter bodies. Colours are always the shared palette, so the
    palette can never drift between charts even when geometry differs.
    """
    n = len(C)
    cap = "butt" if butt_cap else "projecting"
    for i in range(n):
        o, h, l, c = float(O[i]), float(H[i]), float(L[i]), float(C[i])
        if np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c):
            continue
        col = CANDLE_UP if c >= o else CANDLE_DOWN
        ax.plot([i, i], [l, h], color=col, linewidth=wick_w, zorder=2,
                solid_capstyle=cap)
        body = abs(c - o) or (h - l) * MIN_BODY_FR
        ax.add_patch(patches.Rectangle(
            (i - body_w / 2, min(o, c)), body_w, body,
            facecolor=col, linewidth=0, alpha=body_alpha, zorder=3,
        ))


def event_color(tag: str, tier: str) -> str:
    """BOS / Range BOS / CHoCH line colour. Was duplicated in both renderers.

    tag  : 'CHoCH' vs anything-else (BOS family)
    tier : 'Range' marks an H4-wall break (Range BOS)
    """
    if tag == "CHoCH":
        return CHOCH_COLOR
    if tier == "Range":
        return RANGE_BOS_COLOR
    return BOS_COLOR


def fig_to_b64(fig) -> str:
    """Encode a figure to a base64 PNG string and close it. The one save path."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=SAVE_DPI, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return out
