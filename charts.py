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
BOS_COLOR       = "#e91e63"   # magenta — internal swing break
RANGE_BOS_COLOR = "#00897b"   # teal  — H4 dealing-range wall break
CHOCH_COLOR     = "#ff9800"   # orange — trend flip

# Dealing-range walls / equilibrium
DR_FACE = "#3498db"
DR_EDGE = "#5dade2"

# PD/PW liquidity pool levels (prior-day / prior-week high & low).
# One violet family, distinct from every other palette entry (OB purple is
# redder, DR blue is cyan-blue): D1 = softer periwinkle, W1 = bolder violet.
# Weekly is the higher-tier level so it draws thicker (POOL_LW_W1 > POOL_LW_D1).
POOL_D1_COLOR = "#b0a8ff"   # PDH / PDL — prior-day high & low
POOL_W1_COLOR = "#c8a2ff"   # PWH / PWL — prior-week high & low
POOL_LW_D1    = 1.0
POOL_LW_W1    = 1.4

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
FIG_HEIGHT_ZOOM = 8.0


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


# PD/PW pool line spec: (snapshot-key, label, tier). Order = draw order.
# Tier picks color + linewidth from the constants above.
_POOL_LINES = (
    ("pdh", "PDH", "D1"),
    ("pdl", "PDL", "D1"),
    ("pwh", "PWH", "W1"),
    ("pwl", "PWL", "W1"),
)


def draw_pool_lines(ax, pools, *, x_right, in_view, zorder=2):
    """Draw the PD/PW liquidity levels as leftmost-labelled dotted lines.

    Shared by Phase 1 (scout) and Phase 2 (trade) so the pool levels can
    never visually drift between the two emails — same colours, same dotted
    style, same D1/W1 tier weighting, same left-edge label.

    Args:
      pools   : the ``pools`` sub-dict of a pool_builder snapshot, i.e.
                ``{"pdh": {"level": float|None, ...}, "pdl": {...}, ...}``.
                None / empty is a no-op (feed degrade — chart still renders).
      x_right : plot x-coordinate of the right edge (where labels would clash
                with candles); labels are placed at the LEFT edge (x=0) so
                they read cleanly against the oldest candles.
      in_view : callable(price) -> bool. Only levels inside the caller's
                proximity band are drawn, so a stale far level can't stretch
                the y-axis or leave a stray line (mirrors the DR-wall gate).

    Returns the number of lines actually drawn (for logging / tests).
    """
    if not pools:
        return 0
    drawn = 0
    for key, label, tier in _POOL_LINES:
        pool = pools.get(key)
        if not pool:
            continue
        level = pool.get("level")
        if level is None:
            continue
        try:
            level = float(level)
        except (TypeError, ValueError):
            continue
        if not in_view(level):
            continue
        color = POOL_D1_COLOR if tier == "D1" else POOL_W1_COLOR
        lw    = POOL_LW_D1 if tier == "D1" else POOL_LW_W1
        ax.axhline(y=level, color=color, linewidth=lw, linestyle=":",
                   alpha=0.7, zorder=zorder)
        # Left-edge label: "D1 PDH" so the vet sees tier + which edge at a
        # glance. Boxed in the chart bg so it stays legible over candles.
        ax.text(0, level, f"  {tier} {label}",
                color=color, fontsize=7, fontweight="bold",
                ha="left", va="center", zorder=zorder + 5,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1.0, alpha=0.78))
        drawn += 1
    return drawn


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
