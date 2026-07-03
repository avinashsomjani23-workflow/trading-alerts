"""Deterministic PNG charts for the backtest email (EMAIL_REBUILD_SPEC §8.5).

Two charts only — equity curve and quarter bars — everything else in the email
is pure HTML table-bars (crisper, no weight). Design constraints:

  * Backend 'Agg' (headless, no display), fixed figsize + dpi, no timestamps in
    metadata (`metadata={}` on savefig) → same input renders the SAME bytes every
    run. The email must reproduce byte-for-byte for audit.
  * Palette locked to §8.1: series blue #2a78d6, negative red #e34948, ink
    #898781, hairline #e1e0d9.
  * Files land next to the HTML (chart_equity.png / chart_quarters.png) so the
    on-disk report renders standalone with relative <img src>. The emailer swaps
    src="chart_equity.png" → src="cid:chart_equity" at send time (§9).

Every caller must ALSO emit alt text carrying the plain numbers — images may be
blocked in Gmail, and the KPI tiles + alt text must carry the story without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless, deterministic — set before pyplot import
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Locked palette (§8.1)
_BLUE   = "#2a78d6"
_RED    = "#e34948"
_INK    = "#898781"
_HAIR   = "#e1e0d9"
_SURFACE = "#fcfcfb"

# Fixed render size: 1280×440 px equity, displayed 640×220 (§8.5).
_DPI = 160
_EQUITY_FIGSIZE = (8.0, 2.75)   # ×160 dpi = 1280×440
_QUARTER_FIGSIZE = (8.0, 2.5)   # ×160 dpi = 1280×400

# Deterministic bytes: no timestamp in PNG metadata.
_SAVE_KW = dict(dpi=_DPI, bbox_inches="tight", pad_inches=0.08,
                facecolor=_SURFACE, metadata={})


def _style_axes(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_HAIR)
    ax.tick_params(colors=_INK, labelsize=8, length=0)
    ax.grid(axis="y", color=_HAIR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def equity_curve_png(
    r_by_fill: List[float],
    out_path: Path,
    quarter_boundaries: Optional[List[int]] = None,
) -> Optional[Path]:
    """Cumulative-R equity curve with drawdown-from-peak shaded.

    `r_by_fill` = per-trade r_realised in FILL-date order (caller sorts).
    `quarter_boundaries` = optional trade-index positions where a new quarter
    starts, drawn as faint vertical hairlines. Returns the path, or None if
    there is nothing to plot.
    """
    if not r_by_fill:
        return None
    equity: List[float] = []
    run = 0.0
    for r in r_by_fill:
        run += float(r)
        equity.append(run)
    x = list(range(len(equity)))
    peak: List[float] = []
    hi = equity[0]
    for v in equity:
        hi = max(hi, v)
        peak.append(hi)

    fig, ax = plt.subplots(figsize=_EQUITY_FIGSIZE)
    _style_axes(ax)
    # Drawdown band: between the running peak and the equity line.
    ax.fill_between(x, equity, peak, color=_RED, alpha=0.12, linewidth=0, zorder=1)
    ax.plot(x, equity, color=_BLUE, linewidth=2.0, zorder=3)
    ax.axhline(0, color=_HAIR, linewidth=0.8, zorder=1)

    if quarter_boundaries:
        for b in quarter_boundaries:
            if 0 < b < len(equity):
                ax.axvline(b, color=_HAIR, linewidth=0.7, zorder=1)

    # Direct-label the final value (no legend box).
    final = equity[-1]
    ax.annotate(f"{'+' if final >= 0 else ''}{final:.1f}R",
                xy=(x[-1], final), xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", color=_INK, fontsize=9, fontweight="bold")
    ax.set_xlim(-0.5, len(equity) - 0.5 + max(2, len(equity) * 0.04))
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}R"))
    fig.savefig(out_path, **_SAVE_KW)
    plt.close(fig)
    return out_path


def quarter_bars_png(
    quarters: List[Tuple[str, float]],
    out_path: Path,
) -> Optional[Path]:
    """Diverging quarterly-total-R bars. `quarters` = [(label, totalR), ...] in
    chronological order. Positive blue, negative red, value labels on each bar.
    """
    if not quarters:
        return None
    labels = [q for q, _ in quarters]
    vals = [float(v) for _, v in quarters]
    colors = [_BLUE if v >= 0 else _RED for v in vals]

    fig, ax = plt.subplots(figsize=_QUARTER_FIGSIZE)
    _style_axes(ax)
    ax.grid(axis="y", color=_HAIR, linewidth=0.6)
    xs = list(range(len(vals)))
    ax.bar(xs, vals, color=colors, width=0.6, zorder=3)
    ax.axhline(0, color="#c3c2b7", linewidth=1.0, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color=_INK, fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}R"))

    pad = max((max(vals) - min(vals)) * 0.08, 0.3)
    for xi, v in zip(xs, vals):
        ax.annotate(f"{'+' if v >= 0 else ''}{v:.1f}R",
                    xy=(xi, v), xytext=(0, 3 if v >= 0 else -12),
                    textcoords="offset points", ha="center",
                    color=_INK, fontsize=8, fontweight="bold")
    lo = min(0.0, min(vals)) - pad
    hi = max(0.0, max(vals)) + pad
    ax.set_ylim(lo, hi)
    fig.savefig(out_path, **_SAVE_KW)
    plt.close(fig)
    return out_path
