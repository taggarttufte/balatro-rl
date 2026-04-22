"""
V1 -> V8 progression chart for the Balatro RL project README.
Shows per-version ante-reached distribution as stacked horizontal bars:
how many runs died at ante 1, made it to 2-3, 4-5, 6-8, or won.

Data sources:
 - V1, V2, V3: preserved ante distributions from live-training logs/docs
 - V4: INVALIDATED (fixed seeds + broken Burglar joker; sim audit in V6 caught it)
 - V5, V6: approximated from win-rate + known failure modes (no loadable checkpoint
   under the current V7 architecture)
 - V7 Run 4, V7 Runs 5-6, V7 Run 7, V8: fresh 5k-episode stochastic-policy eval
   of each run's best checkpoint (see results/ante_distributions.json)

Styled for GitHub dark-mode display.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ── Palette (matches viz/style.css and project dark theme) ─────────────────
BG          = "#0d1117"
PANEL       = "#151c2a"
TEXT        = "#e6edf3"
TEXT_MUTED  = "#8b95a5"
TEXT_FAINT  = "#5a6478"
GRID        = "#262d3d"

# Milestone colors: muted red -> orange -> gold -> green -> bright green
C_DIED_1    = "#5b4a4a"   # died at ante 1 (muted)
C_DIED_23   = "#a3622b"   # died at ante 2-3
C_DIED_45   = "#c09731"   # died at ante 4-5
C_DIED_68   = "#2f8c4a"   # died at ante 6-8
C_WON       = "#4ade80"   # cleared ante 8 (won)
C_INVALID   = "#6e7681"   # hatched invalidated bar

DATA_PATH = Path(__file__).parent / "ante_distributions.json"
with DATA_PATH.open() as f:
    RAW = json.load(f)

# Rows (keep explicit order so V4 slots between V3 and V5)
ROWS = [
    ("V1",          RAW["V1"],          None),
    ("V2",          RAW["V2"],          None),
    ("V3",          RAW["V3"],          None),
    ("V4",          None,               "INVALIDATED"),   # fixed seeds + Burglar bug
    ("V5",          RAW["V5 (approx)"], None),
    ("V6",          RAW["V6 (approx)"], None),
    ("V7 Run 4",    RAW["V7 Run 4"],    "peak"),
    ("V7 Runs 5-6", RAW["V7 Runs 5-6"], None),
    ("V7 Run 7",    RAW["V7 Run 7"],    None),
    ("V8",          RAW["V8"],          None),
]

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 8), facecolor=BG)
ax.set_facecolor(BG)

y_pos = np.arange(len(ROWS))
bar_h = 0.66

for i, (label, d, tag) in enumerate(ROWS):
    if tag == "INVALIDATED":
        # Hatched placeholder bar, spanning the chart to indicate it exists
        # but has no meaningful distribution data
        ax.barh(
            i, 100, color=C_INVALID, alpha=0.22, height=bar_h,
            hatch="////", edgecolor=TEXT_FAINT, linewidth=0.6,
        )
        ax.text(
            50, i, "V4 INVALIDATED · fixed seeds + broken Burglar joker",
            va="center", ha="center", fontsize=10, style="italic",
            color=TEXT_MUTED,
        )
        continue

    m = d["milestones_pct"]
    segments = [
        (m["died_ante_1"],   C_DIED_1,  "died ante 1"),
        (m["died_ante_2_3"], C_DIED_23, "died ante 2-3"),
        (m["died_ante_4_5"], C_DIED_45, "died ante 4-5"),
        (m["died_ante_6_8"], C_DIED_68, "died ante 6-8"),
        (m["won"],           C_WON,     "won"),
    ]
    left = 0
    for pct, color, _ in segments:
        if pct <= 0:
            continue
        ax.barh(
            i, pct, left=left, color=color, edgecolor=BG,
            linewidth=0.8, height=bar_h,
        )
        left += pct

    # Annotate notable numbers on the right
    anno_parts = [f"reached ante 2+: {100 - m['died_ante_1']:.1f}%"]
    if m["won"] > 0:
        anno_parts.append(f"won: {m['won']:.2f}%")
    elif m["died_ante_6_8"] > 0:
        anno_parts.append(f"reached late: {m['died_ante_6_8']:.1f}%")
    ax.text(
        101, i, "  ·  ".join(anno_parts),
        va="center", fontsize=9.5, color=TEXT_MUTED,
    )

    # Mark the peak run
    if tag == "peak":
        ax.text(
            -0.5, i, "★",
            va="center", ha="right", fontsize=16, color=C_WON,
            transform=ax.get_yaxis_transform(),
        )

# ── Axes ───────────────────────────────────────────────────────────────────
ax.set_yticks(y_pos)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=12, fontweight="bold", color=TEXT)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.tick_params(axis="x", colors=TEXT_MUTED, labelsize=10)
ax.tick_params(axis="y", colors=TEXT, length=0, pad=10)
ax.set_xlabel("Share of episodes (stacked by reached-ante milestone)",
              fontsize=11, color=TEXT_MUTED, labelpad=10)

ax.set_title(
    "Balatro RL · Version Progression by Reached Ante\n"
    "where each version's runs died — not just whether they won",
    fontsize=14, pad=18, fontweight="bold", color=TEXT, loc="left",
)

# ── Legend (milestones) ────────────────────────────────────────────────────
legend_items = [
    Patch(facecolor=C_DIED_1,  label="died at ante 1"),
    Patch(facecolor=C_DIED_23, label="died at ante 2-3"),
    Patch(facecolor=C_DIED_45, label="died at ante 4-5"),
    Patch(facecolor=C_DIED_68, label="died at ante 6-8"),
    Patch(facecolor=C_WON,     label="won (cleared ante 8)"),
]
leg = ax.legend(
    handles=legend_items, loc="lower right",
    bbox_to_anchor=(1.0, -0.23), ncol=5,
    frameon=True, fontsize=10, handlelength=1.4,
    facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
)

# Human-benchmark callout (top-right panel) — moved to avoid overlap with annotations
ax.text(
    0.985, 1.03,
    "skilled human ≈ 70% win rate (off-chart at this scale)",
    transform=ax.transAxes, ha="right", va="bottom",
    fontsize=9, style="italic", color=TEXT_MUTED,
)

# Data source note
ax.text(
    0.0, -0.17,
    "Sources: V1-V3 from preserved training logs · V4 hatched (invalidated in V6 audit) · "
    "V5/V6 approximated from documented failure modes + win rate · "
    "V7/V8 from 5k-episode evaluation of each run's best checkpoint · "
    "N per row in results/ante_distributions.json",
    transform=ax.transAxes, ha="left", va="top",
    fontsize=8.5, style="italic", color=TEXT_FAINT, wrap=True,
)

# ── Spines and grid ────────────────────────────────────────────────────────
ax.grid(axis="x", color=GRID, alpha=0.7, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)

plt.tight_layout()
out = Path(__file__).parent / "v1_v8_progression.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved {out}")
