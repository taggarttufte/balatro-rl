"""
V1 -> V8 progression chart for the Balatro RL project README / Featured section.
Data sourced from results/PROJECT_RETROSPECTIVE.md cross-version summary table.
"""
import matplotlib.pyplot as plt
import numpy as np

# (label, peak_wr_pct, annotation, color_category)
# V4 excluded from the chart: "inflated" win rates were memorization + sim bugs, not legitimate
versions = [
    ("V1-V3",        0.01, "Live-game IPC\n<0.01% (RAM leak, ~14 sps)",                    "gray"),
    ("V5",           0.00, "Dual-agent split\nShop starvation (12 failed runs)",            "firebrick"),
    ("V6",           1.90, "Single-agent + combo ranker\nFirst legitimate result",          "darkorange"),
    ("V7 Run 4",     2.35, "Hierarchical intent + card-subset head\nPEAK -- 2.35%",         "seagreen"),
    ("V7 Runs 5-6",  2.15, "Reward-shape retunes\nPlateau confirmed (2.07-2.23%)",          "mediumseagreen"),
    ("V7 Run 7",     0.16, "5.5x network scaling (13.6M params)\nSame plateau, killed early","steelblue"),
    ("V8 Runs 1-4",  0.00, "Self-play multiplayer\nMigration + symmetry failures",          "slategray"),
]

labels       = [v[0] for v in versions]
vals         = [v[1] for v in versions]
annotations  = [v[2] for v in versions]
colors       = [v[3] for v in versions]

fig, ax = plt.subplots(figsize=(11, 6.5))
y_pos = np.arange(len(versions))
bars = ax.barh(y_pos, vals, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)

# Highlight V7 Run 4 (the peak)
bars[3].set_edgecolor('black')
bars[3].set_linewidth(2.0)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel("Peak Solo Win Rate (%)", fontsize=11)
ax.set_title("Balatro RL -- Version Progression\n"
             "8 architecture iterations, ~366 GPU-hours, RTX 3080 Ti",
             fontsize=13, pad=15, fontweight='bold')

# Per-bar annotation to the right
for i, (v, ann) in enumerate(zip(vals, annotations)):
    x_pos = v + 0.05 if v > 0.02 else 0.05
    ax.text(x_pos, i, ann, va='center', fontsize=8.5, color='#222')

# Reference lines (comparison benchmarks)
ax.axvline(x=0.01, color='gray',      linestyle=':',  alpha=0.7, linewidth=1.2)
ax.text(0.012, -0.55, "random play (~0.01%)", fontsize=8, color='gray', style='italic')

# Baselines callout (text only -- 70% human would blow up the x-axis scale)
ax.text(0.99, 0.02, "For scale: skilled human ~70% win rate",
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=8.5, style='italic', color='#555',
        bbox=dict(boxstyle='round,pad=0.35', fc='#fafafa', ec='#ccc'))

# V4 disclaimer (important for honesty)
ax.text(0.01, -0.14,
        "V4 (Python-sim pivot) excluded: reported win rates were memorization artifacts "
        "from fixed seeds + 30% joker bugs (audited and fixed in V6).",
        transform=ax.transAxes, ha='left', va='top',
        fontsize=7.5, style='italic', color='#666', wrap=True)

ax.set_xlim(0, 3.1)
ax.grid(axis='x', alpha=0.25)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = 'v1_v8_progression.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f"Saved {out}")
