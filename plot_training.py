"""
plot_training.py  —  visualise Balatro RL training progress

Usage:
  python plot_training.py                  # plot and save
  python plot_training.py --show           # also open interactive window
  python plot_training.py --window 50      # rolling-average window (default 50)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

LOG_DIR = Path("logs")
EP_LOG  = LOG_DIR / "episode_log.jsonl"
OUT_PNG = LOG_DIR / "training_progress.png"

STYLE = {
    "figure.facecolor":  "#0F1117",
    "axes.facecolor":    "#1A1D27",
    "axes.edgecolor":    "#3A3D4D",
    "axes.labelcolor":   "#C8CDD8",
    "text.color":        "#C8CDD8",
    "xtick.color":       "#8A8FA0",
    "ytick.color":       "#8A8FA0",
    "grid.color":        "#2A2D3D",
    "grid.linewidth":    0.5,
    "legend.facecolor":  "#1A1D27",
    "legend.edgecolor":  "#3A3D4D",
    "legend.labelcolor": "#C8CDD8",
}

BLUE   = "#4C9BE8"
BLUE_D = "#1A6BB5"
ORG    = "#E8944C"
ORG_D  = "#B5621A"
GRN    = "#6EBB7C"
GRN_D  = "#2B8A3A"
RED    = "#E8504C"
GOLD   = "#F5C542"


def load_episodes():
    records = []
    with EP_LOG.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    # Filter out corrupted Endless Mode episodes (ante > 8)
    clean = [r for r in records if r.get("ante", 1) <= 8]
    n_dropped = len(records) - len(clean)
    if n_dropped:
        print(f"  (dropped {n_dropped} corrupted Endless Mode episodes)")
    return clean


def rolling(values, window):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(np.mean(values[start:i+1]))
    return np.array(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show",   action="store_true")
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    eps = load_episodes()
    n   = len(eps)
    print(f"Loaded {n} episodes")

    # Use sequential log index as x-axis — episode numbers and timesteps both
    # reset to 0 on each training restart, making them non-monotonic.
    idx       = np.arange(len(eps))
    rewards   = np.array([e["reward"]   for e in eps])
    lengths   = np.array([e["length"]   for e in eps])
    antes     = np.array([e["ante"]     for e in eps])
    raw_ts    = np.array([e["timestep"] for e in eps])

    # Build cumulative timesteps across restarts
    cum_ts = np.zeros(len(eps), dtype=int)
    offset = 0
    for i in range(len(eps)):
        if i > 0 and raw_ts[i] < raw_ts[i - 1]:
            offset += raw_ts[i - 1]
        cum_ts[i] = raw_ts[i] + offset

    W        = args.window
    roll_rew = rolling(rewards, W)
    roll_len = rolling(lengths, W)
    roll_ant = rolling(antes,   W)

    episodes  = idx   # alias for annotation code below

    # Best episodes to annotate
    best_idx = int(np.argmax(rewards))
    best_ant_idx = int(np.argmax(antes))

    # ── Layout ────────────────────────────────────────────────────────────────
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14, 12))
        fig.suptitle("Balatro RL — Training Progress", fontsize=15,
                     fontweight="bold", color="#E8E8F0", y=0.98)

        gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32,
                              left=0.07, right=0.96, top=0.93, bottom=0.06)
        ax_rew  = fig.add_subplot(gs[0, :])   # full width
        ax_ante = fig.add_subplot(gs[1, :])   # full width
        ax_len  = fig.add_subplot(gs[2, 0])
        ax_dist = fig.add_subplot(gs[2, 1])

        # ── Panel 1: Reward ───────────────────────────────────────────────
        ax_rew.plot(idx, rewards, alpha=0.18, color=BLUE, linewidth=0.7)
        ax_rew.plot(idx, roll_rew, color=BLUE_D, linewidth=2.2,
                    label=f"Rolling avg ({W} ep)")
        ax_rew.axhline(0, color="#555566", linewidth=0.8, linestyle="--")

        # Annotate best reward
        ax_rew.scatter([idx[best_idx]], [rewards[best_idx]],
                       color=GOLD, zorder=5, s=60)
        ax_rew.annotate(f"Best: {rewards[best_idx]:.1f}",
                        xy=(idx[best_idx], rewards[best_idx]),
                        xytext=(20, -18), textcoords="offset points",
                        fontsize=8, color=GOLD,
                        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2))

        ax_rew.set_ylabel("Total Reward", fontsize=10)
        ax_rew.set_title("Reward per Episode", fontsize=11, pad=6)
        ax_rew.legend(loc="upper left", fontsize=8)
        ax_rew.grid(True)
        ax_rew.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

        # ── Panel 2: Ante reached ─────────────────────────────────────────
        cmap = plt.get_cmap("RdYlGn")
        colors = [cmap((a - 1) / 7) for a in antes]
        ax_ante.scatter(idx, antes, alpha=0.35, s=8, color=colors)
        ax_ante.plot(idx, roll_ant, color=ORG_D, linewidth=2.2,
                     label=f"Rolling avg ({W} ep)")

        if best_ant_idx != best_idx:
            ax_ante.scatter([idx[best_ant_idx]], [antes[best_ant_idx]],
                            color=GOLD, zorder=5, s=60)
            ax_ante.annotate(f"Ante {antes[best_ant_idx]}",
                             xy=(idx[best_ant_idx], antes[best_ant_idx]),
                             xytext=(15, 10), textcoords="offset points",
                             fontsize=8, color=GOLD,
                             arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2))

        ax_ante.set_ylabel("Ante Reached", fontsize=10)
        ax_ante.set_title("Ante Reached per Episode  (green = deeper)", fontsize=11, pad=6)
        ax_ante.set_ylim(0.5, 8.5)
        ax_ante.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax_ante.legend(loc="upper left", fontsize=8)
        ax_ante.grid(True)

        # ── Panel 3: Episode length ───────────────────────────────────────
        ax_len.plot(idx, lengths, alpha=0.18, color=GRN, linewidth=0.7)
        ax_len.plot(idx, rolling(lengths, W), color=GRN_D, linewidth=2.2,
                    label=f"Rolling avg ({W} ep)")
        ax_len.set_ylabel("Steps", fontsize=9)
        ax_len.set_xlabel("Log index (sequential)", fontsize=9)
        ax_len.set_title("Episode Length (steps)", fontsize=10, pad=6)
        ax_len.legend(loc="upper left", fontsize=7)
        ax_len.grid(True)

        # Secondary x-axis: cumulative timesteps
        ax2 = ax_len.twiny()
        ax2.set_xlim(ax_len.get_xlim())
        tidx = np.linspace(0, n - 1, min(6, n), dtype=int)
        ax2.set_xticks(idx[tidx])
        ax2.set_xticklabels([f"{cum_ts[i]//1000}k" for i in tidx], fontsize=7)
        ax2.set_xlabel("Cumulative Timesteps", fontsize=8, labelpad=4)

        # ── Panel 4: Ante distribution ────────────────────────────────────
        ante_counts = {}
        for a in antes:
            ante_counts[a] = ante_counts.get(a, 0) + 1
        ante_labels = sorted(ante_counts)
        ante_vals   = [ante_counts[a] for a in ante_labels]
        bar_colors  = [cmap((a - 1) / 7) for a in ante_labels]

        bars = ax_dist.bar(ante_labels, ante_vals, color=bar_colors,
                           edgecolor="#3A3D4D", linewidth=0.8)
        for bar, val in zip(bars, ante_vals):
            pct = 100 * val / n
            ax_dist.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(ante_vals) * 0.01,
                         f"{pct:.1f}%", ha="center", va="bottom",
                         fontsize=7.5, color="#C8CDD8")

        ax_dist.set_xlabel("Ante Reached", fontsize=9)
        ax_dist.set_ylabel("Episodes", fontsize=9)
        ax_dist.set_title("Ante Distribution (all runs)", fontsize=10, pad=6)
        ax_dist.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax_dist.grid(True, axis="y")

        plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved to {OUT_PNG}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "-"*44)
    print(f"  Episodes         : {n}")
    print(f"  Cumulative steps : {cum_ts[-1]:,}")
    print(f"  Best reward      : {rewards.max():.2f}  (log idx {best_idx})")
    print(f"  Best ante        : {int(antes.max())}  (log idx {best_ant_idx})")
    print(f"  Avg reward       : {rewards.mean():.3f}")
    print(f"  Last {W} avg rew  : {roll_rew[-1]:.3f}")
    print(f"  Ante 2+ rate     : {100*(antes>=2).mean():.1f}%")
    print(f"  Ante 3+ rate     : {100*(antes>=3).mean():.2f}%")
    print("-"*44)

    if args.show:
        import subprocess, sys
        subprocess.Popen(["start", str(OUT_PNG)], shell=True)


if __name__ == "__main__":
    main()
