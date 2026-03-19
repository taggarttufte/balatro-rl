"""
plot_training.py  —  visualise Balatro RL training progress

Usage:
  python plot_training.py                  # plot and save to logs/training_progress.png
  python plot_training.py --show           # also open interactive window
  python plot_training.py --window 20      # rolling-average window (default 30)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive by default
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

LOG_DIR = Path("logs")
EP_LOG  = LOG_DIR / "episode_log.jsonl"
OUT_PNG = LOG_DIR / "training_progress.png"

def load_episodes():
    if not EP_LOG.exists():
        raise FileNotFoundError(f"No episode log found at {EP_LOG}. Run train.py first.")
    records = []
    with EP_LOG.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def rolling(values, window):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(np.mean(values[start:i+1]))
    return np.array(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show",   action="store_true", help="Open interactive window")
    parser.add_argument("--window", type=int, default=30, help="Rolling average window")
    args = parser.parse_args()

    eps = load_episodes()
    n   = len(eps)
    print(f"Loaded {n} episodes")

    episodes   = [e["episode"]  for e in eps]
    rewards    = [e["reward"]   for e in eps]
    lengths    = [e["length"]   for e in eps]
    antes      = [e["ante"]     for e in eps]
    timesteps  = [e["timestep"] for e in eps]

    W = args.window
    roll_rew = rolling(rewards, W)
    roll_len = rolling(lengths, W)

    # ── Figure: 3 panels ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Balatro RL — Training Progress", fontsize=14, fontweight="bold")

    ep_arr = np.array(episodes)

    # Panel 1: Episode reward + rolling avg
    ax = axes[0]
    ax.plot(ep_arr, rewards,  alpha=0.25, color="#4C9BE8", linewidth=0.8, label="Episode reward")
    ax.plot(ep_arr, roll_rew, color="#1A6BB5", linewidth=2,   label=f"Rolling avg ({W} ep)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Total Reward")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Reward per Episode")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # Panel 2: Ante reached (progress proxy)
    ax = axes[1]
    ax.scatter(ep_arr, antes, alpha=0.3, s=10, color="#E87C4C", label="Ante reached")
    ax.plot(ep_arr, rolling(antes, W), color="#B5521A", linewidth=2, label=f"Rolling avg ({W} ep)")
    ax.set_ylabel("Ante Reached")
    ax.set_ylim(0.5, 9)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Ante Reached per Episode")

    # Panel 3: Episode length (hands played)
    ax = axes[2]
    ax.plot(ep_arr, lengths,  alpha=0.25, color="#7CBB6E", linewidth=0.8, label="Steps per episode")
    ax.plot(ep_arr, roll_len, color="#3A8A2B", linewidth=2,   label=f"Rolling avg ({W} ep)")
    ax.set_ylabel("Steps (hands/discards)")
    ax.set_xlabel("Episode")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Episode Length")

    # Secondary x-axis: timesteps
    ax2 = axes[2].twiny()
    ax2.set_xlim(axes[2].get_xlim())
    ts_ticks = np.linspace(0, n-1, min(6, n), dtype=int)
    ax2.set_xticks(ts_ticks)
    ax2.set_xticklabels([f"{timesteps[i]//1000}k" for i in ts_ticks], fontsize=7)
    ax2.set_xlabel("Timesteps", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUT_PNG}")

    # Print summary stats
    print(f"\n{'─'*40}")
    print(f"  Total episodes : {n}")
    print(f"  Best reward    : {max(rewards):.2f}  (ep {episodes[rewards.index(max(rewards))]})")
    print(f"  Best ante      : {max(antes)}  (ep {episodes[antes.index(max(antes))]})")
    print(f"  Avg reward     : {np.mean(rewards):.2f}  (last {W}: {roll_rew[-1]:.2f})")
    print(f"  Avg ante       : {np.mean(antes):.2f}  (last {W}: {rolling(antes,W)[-1]:.2f})")
    print(f"  Avg ep length  : {np.mean(lengths):.1f}  (last {W}: {roll_len[-1]:.1f})")
    print(f"{'─'*40}")

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()

if __name__ == "__main__":
    main()
