"""
train.py
PPO training on live Balatro via BalatroEnv.

Usage:
  python train.py               # train from scratch
  python train.py --resume      # resume from latest checkpoint

Before running:
  1. Balatro must be open with BalatroRL mod enabled
  2. The mod will auto-select blind and start playing automatically
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, ".")
from balatro_rl.env import BalatroEnv

# ── Config ───────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR        = Path("logs")
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

PPO_KWARGS = dict(
    policy          = "MlpPolicy",
    learning_rate   = 3e-4,
    n_steps         = 512,
    batch_size      = 64,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    verbose         = 1,
    tensorboard_log = None,
    policy_kwargs   = dict(net_arch=[256, 256, 128]),
)

TOTAL_TIMESTEPS = 100_000

# ── Best-Run Logger ───────────────────────────────────────────────────────────

class BestRunCallback(BaseCallback):
    """
    Tracks every episode. When a new episode-reward record is set,
    saves seed + action sequence to logs/best_runs.jsonl.
    Also appends every episode summary to logs/episode_log.jsonl
    so plot_training.py can visualise progress.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_reward    = -np.inf
        self.episode_count  = 0
        self._ep_actions    = []   # action taken each step
        self._ep_reward     = 0.0
        self._ep_start_step = 0

        self.best_log  = LOG_DIR / "best_runs.jsonl"
        self.ep_log    = LOG_DIR / "episode_log.jsonl"

    def _on_step(self) -> bool:
        # Accumulate action and reward for current episode
        # MultiBinary(9) → actions shape is (n_envs, 9); grab env 0 as a list
        raw = self.locals["actions"]
        action = raw[0].tolist() if hasattr(raw, "__len__") else [int(raw)]
        reward = float(self.locals["rewards"][0])
        done   = bool(self.locals["dones"][0])

        self._ep_actions.append(action)
        self._ep_reward += reward

        if done:
            self.episode_count += 1
            ep_len = len(self._ep_actions)

            # Pull metadata from the underlying env
            env    = self.training_env.envs[0].env  # unwrap Monitor
            seed   = getattr(env, "_last_seed",  "unknown")
            ante   = getattr(env, "_last_ante",  1)
            score  = getattr(env, "_last_score", 0)

            ep_record = {
                "episode":    self.episode_count,
                "timestep":   self.num_timesteps,
                "reward":     round(self._ep_reward, 3),
                "length":     ep_len,
                "seed":       seed,
                "ante":       ante,
                "score":      score,
                "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            # Log every episode
            with self.ep_log.open("a") as f:
                f.write(json.dumps(ep_record) + "\n")

            # Save best run with full action sequence
            if self._ep_reward > self.best_reward:
                self.best_reward = self._ep_reward
                best_record = {**ep_record, "actions": self._ep_actions}
                with self.best_log.open("a") as f:
                    f.write(json.dumps(best_record) + "\n")
                print(f"\n★ New best! Episode {self.episode_count} | "
                      f"reward={self._ep_reward:.1f} | ante={ante} | seed={seed}")

            # Reset episode state
            self._ep_actions    = []
            self._ep_reward     = 0.0
            self._ep_start_step = self.num_timesteps

        return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps",  type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    base_env = BalatroEnv()
    env      = Monitor(base_env, filename=str(LOG_DIR / "monitor"))

    checkpoint_cb = CheckpointCallback(
        save_freq   = 2000,
        save_path   = str(CHECKPOINT_DIR),
        name_prefix = "balatro_ppo",
        verbose     = 1,
    )
    best_run_cb = BestRunCallback(verbose=1)

    # Sort numerically by step count, skipping 'final' which sorts after digits alphabetically
    def _ckpt_step(p):
        try:
            return int(p.stem.split("_")[-2])  # balatro_ppo_114352_steps -> 114352
        except (ValueError, IndexError):
            return -1  # 'final' and malformed names go first, not last

    latest = sorted(CHECKPOINT_DIR.glob("balatro_ppo_*.zip"), key=_ckpt_step)
    if args.resume and latest:
        model_path = str(latest[-1])
        print(f"Resuming from {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("Starting fresh PPO model")
        model = PPO(env=env, **PPO_KWARGS)

    print(f"\nPolicy network:\n{model.policy}\n")
    print(f"Training for {args.steps:,} timesteps...")
    print("=" * 60)

    model.learn(
        total_timesteps     = args.steps,
        callback            = [checkpoint_cb, best_run_cb],
        reset_num_timesteps = not args.resume,
    )

    model.save(str(CHECKPOINT_DIR / "balatro_ppo_final"))
    print("\nTraining complete. Saved to checkpoints/balatro_ppo_final.zip")

if __name__ == "__main__":
    main()
