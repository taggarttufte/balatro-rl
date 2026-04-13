"""
env_mp.py — Multiplayer Balatro environment for V8 self-play training.

Wraps two V7-style agents playing against each other via MultiplayerBalatro.
Each player gets the standard V7 observation (434 dims) and hierarchical action
space (intent + card subset for SELECTING_HAND, phase actions for blind/shop).

The multiplayer coordinator handles:
  - Same-seed card draws for both players
  - Lives system (4 per player)
  - HOUSE RULE: regular blind failures cost a life
  - PvP blind resolution (higher score wins, loser loses life + comeback money)
  - Game termination on 0 lives

Reward structure per player:
  - All V7 rewards (card quality, synergy, blind clear, etc.)
  - NEW: PvP win: +3.0 | PvP loss: -2.0 | PvP tie: 0
  - NEW: Game win (opponent hit 0 lives): +20.0 | Game loss: -10.0

API:
  env = MultiplayerBalatroEnv(seed=42)
  p1_obs, p2_obs = env.reset()
  # Both players act simultaneously each step
  (p1_obs, p2_obs), (p1_reward, p2_reward), done, info = env.step(p1_action, p2_action)
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from .mp_game import MultiplayerBalatro, MPPhase
from .game import State
from .env_v7 import (
    BalatroV7Env, OBS_DIM, N_PHASE_ACTIONS, N_INTENTS, N_HAND_SLOTS,
    PHASE_SELECTING_HAND, PHASE_BLIND_SELECT, PHASE_SHOP, PHASE_GAME_OVER,
)
from .card_selection import INTENT_PLAY, INTENT_DISCARD, INTENT_USE_CONSUMABLE
from .shop import set_banned_jokers


# Standard Ranked multiplayer ruleset disables these jokers because they
# interact with boss blinds (which are now PvP blinds in multiplayer):
#   - Chicot: disables boss blind effect (irrelevant in PvP)
#   - Matador: $8 when boss triggers (irrelevant in PvP)
#   - Mr. Bones: prevents loss at 25% (broken in lives system)
#   - Luchador: sell to disable boss blind (irrelevant in PvP)
MULTIPLAYER_BANNED_JOKERS = {
    "j_chicot",
    "j_matador",
    "j_mr_bones",
    "j_luchador",
}

# Apply ban list at module import time so any MP env construction inherits it.
# This is global state — if you also use single-player envs in the same process,
# call clear_banned_jokers() before doing so.
set_banned_jokers(MULTIPLAYER_BANNED_JOKERS)


# V8 multiplayer-specific rewards (added on top of V7 rewards)
R_PVP_WIN     = 3.0
R_PVP_LOSS    = -2.0
R_GAME_WIN    = 20.0   # opponent hit 0 lives before you
R_GAME_LOSS   = -10.0  # you hit 0 lives first
R_LIFE_LOST   = -1.5   # penalty each time a life is lost (both players)


class _PlayerEnvProxy:
    """
    A thin proxy that looks like a V7 env but shares the underlying BalatroGame
    with the MP coordinator. Reuses V7's obs encoding and reward logic.

    We don't subclass BalatroV7Env directly because we need to point it at a
    game instance owned by the MP coordinator, not create its own.
    """
    def __init__(self, mp_coord: MultiplayerBalatro, player: int):
        self.mp = mp_coord
        self.player = player
        # Create a V7 env as a helper, then swap its game for the MP coord's game
        self._v7 = BalatroV7Env(seed=mp_coord._seed)
        self._v7.game = mp_coord.get_player_game(player)
        # Re-initialize tracking state
        self._v7._prev_progress = 0.0
        self._v7._prev_ante = 1
        self._v7._prev_blind_idx = 0
        self._v7._steps = 0
        self._v7._play_history = []
        self._v7._episode_reward = 0.0
        self._v7._joker_acquisition_ante = {}

    @property
    def game(self):
        return self._v7.game

    def encode_obs(self) -> np.ndarray:
        return self._v7._encode_obs()

    def get_intent_mask(self) -> np.ndarray:
        return self._v7.get_intent_mask()

    def get_phase_mask(self) -> np.ndarray:
        return self._v7.get_phase_mask()

    def get_phase(self) -> int:
        return self._v7.get_phase()

    def step_hand(self, intent: int, subset: tuple):
        return self._v7.step_hand(intent, subset)

    def step_phase(self, action: int):
        return self._v7.step_phase(action)

    def auto_advance(self):
        self._v7._auto_advance()


class MultiplayerBalatroEnv:
    """
    Self-play environment for V8. Two V7 agents play against each other.

    Both players step simultaneously. After each step, we check if the
    underlying MP coordinator should transition phases (e.g. both finished
    a blind -> resolve lives/PvP).
    """

    def __init__(self, seed: Optional[int] = None, lives: int = 4):
        self._seed = seed
        self._lives = lives
        self.mp: Optional[MultiplayerBalatro] = None
        self.p1: Optional[_PlayerEnvProxy] = None
        self.p2: Optional[_PlayerEnvProxy] = None
        self._episode_reward = [0.0, 0.0]  # cumulative per player

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self._seed = seed
        self.mp = MultiplayerBalatro(seed=self._seed, lives=self._lives)
        self.p1 = _PlayerEnvProxy(self.mp, 1)
        self.p2 = _PlayerEnvProxy(self.mp, 2)
        # Auto-advance both games to first decision point
        self.p1.auto_advance()
        self.p2.auto_advance()
        self._episode_reward = [0.0, 0.0]
        return self.p1.encode_obs(), self.p2.encode_obs()

    # ─────────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────────

    def step(self, p1_action: dict, p2_action: dict):
        """
        Step both players simultaneously.

        Each action is a dict:
          SELECTING_HAND:  {"type": "hand", "intent": int, "subset": tuple}
          Other phases:    {"type": "phase", "action": int}

        Returns:
            (obs_tuple, reward_tuple, done, info)
        """
        p1_reward = self._apply_action(self.p1, p1_action)
        p2_reward = self._apply_action(self.p2, p2_action)

        # After actions, check if both players have finished a blind and
        # need MP-level resolution
        mp_rewards = self._check_blind_resolution()
        p1_reward += mp_rewards[0]
        p2_reward += mp_rewards[1]

        self._episode_reward[0] += p1_reward
        self._episode_reward[1] += p2_reward

        done = self.mp.game_over
        info = {
            "p1_lives": self.mp.p1_lives,
            "p2_lives": self.mp.p2_lives,
            "p1_ante": self.mp.p1_game.ante,
            "p2_ante": self.mp.p2_game.ante,
            "phase": self.mp.phase.name,
            "winner": self.mp.winner,
            "p1_total_reward": self._episode_reward[0],
            "p2_total_reward": self._episode_reward[1],
        }
        return (
            (self.p1.encode_obs(), self.p2.encode_obs()),
            (p1_reward, p2_reward),
            done,
            info,
        )

    def _apply_action(self, player_env: _PlayerEnvProxy, action: dict) -> float:
        """Execute one player's action and return their reward delta."""
        game = player_env.game
        # Skip if game already over (e.g. one player is waiting for the other)
        if game.state == State.GAME_OVER:
            return 0.0

        action_type = action.get("type")
        if action_type == "hand":
            obs, reward, term, trunc, info = player_env.step_hand(
                action["intent"], action["subset"]
            )
        elif action_type == "phase":
            obs, reward, term, trunc, info = player_env.step_phase(action["action"])
        else:
            reward = 0.0
        return reward

    # ─────────────────────────────────────────────────────────────────────
    # Phase coordination
    # ─────────────────────────────────────────────────────────────────────

    def _check_blind_resolution(self) -> tuple[float, float]:
        """
        After both players step, check if we need to resolve a blind at the MP level.

        A blind is "resolved" when both players have either:
          - Cleared the current blind (score >= target)
          - Run out of hands without clearing (failure)
          - Reached GAME_OVER (individual game ended)

        When both are done with current blind, apply MP rules (life loss, PvP).
        Returns (p1_reward, p2_reward) for MP-level events.
        """
        p1, p2 = self.mp.p1_game, self.mp.p2_game

        # Only resolve if both players are "done" with the current blind
        p1_done = self._is_blind_done(p1)
        p2_done = self._is_blind_done(p2)
        if not (p1_done and p2_done):
            return (0.0, 0.0)

        p1_reward = 0.0
        p2_reward = 0.0

        # Determine which blind we're resolving based on blind_idx
        # blind_idx: 0=Small, 1=Big, 2=Boss (PvP)
        p1_cleared = p1.chips_scored >= p1.current_blind.chips_target
        p2_cleared = p2.chips_scored >= p2.current_blind.chips_target

        phase = self.mp.phase
        if phase == MPPhase.SMALL_BLIND:
            self._resolve_pre_pvp_blind(p1_cleared, p2_cleared, is_small=True)
            # Lives lost get a penalty signal
            if not p1_cleared:
                p1_reward += R_LIFE_LOST
            if not p2_cleared:
                p2_reward += R_LIFE_LOST
            self.mp.phase = MPPhase.BIG_BLIND if not self.mp.game_over else self.mp.phase

        elif phase == MPPhase.BIG_BLIND:
            self._resolve_pre_pvp_blind(p1_cleared, p2_cleared, is_small=False)
            if not p1_cleared:
                p1_reward += R_LIFE_LOST
            if not p2_cleared:
                p2_reward += R_LIFE_LOST
            self.mp.phase = MPPhase.PVP_BLIND if not self.mp.game_over else self.mp.phase

        elif phase == MPPhase.PVP_BLIND:
            # PvP: compare scores directly (no target requirement in our simple model)
            p1_score = p1.chips_scored
            p2_score = p2.chips_scored
            p1_before_lives = self.mp.p1_lives
            p2_before_lives = self.mp.p2_lives
            self.mp.resolve_pvp_blind(p1_score=p1_score, p2_score=p2_score)
            if self.mp.p1_lives < p1_before_lives:
                p1_reward += R_PVP_LOSS + R_LIFE_LOST
            if self.mp.p2_lives < p2_before_lives:
                p2_reward += R_PVP_LOSS + R_LIFE_LOST
            if p1_score > p2_score:
                p1_reward += R_PVP_WIN
            elif p2_score > p1_score:
                p2_reward += R_PVP_WIN
            # Advance to next ante if game isn't over
            if not self.mp.game_over:
                self.mp.advance_to_next_ante()

        # Game-over bonus
        if self.mp.game_over:
            winner = self.mp.winner
            if winner == 1:
                p1_reward += R_GAME_WIN
                p2_reward += R_GAME_LOSS
            elif winner == 2:
                p2_reward += R_GAME_WIN
                p1_reward += R_GAME_LOSS
            # Draw: no bonus

        return (p1_reward, p2_reward)

    def _resolve_pre_pvp_blind(self, p1_cleared: bool, p2_cleared: bool, is_small: bool):
        """
        Resolve small or big blind. HOUSE RULE: failure costs a life.
        Also advance each player's individual game to the next blind.
        """
        # Apply life penalties first
        if not p1_cleared:
            self.mp.apply_blind_failure(1)
        if not p2_cleared:
            self.mp.apply_blind_failure(2)

        # For each player, manually advance their game past this blind:
        # - If cleared: the BalatroGame already transitioned to ROUND_EVAL -> SHOP
        # - If failed: the BalatroGame transitioned to GAME_OVER. We need to
        #   "revive" them (house rule: they still get to keep playing on next blind
        #   because they have a life system, but the underlying BalatroGame
        #   considers itself done)
        self._revive_if_needed(self.mp.p1_game, p1_cleared)
        self._revive_if_needed(self.mp.p2_game, p2_cleared)

    def _revive_if_needed(self, game, cleared: bool):
        """
        If a player's game thinks it's GAME_OVER but they still have lives,
        reset the game state to continue to next blind.

        This is a hack — BalatroGame doesn't know about lives. We force-advance
        it to the next blind as if they'd cleared, but without awarding points
        or money for clearing.
        """
        if cleared:
            return  # game naturally advanced to SHOP
        if game.state != State.GAME_OVER:
            return
        # Force progression: set state back to BLIND_SELECT and advance blind_idx
        from .game import BlindInfo
        from .constants import BLIND_CHIPS
        game.blind_idx += 1
        if game.blind_idx >= 3:
            # Wrapped past boss — increment ante
            game.ante += 1
            game.blind_idx = 0
        # Reset blind state (like _start_blind but without the gameplay)
        game.chips_scored = 0
        game.hands_left = game.base_hands
        game.discards_left = game.base_discards
        game.hand = []
        game.deck = []  # re-initialized on play_blind
        game.played_hand_types_this_round = set()
        # Set up next blind info
        blind_kinds = ["Small", "Big", "Boss"]
        kind = blind_kinds[game.blind_idx]
        chips = BLIND_CHIPS.get(game.ante, (300, 450, 600))[game.blind_idx]
        is_boss = (game.blind_idx == 2)
        game.current_blind = BlindInfo("", kind, chips, is_boss=is_boss)
        # Back to BLIND_SELECT
        game.state = State.BLIND_SELECT

    def _is_blind_done(self, game) -> bool:
        """A player's current blind is "done" when they've scored enough, run out
        of hands, or reached game over."""
        if game.state == State.GAME_OVER:
            return True
        if game.state == State.SHOP:
            # Cleared the blind
            return True
        if game.state == State.SELECTING_HAND and game.hands_left <= 0:
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    # Player-specific helpers
    # ─────────────────────────────────────────────────────────────────────

    def get_phase(self, player: int) -> int:
        return (self.p1 if player == 1 else self.p2).get_phase()

    def get_intent_mask(self, player: int) -> np.ndarray:
        return (self.p1 if player == 1 else self.p2).get_intent_mask()

    def get_phase_mask(self, player: int) -> np.ndarray:
        return (self.p1 if player == 1 else self.p2).get_phase_mask()
