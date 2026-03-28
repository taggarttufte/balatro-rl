"""
env_socket.py â€” v3 Balatro RL environment using socket-based IPC.

Protocol:
  - Lua listens on TCP port (5000 + instance_id), default 5009 for testing
  - Python connects as client
  - Lua sends full state JSON (newline-terminated) on every game event
  - Python sends action JSON (newline-terminated) to execute
  - Messages: newline-delimited JSON

This replaces the file-polling IPC of env_parallel.py.
"""

import json
import socket
import time
import gymnasium as gym
import numpy as np
from balatro_rl.state_v2 import parse_state, state_to_obs, OBS_SIZE
from balatro_rl.action_v2 import action_to_cards_and_type, generate_action_mask

ACTION_SIZE = 20  # Discrete(20) -- same as env_parallel

DEFAULT_INSTANCE = 9
BASE_PORT = 5000  # port = BASE_PORT + instance_id


class BalatroSocketEnv(gym.Env):
    """
    Balatro RL environment with socket-based IPC (v3).

    Connect/disconnect is handled automatically. If the socket drops
    Python will attempt to reconnect on the next reset().
    """

    metadata = {"render_modes": []}

    def __init__(self, instance_id: int = DEFAULT_INSTANCE, timeout: float = 60.0):
        super().__init__()
        self.instance_id = instance_id
        self.port = BASE_PORT + instance_id
        self.timeout = timeout

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(ACTION_SIZE)

        self._sock: socket.socket | None = None
        self._buf = ""
        self._episode_seed = None
        self._connected = False

    # â”€â”€ Connection management â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _connect(self, retries: int = 30, retry_delay: float = 1.0) -> bool:
        """Connect to the Lua socket server. Returns True on success."""
        for attempt in range(retries):
            try:
                # Use IPv6 - Lua's bind("*") only binds IPv6 on Windows
                s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
                s.settimeout(self.timeout)
                s.connect(("::1", self.port))
                self._sock = s
                self._buf = ""
                self._connected = True
                print(f"[env_socket:{self.instance_id}] Connected on port {self.port}")
                return True
            except (ConnectionRefusedError, OSError) as e:
                if attempt % 5 == 0:
                    print(f"[env_socket:{self.instance_id}] Waiting for Lua server (attempt {attempt+1}/{retries})...")
                time.sleep(retry_delay)
        print(f"[env_socket:{self.instance_id}] Failed to connect after {retries} attempts")
        return False

    def _disconnect(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        self._connected = False
        self._buf = ""

    # â”€â”€ Message I/O â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _recv_line(self) -> str | None:
        """Read one newline-terminated message from the socket."""
        deadline = time.time() + self.timeout
        while True:
            if "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                return line
            if time.time() > deadline:
                return None
            if self._sock is None:
                return None
            try:
                chunk = self._sock.recv(65536).decode("utf-8", errors="replace")
                if not chunk:
                    # Connection closed
                    self._disconnect()
                    return None
                self._buf += chunk
            except socket.timeout:
                return None
            except OSError:
                self._disconnect()
                return None

    def _send_action(self, action_type: str, card_indices: list[int]) -> bool:
        """Send an action JSON message to Lua."""
        if self._sock is None:
            return False
        msg = json.dumps({"action": action_type, "card_indices": card_indices}) + "\n"
        try:
            self._sock.sendall(msg.encode("utf-8"))
            return True
        except OSError:
            self._disconnect()
            return False

    # Actionable game states -- states where the env needs to act or terminate
    _ACTIONABLE = frozenset({
        "selecting_hand", "shop", "blind_select",
        "round_eval", "game_over", "new_round",
    })

    def _read_state(self) -> dict | None:
        """Block until we get any actionable game state from Lua."""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            line = self._recv_line()
            if line is None:
                time.sleep(0.003)
                continue
            try:
                gs = json.loads(line)
                if gs.get("event") in self._ACTIONABLE:
                    return gs
            except json.JSONDecodeError:
                continue
        return None

    def _read_after_action(self, pre_gs: dict, action_type: str) -> dict | None:
        """
        Wait for the single state message Lua sends after processing an action.
        No stale-buffer draining needed: keep_fresh was removed from the Lua mod,
        so Lua only sends state (a) on G.STATE transitions or (b) via _send_after_action.
        """
        return self._read_state()

    # â”€â”€ Gymnasium interface â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Persistent connection: connect once, keep alive across episodes.
        # Only connect if not already connected (first reset or after a drop).
        if not self._connected:
            if not self._connect():
                return np.zeros(OBS_SIZE, dtype=np.float32), {}

        # Send new_run to trigger a fresh game
        # (lua_nav may also auto-restart, either way we just need the next selecting_hand)
        self._send_action("new_run", [])

        # Drain until we get selecting_hand (first one after new_run = new episode)
        # Skip game_over, round_eval, shop, blind_select — those are from the transition
        deadline = time.time() + 30.0
        while time.time() < deadline:
            gs = self._read_state()
            if gs is None:
                break
            if gs.get("event") == "selecting_hand":
                self._episode_seed = gs.get("seed")
                self._last_gs = gs
                return state_to_obs(parse_state(gs)), {}
            # Non-selecting_hand states (blind_select, shop, etc.) — keep draining
        return np.zeros(OBS_SIZE, dtype=np.float32), {}

    def step(self, action: int):
        if not self._connected:
            return np.zeros(OBS_SIZE, dtype=np.float32), 0.0, True, False, {}

        pre_gs = getattr(self, "_last_gs", None) or {}
        card_indices, action_type = action_to_cards_and_type(action, parse_state(pre_gs)) if pre_gs else ([], "play")

        if not self._send_action(action_type, card_indices):
            return np.zeros(OBS_SIZE, dtype=np.float32), 0.0, True, False, {"terminal_reason": "send_failed"}

        # Drain stale buffered messages, wait for a fresh state
        gs = self._read_after_action(pre_gs, action_type)
        if gs is None:
            return np.zeros(OBS_SIZE, dtype=np.float32), 0.0, True, False, {"terminal_reason": "timeout"}

        self._last_gs = gs

        # Terminal detection
        terminated = False
        terminal_reason = None

        if gs.get("event") == "game_over":
            terminated = True
            terminal_reason = "game_over"
        elif gs.get("seed") != self._episode_seed and self._episode_seed is not None:
            # Seed changed = new game started = we missed the terminal
            terminated = True
            terminal_reason = "missed_terminal"

        obs = state_to_obs(parse_state(gs))
        reward = self._compute_reward(gs, terminated)
        info = {
            "ante": gs.get("ante", 1),
            "seed": gs.get("seed"),
            "event": gs.get("event"),
        }
        if terminal_reason:
            info["terminal_reason"] = terminal_reason

        return obs, reward, terminated, False, info

    def _compute_reward(self, gs: dict, terminated: bool) -> float:
        ante      = gs.get("ante", 1)
        score     = gs.get("current_score", 0)
        target    = gs.get("score_target", 1) or 1
        event     = gs.get("event", "")

        # Dense score progress: reward proportional to how close to clearing the blind
        prev_prog  = getattr(self, "_prev_score_progress", 0.0)
        curr_prog  = min(score / target, 1.0)
        prog_delta = max(curr_prog - prev_prog, 0.0)
        self._prev_score_progress = curr_prog if not terminated else 0.0

        reward = prog_delta * 0.5   # up to +0.5 per blind for full score progress

        if terminated:
            self._prev_score_progress = 0.0
            reward += float(ante - 1) * 2.0   # +2 per ante cleared
            reward -= 1.0                       # lose penalty

        return reward

    def action_masks(self) -> np.ndarray:
        # Action masking disabled for now — all actions permitted, agent learns
        # to avoid invalid actions via reward signal. Re-enable once training is stable.
        return np.ones(ACTION_SIZE, dtype=bool)

    def close(self):
        self._disconnect()
