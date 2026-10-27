"""Gymnasium env over the pusht-core Rust sim (product mode).

Interface mirrors gym_pusht's `obs_type="pixels_agent_pos"`:
    obs   = {"pixels": (96, 96, 3) uint8, "agent_pos": (2,) float64}
    action = target position in [0, 512]^2
    reward = clip(coverage / 0.95, 0, 1); terminated once coverage >= 0.95

With obs_type="state", obs is float32 [agent_x, agent_y, T_x, T_y, T_angle].

The physics is the exact engine (rapier2d 0.22) and logic the mobile
demonstrations were collected with; observations are rendered with the same
code that produced the training dataset images.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import pusht_core as pc
from gymnasium import spaces


class PushTMobileEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": pc.CONTROL_HZ}

    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        observation_size: int = 96,
        render_size: int = 480,
        obs_type: str = "pixels_agent_pos",
    ):
        if obs_type not in ("pixels_agent_pos", "state"):
            raise ValueError(f"Unknown obs_type: {obs_type}")
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_size = observation_size
        self.render_size = render_size
        self.sim = pc.PushTSim("product")
        self.goal = pc.GOAL_POSE

        if obs_type == "state":
            self.observation_space = spaces.Box(
                np.array([0, 0, 0, 0, -np.pi], dtype=np.float32),
                np.array([pc.WORLD] * 4 + [np.pi], dtype=np.float32),
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(0, 255, (observation_size, observation_size, 3), np.uint8),
                    "agent_pos": spaces.Box(0.0, pc.WORLD, (2,), np.float64),
                }
            )
        self.action_space = spaces.Box(0.0, pc.WORLD, (2,), np.float32)

    def _frame(self, size: int) -> np.ndarray:
        from .render import render_frame

        return render_frame(
            self.sim.agent_pos(),
            self.sim.block_pose(),
            self.goal,
            pc.PRODUCT_SCALE,
            pc.PRODUCT_AGENT_RADIUS,
            size,
        )

    def _pixels(self) -> np.ndarray:
        return self._frame(self.observation_size)

    def _obs(self) -> dict[str, np.ndarray] | np.ndarray:
        if self.obs_type == "state":
            x, y, angle = self.sim.block_pose()
            return np.array([*self.sim.agent_pos(), x, y,
                             (angle + np.pi) % (2 * np.pi) - np.pi], dtype=np.float32)
        return {
            "pixels": self._pixels(),
            "agent_pos": np.array(self.sim.agent_pos(), dtype=np.float64),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if options is not None and "state" in options:
            # Explicit state: ((ax, ay), (bx, by), block_angle) — for replays.
            agent, block_pos, block_angle = options["state"]
        else:
            init_seed = int(self.np_random.integers(0, 2**32))
            agent, block_pos, block_angle = pc.PushTSim.random_init(init_seed)
        self.sim.reset(tuple(agent), tuple(block_pos), float(block_angle))
        cov = self.sim.coverage()
        return self._obs(), {"coverage": cov, "is_success": False}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), 0.0, pc.WORLD)
        self.sim.step((float(a[0]), float(a[1])))
        cov = self.sim.coverage()
        terminated = cov >= pc.SUCCESS_THRESHOLD
        reward = min(cov / pc.SUCCESS_THRESHOLD, 1.0)
        return self._obs(), reward, terminated, False, {"coverage": cov, "is_success": terminated}

    def render(self):
        # Observations stay observation_size; render() serves render_size for
        # human-friendly rollout videos (gym_pusht renders 680, we default 480).
        return self._frame(self.render_size)
