"""Checks for state observations and compatibility with the pixel environment."""
import sys
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pusht_mobile_env  # noqa: F401


class ObservationTests(unittest.TestCase):
    def test_state_observations_match_simulator_without_rendering(self):
        with patch.dict(sys.modules, {'pusht_mobile_env.render': None}):
            env = gym.make('PushTMobile-v0', obs_type='state', render_mode=None)
            try:
                obs, _ = env.reset(options={'state': ((100, 100), (250, 250), 4.)})
                for _ in range(3):
                    x, y, angle = env.unwrapped.sim.block_pose()
                    expected = [*env.unwrapped.sim.agent_pos(), x, y,
                                (angle + np.pi) % (2 * np.pi) - np.pi]
                    np.testing.assert_allclose(obs, expected, atol=1e-5)
                    self.assertEqual(obs.dtype, np.float32)
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertTrue(-np.pi <= obs[4] < np.pi)
                    obs, *_ = env.step(np.array([200., 200.]))
            finally:
                env.close()

    def test_default_pixels_and_physics_are_unchanged(self):
        pixels = gym.make('PushTMobile-v0')
        state = gym.make('PushTMobile-v0', obs_type='state', render_mode=None)
        try:
            pixel_obs, pixel_info = pixels.reset(seed=42)
            state_obs, state_info = state.reset(seed=42)
            self.assertEqual(pixel_info, state_info)
            self.assertEqual(pixel_obs['pixels'].shape, (96, 96, 3))
            self.assertEqual(pixels.render().shape, (480, 480, 3))
            for action in ([200., 200.], [300., 300.], [-10., 600.]):
                pixel_obs, *pixel_result = pixels.step(action)
                state_obs, *state_result = state.step(action)
                self.assertEqual(pixel_result, state_result)
                np.testing.assert_allclose(pixel_obs['agent_pos'], state_obs[:2], atol=1e-5)
                self.assertEqual(pixels.unwrapped.sim.block_pose(), state.unwrapped.sim.block_pose())
        finally:
            pixels.close()
            state.close()

    def test_invalid_observation_type_is_rejected(self):
        with self.assertRaises(ValueError):
            gym.make('PushTMobile-v0', obs_type='unknown')


if __name__ == '__main__':
    unittest.main()
