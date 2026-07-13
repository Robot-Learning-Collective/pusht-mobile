from gymnasium.envs.registration import register

# entry_point is a string: env.py (and the pusht_core extension it needs) is
# only imported on gym.make(), so `pusht_mobile_env.render` stays importable
# on machines without the rust wheel (e.g. the collector VM running exports).
register(
    id="PushTMobile-v0",
    entry_point="pusht_mobile_env.env:PushTMobileEnv",
    max_episode_steps=300,  # same cap as gym_pusht/PushT-v0
)
