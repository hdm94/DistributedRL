# Reward wrappers. Select one with the `reward` config field (see REWARDS below).
# These only depend on gymnasium, so they work with any environment, not just Atari.
import gymnasium as gym
import numpy as np


class CustomRewardEnv(gym.Wrapper):
    """Example shaped reward for Atari: the sign of the game reward minus a penalty for each life lost.

    Use `gym.Wrapper` (not `RewardWrapper`) when you need the observation, `info` or the action.
    For Atari, `info["lives"]` and `env.unwrapped.ale.getRAM()` are available.
    """

    life_loss_penalty = 1.0

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get("lives", 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = float(np.sign(reward))  # clip the game score, as the standard Atari setup does
        lives = info.get("lives", self.lives)
        if lives < self.lives:
            shaped -= self.life_loss_penalty
        self.lives = lives
        return obs, shaped, terminated, truncated, info


# name -> wrapper class, or None to leave the environment reward untouched
REWARDS = {
    "raw": None,
    "custom": CustomRewardEnv,
}


def apply_reward(env, name):
    if name not in REWARDS:
        raise ValueError(f"unknown reward {name!r}, choose from {sorted(REWARDS)}")
    wrapper = REWARDS[name]
    return env if wrapper is None else wrapper(env)
