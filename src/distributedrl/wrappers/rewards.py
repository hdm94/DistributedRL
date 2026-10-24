# Reward wrappers. Select one with the `reward` config field (see REWARDS below).
# These only depend on gymnasium, so they work with any environment, not just Atari.
import gymnasium as gym
import numpy as np

from distributedrl.wrappers.ram import ram_fields_for


class CustomRewardEnv(gym.Wrapper):
    """Shaped reward for Seaquest (the oxygen and diver terms read the emulator RAM, see wrappers/ram.py).

    reward = sign(game reward)
             - life_loss_penalty                          when a life is lost
             + diver_bonus                                when a diver is picked up
             + oxygen_coef * (potential(new) - potential(old))

    The oxygen potential is 0 when oxygen >= oxygen_threshold and falls linearly to -1 at 0 oxygen. Using the
    change in potential means running low is penalised once as the bar drops, recovering it (surfacing to
    breathe) pays that back, and diving and resurfacing repeatedly nets zero, so it can't be farmed. After a
    reset or a lost life the oxygen bar refills from near zero; that refill is ignored until the sub dives
    or the bar is full, otherwise dying would be rewarded. A single-step change bigger than the real drain or
    refill rate (e.g. the byte reads 0 during the death animation) is treated as a discontinuity and ignored.
    Oxygen decreases at the surface are ignored too: surfacing with all six divers converts the bar into bonus points.

    Use `gym.Wrapper` (not `RewardWrapper`) when you need the observation, `info` or the action. For games
    without a RAM table only the life-loss penalty applies.
    """

    life_loss_penalty = 1.0
    diver_bonus = 0.5
    oxygen_coef = 0.5
    oxygen_threshold = 16  # the oxygen byte runs 0..64; penalise below this
    oxygen_max = 64
    surface_depth = 13  # depth byte value when the sub is at the surface
    max_oxygen_step = 8  # real drain/refill is at most ~2 per agent step; bigger jumps are not oxygen changes

    def __init__(self, env):
        super().__init__(env)
        spec = env.unwrapped.spec
        self.fields = ram_fields_for(spec.id if spec is not None else "")
        self.use_ram = {"oxygen", "depth", "divers"} <= set(self.fields)
        self.lives = 0
        self.oxygen = self.divers = 0
        self.refilling = True
        self.last_parts = {}  # per-component reward of the last step, handy for debugging

    def _read(self):
        ram = self.env.unwrapped.ale.getRAM()
        return {name: int(ram[self.fields[name]]) for name in ("oxygen", "depth", "divers")}

    def _potential(self, oxygen):
        return -max(0, self.oxygen_threshold - oxygen) / self.oxygen_threshold

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get("lives", 0)
        self.refilling = True
        if self.use_ram:
            ram = self._read()
            self.oxygen, self.divers = ram["oxygen"], ram["divers"]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        parts = {"game": float(np.sign(reward)), "life": 0.0, "diver": 0.0, "oxygen": 0.0}

        lives = info.get("lives", self.lives)
        life_lost = lives < self.lives
        if life_lost:
            parts["life"] = -self.life_loss_penalty
        self.lives = lives

        if self.use_ram:
            ram = self._read()
            if life_lost:
                # the bar refills and the diver count resets on respawn: re-baseline, no reward
                self.refilling = True
            else:
                if ram["divers"] > self.divers:
                    parts["diver"] = self.diver_bonus
                if self.refilling:
                    if ram["oxygen"] >= self.oxygen_max - 1 or ram["depth"] > self.surface_depth + 2:
                        self.refilling = False
                elif abs(ram["oxygen"] - self.oxygen) <= self.max_oxygen_step:
                    delta = self._potential(ram["oxygen"]) - self._potential(self.oxygen)
                    if ram["depth"] <= self.surface_depth:
                        delta = max(delta, 0.0)  # at the surface the bar only drains when it is cashed in for bonus points
                    parts["oxygen"] = self.oxygen_coef * delta
            self.oxygen, self.divers = ram["oxygen"], ram["divers"]

        self.last_parts = parts
        return obs, sum(parts.values()), terminated, truncated, info


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
