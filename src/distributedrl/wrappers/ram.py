# Expose the Atari emulator RAM through `info` so game state (lives, depth, oxygen, ...) is available per step.
import gymnasium as gym

# Named RAM byte indices per game, keyed by a substring of the env id. These come from community RAM
# annotations; lives_ram, depth and oxygen were checked in play, player_x and divers were not.
RAM_FIELDS = {
    "Seaquest": {"lives_ram": 59, "depth": 97, "oxygen": 102, "player_x": 70, "divers": 62},
}


def ram_fields_for(env_id):
    for game, fields in RAM_FIELDS.items():
        if game in env_id:
            return fields
    return {}


class RamInfoEnv(gym.Wrapper):
    """Adds `info["ram"]` (all 128 bytes) and one decoded int per named field to every reset and step."""

    def __init__(self, env, fields):
        super().__init__(env)
        self.fields = fields

    def _add_ram(self, info):
        ram = self.env.unwrapped.ale.getRAM()
        info["ram"] = ram.copy()
        for name, idx in self.fields.items():
            info[name] = int(ram[idx])
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._add_ram(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._add_ram(info)
