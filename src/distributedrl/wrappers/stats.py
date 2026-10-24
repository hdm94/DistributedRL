# Per-episode game statistics, reported through `info` when a real game ends so they can be logged.
import gymnasium as gym

from distributedrl.wrappers.ram import ram_fields_for


class SeaquestStatsEnv(gym.Wrapper):
    """Counts Seaquest events over a whole game (all lives) and adds them to `info` on the step the game ends.

    surfacings        times the sub returned to the surface after a real dive (not bobbing, not respawns)
    divers_collected  divers picked up
    divers_delivered  divers handed in at the surface (the diver count drops while at the surface)
    episode_steps     agent steps in the game (4 emulator frames each)

    The game score is already logged as `charts/episodic_return` (the raw game reward).
    """

    stat_names = ("surfacings", "divers_collected", "divers_delivered", "episode_steps")
    surface_depth = 13  # depth byte value when the sub is at the surface
    dive_margin = 8  # the sub must get this far below the surface before returning counts as a surfacing

    def __init__(self, env):
        super().__init__(env)
        spec = env.unwrapped.spec
        self.fields = ram_fields_for(spec.id if spec is not None else "")
        assert {"depth", "divers"} <= set(self.fields), "SeaquestStatsEnv needs the Seaquest RAM table"
        self.game_over = True  # forces a fresh count on the first reset
        self._zero()

    def _zero(self):
        self.counts = dict.fromkeys(self.stat_names, 0)

    def _read(self):
        ram = self.env.unwrapped.ale.getRAM()
        return int(ram[self.fields["depth"]]), int(ram[self.fields["divers"]])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.game_over:  # a reset after losing a life keeps counting; only a finished game starts over
            self._zero()
            self.game_over = False
        self.lives = info.get("lives", 0)
        self.depth, self.divers = self._read()
        self.dived = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.counts["episode_steps"] += 1
        depth, divers = self._read()
        lives = info.get("lives", self.lives)
        if lives < self.lives:  # a life was lost: the sub respawns at the surface and the diver count resets
            self.dived = False
        else:
            if depth >= self.surface_depth + self.dive_margin:
                self.dived = True
            elif depth <= self.surface_depth and self.dived:
                self.counts["surfacings"] += 1
                self.dived = False
            if divers > self.divers:
                self.counts["divers_collected"] += divers - self.divers
            elif divers < self.divers and depth <= self.surface_depth:
                self.counts["divers_delivered"] += self.divers - divers
        self.lives, self.depth, self.divers = lives, depth, divers
        if truncated or (terminated and self.env.unwrapped.ale.game_over()):
            self.game_over = True
            info.update(self.counts)
        return obs, reward, terminated, truncated, info
