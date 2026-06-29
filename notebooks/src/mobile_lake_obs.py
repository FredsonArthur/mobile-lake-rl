
import numpy as np
from gymnasium import spaces
from src.mobile_lake import MobileLake


class MobileLakeObs(MobileLake):
    """MobileLake com observação estendida: vizinhança + coordenadas do objetivo"""

    def __init__(self, hole_prob=0.2, is_slippery=False):
        super().__init__(hole_prob=hole_prob, is_slippery=is_slippery)
        self.observation_space = spaces.MultiDiscrete([2, 2, 2, 2, 4, 4])

    def _get_obs(self):
        row, col = self._row, self._col
        up = 1 if row > 0 else 0
        down = 1 if row < 3 else 0
        left = 1 if col > 0 else 0
        right = 1 if col < 3 else 0
        goal_row, goal_col = self.goal_pos
        return np.array([up, down, left, right, goal_row, goal_col])

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._get_obs(), reward, terminated, truncated, info
