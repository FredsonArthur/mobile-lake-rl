
import numpy as np
from src.mobile_lake_obs import MobileLakeObs


class MobileLakeObsRew(MobileLakeObs):
    """MobileLakeObs com recompensa densa"""

    def __init__(self, hole_prob=0.2, is_slippery=False):
        super().__init__(hole_prob=hole_prob, is_slippery=is_slippery)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        row, col = self._row, self._col

        if (row, col) == self.goal_pos:
            reward = 1.0
        elif self.desc[row][col] == 'H':
            reward = -0.5
        else:
            reward = -0.01
            if action == 0 and row == 0:    # cima na borda
                reward -= 0.05
            elif action == 1 and row == 3:  # baixo na borda
                reward -= 0.05
            elif action == 2 and col == 0:  # esquerda na borda
                reward -= 0.05
            elif action == 3 and col == 3:  # direita na borda
                reward -= 0.05

        return self._get_obs(), reward, terminated, truncated, info
