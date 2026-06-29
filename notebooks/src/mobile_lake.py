
"""
MobileLake - Ambiente com objetivo móvel
"""

import numpy as np
from src.frozen_pond import RandomLake


class MobileLake(RandomLake):
    """RandomLake com objetivo sorteado aleatoriamente a cada episódio"""

    def __init__(self, hole_prob=0.2, is_slippery=False):
        super().__init__(hole_prob=hole_prob, is_slippery=is_slippery)
        self.goal_pos = (3, 3)
        self._row = 0
        self._col = 0

    def _generate_random_map_with_goal(self, goal_pos):
        desc = [['F' for _ in range(4)] for _ in range(4)]
        desc[0][0] = 'S'
        gx, gy = goal_pos
        desc[gx][gy] = 'G'
        for i in range(4):
            for j in range(4):
                if (i, j) != (0, 0) and (i, j) != goal_pos:
                    if np.random.random() < self.hole_prob:
                        desc[i][j] = 'H'
        return desc

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        possible_goals = [(i, j) for i in range(4) for j in range(4) if (i, j) != (0, 0)]
        self.goal_pos = possible_goals[np.random.randint(len(possible_goals))]

        desc = self._generate_random_map_with_goal(self.goal_pos)
        self.desc = desc
        self._holes = []
        for i in range(4):
            for j in range(4):
                if self.desc[i][j] == 'H':
                    self._holes.append((i, j))
                elif self.desc[i][j] == 'G':
                    self._goal_row = i
                    self._goal_col = j

        obs, info = super(RandomLake, self).reset(seed=seed, options=options)
        info['goal'] = self.goal_pos
        self._row = 0
        self._col = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super(RandomLake, self).step(action)
        if obs is not None and not terminated:
            self._row = obs // self.ncol
            self._col = obs % self.ncol
        return obs, reward, terminated, truncated, info
