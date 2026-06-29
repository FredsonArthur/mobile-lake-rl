
"""
Classes base para o projeto MobileLake
FrozenPond, RandomLake, RandomLakeObs
"""

import numpy as np
from gymnasium import spaces, Env


class FrozenPond(Env):
    """Ambiente FrozenLake simplificado"""

    def __init__(self, desc=None, is_slippery=False):
        super().__init__()

        self.nrow = 4
        self.ncol = 4
        self.nS = 16
        self.nA = 4
        self.is_slippery = is_slippery

        if desc is not None:
            self.desc = [list(row) for row in desc]
        else:
            self.desc = [
                ['S', 'F', 'F', 'F'],
                ['F', 'H', 'F', 'H'],
                ['F', 'F', 'F', 'H'],
                ['H', 'F', 'F', 'G']
            ]

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self._row = 0
        self._col = 0
        self._goal_row = 3
        self._goal_col = 3
        self._holes = []

        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i][j] == 'H':
                    self._holes.append((i, j))
                elif self.desc[i][j] == 'G':
                    self._goal_row = i
                    self._goal_col = j

    def _to_state(self, row, col):
        return row * self.ncol + col

    def reset(self, seed=None, options=None):
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i][j] == 'S':
                    self._row = i
                    self._col = j
                    break
        return self._to_state(self._row, self._col), {}

    def step(self, action):
        row, col = self._row, self._col

        if action == 0:   # esquerda
            col = max(0, col - 1)
        elif action == 1: # baixo
            row = min(self.nrow - 1, row + 1)
        elif action == 2: # direita
            col = min(self.ncol - 1, col + 1)
        elif action == 3: # cima
            row = max(0, row - 1)

        if (row, col) in self._holes:
            self._row, self._col = row, col
            return self._to_state(row, col), -1.0, True, False, {}
        elif (row, col) == (self._goal_row, self._goal_col):
            self._row, self._col = row, col
            return self._to_state(row, col), 1.0, True, False, {}
        else:
            self._row, self._col = row, col
            return self._to_state(row, col), 0.0, False, False, {}

    def render(self):
        desc = [list(row) for row in self.desc]
        if self._row is not None and self._col is not None:
            if desc[self._row][self._col] not in ['S', 'G']:
                desc[self._row][self._col] = 'A'
        return '\n'.join([' '.join(row) for row in desc])


class RandomLake(FrozenPond):
    """FrozenPond com buracos sorteados aleatoriamente"""

    def __init__(self, hole_prob=0.2, is_slippery=False):
        self.hole_prob = hole_prob
        self.is_slippery = is_slippery
        desc = self._generate_random_map()
        super().__init__(desc=desc, is_slippery=is_slippery)

    def _generate_random_map(self):
        desc = [['F' for _ in range(4)] for _ in range(4)]
        desc[0][0] = 'S'
        desc[3][3] = 'G'
        for i in range(4):
            for j in range(4):
                if (i, j) != (0, 0) and (i, j) != (3, 3):
                    if np.random.random() < self.hole_prob:
                        desc[i][j] = 'H'
        return desc

    def reset(self, seed=None, options=None):
        desc = self._generate_random_map()
        self.desc = desc
        self._holes = []
        for i in range(4):
            for j in range(4):
                if self.desc[i][j] == 'H':
                    self._holes.append((i, j))
                elif self.desc[i][j] == 'G':
                    self._goal_row = i
                    self._goal_col = j
        return super().reset(seed=seed, options=options)


class RandomLakeObs(RandomLake):
    """RandomLake com observação de vizinhança de 4 bits"""

    def __init__(self, hole_prob=0.2, is_slippery=False):
        super().__init__(hole_prob=hole_prob, is_slippery=is_slippery)
        self.observation_space = spaces.MultiDiscrete([2, 2, 2, 2])

    def _get_obs(self):
        row, col = self._row, self._col
        return np.array([
            1 if row > 0 else 0,
            1 if row < 3 else 0,
            1 if col > 0 else 0,
            1 if col < 3 else 0
        ])

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._get_obs(), reward, terminated, truncated, info
