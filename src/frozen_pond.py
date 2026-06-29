"""
Classes base para o projeto MobileLake
FrozenPond, RandomLake, RandomLakeObs, MyCallbacks, ppo_config_base
"""

import numpy as np
from gymnasium import spaces, Env
from stable_baselines3.common.callbacks import BaseCallback


class FrozenPond(Env):
    """Ambiente FrozenLake simplificado sem renderização gráfica"""
    
    def __init__(self, desc=None, map_name="4x4", is_slippery=False):
        super().__init__()
        
        self.nrow = 4
        self.ncol = 4
        self.nS = self.nrow * self.ncol
        self.nA = 4
        self.is_slippery = is_slippery
        
        # Define o mapa
        if desc is not None:
            self.desc = [list(row) for row in desc]
        else:
            # Mapa padrão 4x4
            self.desc = [
                ['S', 'F', 'F', 'F'],
                ['F', 'H', 'F', 'H'],
                ['F', 'F', 'F', 'H'],
                ['H', 'F', 'F', 'G']
            ]
        
        # Espaços de observação e ação
        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)
        
        # Estado atual
        self._row = 0
        self._col = 0
        self._goal_row = 3
        self._goal_col = 3
        self._holes = []
        
        # Identifica buracos e objetivo
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i][j] == 'H':
                    self._holes.append((i, j))
                elif self.desc[i][j] == 'G':
                    self._goal_row = i
                    self._goal_col = j
    
    def _to_state(self, row, col):
        """Converte (row, col) para índice linear"""
        return row * self.ncol + col
    
    def _to_pos(self, state):
        """Converte índice linear para (row, col)"""
        return state // self.ncol, state % self.ncol
    
    def reset(self, seed=None, options=None):
        """Reinicia o ambiente"""
        # Encontra posição inicial (S)
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i][j] == 'S':
                    self._row = i
                    self._col = j
                    break
        
        return self._to_state(self._row, self._col), {}
    
    def step(self, action):
        """Executa uma ação"""
        # Ação: 0=esquerda, 1=baixo, 2=direita, 3=cima
        if self.is_slippery:
            # Com escorregamento, ação pode falhar
            # Implementação simplificada - sem escorregamento por padrão
            pass
        
        # Nova posição baseada na ação
        row, col = self._row, self._col
        
        if action == 0:   # esquerda
            col = max(0, col - 1)
        elif action == 1: # baixo
            row = min(self.nrow - 1, row + 1)
        elif action == 2: # direita
            col = min(self.ncol - 1, col + 1)
        elif action == 3: # cima
            row = max(0, row - 1)
        
        # Verifica se caiu em buraco
        if (row, col) in self._holes:
            self._row, self._col = row, col
            reward = -1.0
            terminated = True
        # Verifica se chegou ao objetivo
        elif (row, col) == (self._goal_row, self._goal_col):
            self._row, self._col = row, col
            reward = 1.0
            terminated = True
        else:
            self._row, self._col = row, col
            reward = 0.0
            terminated = False
        
        return self._to_state(self._row, self._col), reward, terminated, False, {}
    
    def render(self):
        """Renderização em texto"""
        desc = [list(row) for row in self.desc]
        # Marca posição atual
        if self._row is not None and self._col is not None:
            if desc[self._row][self._col] not in ['S', 'G']:
                desc[self._row][self._col] = 'A'
        return '\n'.join([' '.join(row) for row in desc])


class RandomLake(FrozenPond):
    """FrozenPond com buracos sorteados aleatoriamente a cada episódio"""
    
    def __init__(self, hole_prob=0.2, is_slippery=False):
        self.hole_prob = hole_prob
        self.is_slippery = is_slippery
        
        # Gera mapa inicial
        desc = self._generate_random_map()
        super().__init__(desc=desc, is_slippery=is_slippery)
    
    def _generate_random_map(self):
        """Gera mapa aleatório com buracos em 20% das células"""
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
        """Gera novo mapa aleatório a cada reset"""
        desc = self._generate_random_map()
        self.desc = desc
        self._holes = []
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i][j] == 'H':
                    self._holes.append((i, j))
                elif self.desc[i][j] == 'G':
                    self._goal_row = i
                    self._goal_col = j
        
        return super().reset(seed=seed, options=options)
    
    def step(self, action):
        """Executa ação com recompensa do RandomLake"""
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info


class RandomLakeObs(RandomLake):
    """RandomLake com observação de vizinhança de 4 bits"""
    
    def __init__(self, hole_prob=0.2, is_slippery=False):
        super().__init__(hole_prob=hole_prob, is_slippery=is_slippery)
        self.observation_space = spaces.MultiDiscrete([2, 2, 2, 2])
    
    def _get_obs(self):
        """Retorna observação: [cima, baixo, esquerda, direita]"""
        row, col = self._row, self._col
        
        up = 1 if row > 0 else 0
        down = 1 if row < self.nrow - 1 else 0
        left = 1 if col > 0 else 0
        right = 1 if col < self.ncol - 1 else 0
        
        return np.array([up, down, left, right])
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._get_obs(), reward, terminated, truncated, info


class MyCallbacks(BaseCallback):
    """Callback para registrar métricas"""
    
    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.goal_reached = []
        self.episode_len = []
        self.rewards = []
    
    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        n_episodes = 10
        goals = 0
        lengths = []
        total_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            steps = 0
            ep_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.eval_env.step(action)
                steps += 1
                ep_reward += reward
                
                if reward == 1:
                    goals += 1
            
            lengths.append(steps)
            total_rewards.append(ep_reward)
        
        self.goal_reached.append(goals / n_episodes)
        self.episode_len.append(np.mean(lengths))
        self.rewards.append(np.mean(total_rewards))


def ppo_config_base(env):
    """Configuração base do PPO"""
    return {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': 0.0003,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'verbose': 0
    }