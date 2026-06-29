"""
MobileLake - Ambiente com objetivo móvel
Disciplina: Tópicos Especiais em Inteligência Artificial - IFPE
Seção A: Implementação do ambiente
"""

import numpy as np
from src.frozen_pond import RandomLake


class MobileLake(RandomLake):
    """
    RandomLake com objetivo sorteado aleatoriamente a cada episódio.
    O objetivo pode ser qualquer célula exceto (0,0).
    
    Seção A1: Método reset sobrescrito
    Seção A2: Métodos reward e done sobrescritos
    """
    
    def __init__(self, hole_prob=0.2, is_slippery=False):
        super().__init__(hole_prob=hole_prob, is_slippery=is_slippery)
        self.goal_pos = (3, 3)  # será sobrescrito no reset
        self._row = 0
        self._col = 0
        
    def _generate_random_map_with_goal(self, goal_pos):
        """
        Gera mapa aleatório com objetivo em goal_pos.
        Buracos não podem estar no início ou no objetivo.
        """
        desc = [['F' for _ in range(self.ncol)] for _ in range(self.nrow)]
        desc[0][0] = 'S'
        
        gx, gy = goal_pos
        desc[gx][gy] = 'G'
        
        # Adiciona buracos (não no início nem no objetivo)
        for i in range(self.nrow):
            for j in range(self.ncol):
                if (i, j) != (0, 0) and (i, j) != goal_pos:
                    if np.random.random() < self.hole_prob:
                        desc[i][j] = 'H'
        
        return desc
    
    def reset(self, seed=None, options=None):
        """
        Sorteia o objetivo aleatoriamente e gera o mapa.
        Retorna: obs, info com chave 'goal'
        """
        # Sorteia objetivo (exceto (0,0))
        if seed is not None:
            np.random.seed(seed)
        
        # Gera posições possíveis (todas exceto (0,0))
        possible_goals = [(i, j) for i in range(self.nrow) for j in range(self.ncol) 
                         if (i, j) != (0, 0)]
        self.goal_pos = possible_goals[np.random.randint(len(possible_goals))]
        
        # Gera mapa com este objetivo
        desc = self._generate_random_map_with_goal(self.goal_pos)
        self.desc = desc
        
        # Atualiza lista de buracos
        self._holes = []
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i][j] == 'H':
                    self._holes.append((i, j))
                elif self.desc[i][j] == 'G':
                    self._goal_row = i
                    self._goal_col = j
        
        # Chama o reset da classe pai
        obs, info = super(RandomLake, self).reset(seed=seed, options=options)
        
        # Atualiza posição
        self._row = 0
        self._col = 0
        
        # Adiciona a posição do objetivo no info
        info['goal'] = self.goal_pos
        
        return obs, info
    
    def step(self, action):
        """Executa um passo e mantém posição atualizada"""
        # Usa o step da classe RandomLake (que chama FrozenPond)
        obs, reward, terminated, truncated, info = super(RandomLake, self).step(action)
        
        # Atualiza posição se não terminou
        if obs is not None and not terminated:
            self._row = obs // self.ncol
            self._col = obs % self.ncol
        
        return obs, reward, terminated, truncated, info
    
    # ==================== SEÇÃO A2 ====================
    
    def reward(self, state, action, next_state):
        """
        Calcula recompensa baseada no objetivo atual.
        +1 se chegou ao objetivo, -1 se caiu em buraco, 0 caso contrário.
        """
        row = next_state // self.ncol
        col = next_state % self.ncol
        
        # Verifica se chegou ao objetivo
        if (row, col) == self.goal_pos:
            return 1.0
        elif self.desc[row][col] == 'H':
            return -1.0
        else:
            return 0.0
    
    def done(self, state, action, next_state):
        """
        Episódio termina se chegou ao objetivo ou caiu em buraco.
        """
        row = next_state // self.ncol
        col = next_state % self.ncol
        
        if (row, col) == self.goal_pos:
            return True
        elif self.desc[row][col] == 'H':
            return True
        return False