from __future__ import annotations

import numpy as np

from .agent import ParticleAgent
from ...utils.custom_objects.data_classes import Solution
from ...utils import global_parameters as gp
from ...utils.functions import discrete


class FOAAgent(ParticleAgent):
    iterations = gp.FOA_ITERATIONS
    
    def __init__(self, supervisor: 'Supervisor'):
        super().__init__(supervisor)
        
        self.W = gp.W_FOA
        
        self.local_best = Solution(
            pos := discrete(np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, gp.DIMENSIONS)), 
            gp.OBJECTIVE_FUNCTION(pos)
        )
        self.global_best: Solution = Solution()

    def set_global_best(self, global_best: Solution):
        self.global_best = global_best
    
    def random_fly(self):
        return discrete(self.global_best.position + self.W * np.random.normal(0, 1, gp.DIMENSIONS))

    def adapt(self, exploration: int, exploitation: int):
        self.W *= exploration * exploitation
    
    def execute(self):
        for iteration in range(self.__class__.iterations):
            new_pos = self.random_fly()
            new_val = gp.OBJECTIVE_FUNCTION(new_pos)
            if new_val < self.local_best.value:
                self.local_best = Solution(new_pos, new_val)
                if self.global_best_agent_type.value > self.local_best.value:
                    self.supervisor.update_global_best(self.local_best, self.__class__)

        self.supervisor.collect_results(self, self.local_best.value)
