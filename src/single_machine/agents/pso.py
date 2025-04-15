from __future__ import annotations
import copy

import numpy as np

from .agent import ParticleAgent
from ...utils.custom_objects.data_classes import Solution
from ...utils import global_parameters as gp


class PSOAgent(ParticleAgent):
    def __init__(self, supervisor: 'Supervisor'):
        super().__init__(supervisor)
        
        position = np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, (gp.DIMENSIONS,))
        self.velocities: np.ndarray[float] = np.random.uniform(-1, 1, (gp.DIMENSIONS,))
        
        self.current: Solution = Solution(position, gp.OBJECTIVE_FUNCTION(position))
        self.local_best: Solution | None = copy.deepcopy(self.current)
        self.global_best: Solution = Solution()

    def set_global_best(self, global_best: Solution):
        self.global_best = global_best
    
    def execute(self) -> None:
        for iteration in range(gp.PSO_ITERATIONS):
            global_best_position = self.global_best.position
        
            self.velocities = (
                gp.W * self.velocities +
                gp.C1 * np.random.rand(gp.DIMENSIONS) * (self.local_best.position - self.current.position) +
                gp.C2 * np.random.rand(gp.DIMENSIONS) * (global_best_position - self.current.position)
            )
            self.current.position += self.velocities
            self.current.value = gp.OBJECTIVE_FUNCTION(self.current.position)
            
            if self.current.value < self.local_best.value:
                self.local_best = copy.deepcopy(self.current)
                if self.global_best.value > self.local_best.value:
                    self.supervisor.update_global_best(self.local_best, self.__class__)
        
        self.supervisor.collect_results(self.__class__, self.local_best.value)