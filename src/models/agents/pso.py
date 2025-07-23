from __future__ import annotations
import copy

import numpy as np

from .agent import ParticleAgent
from ...utils.custom_objects.data_classes import Solution
from ...utils import global_parameters as gp


class PSOAgent(ParticleAgent):
    iterations = gp.PSO_ITERATIONS
    
    def __init__(self, supervisor: 'Supervisor'):
        super().__init__(supervisor)
        
        self.W = gp.W
        self.C1 = gp.C1
        self.C2 = gp.C2
        
        position = np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, (gp.DIMENSIONS,))
        self.velocities: np.ndarray[float] = np.random.uniform(-1, 1, (gp.DIMENSIONS,))
        
        self.current: Solution = Solution(position, gp.OBJECTIVE_FUNCTION(position))
        self.local_best: Solution = copy.deepcopy(self.current)
        self.global_best: Solution = Solution()

    def set_global_best(self, global_best: Solution):
        self.global_best = global_best

    def adapt(self, exploration: int, exploitation: int):
        self.W *= exploitation
        self.C1 *= exploration * exploitation
        self.C2 *= exploration
    
    def execute(self) -> None:
        for iteration in range(self.__class__.iterations):        
            new_vel = (
                self.W * self.velocities +
                self.C1 * np.random.rand(gp.DIMENSIONS) * (self.local_best.position - self.current.position) +
                self.C2 * np.random.rand(gp.DIMENSIONS) * (self.global_best.position - self.current.position)
            )
            new_pos = np.clip(self.current.position + new_vel, gp.MIN_VALUE, gp.MAX_VALUE)
            self.velocities = new_pos - self.current.position
            self.current.position = new_pos
            self.current.value = gp.OBJECTIVE_FUNCTION(self.current.position)
            
            if self.current.value < self.local_best.value:
                self.local_best = copy.deepcopy(self.current)
                if self.global_best_agent_type.value > self.local_best.value:
                    self.supervisor.update_global_best(self.local_best, self.__class__)
        
        self.supervisor.collect_results(self, self.local_best.value)
