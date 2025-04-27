from __future__ import annotations
import copy

import numpy as np

from .agent import ParticleAgent
from ...utils.custom_objects.data_classes import Solution
from ...utils import global_parameters as gp


class BeeAgent(ParticleAgent):
    def __init__(self, supervisor: 'Supervisor'):
        super().__init__(supervisor)
        
        self.W = gp.W_BEE
        
        self.current: Solution = Solution(pos := np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, (gp.DIMENSIONS,)), gp.OBJECTIVE_FUNCTION(pos))
        self.local_best: Solution = copy.deepcopy(self.current)

    def explore_neighbourhood(self, position):
        phi = np.random.uniform(-1, 1, gp.DIMENSIONS)
        partner = np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, gp.DIMENSIONS)
        return np.clip(position + self.W * phi * (position - partner), gp.MIN_VALUE, gp.MAX_VALUE)

    def adapt(self, exploration: int, exploatation: int):
        self.W *= exploatation * exploration
    
    def execute(self):
        for iteration in range(gp.BEE_ITERATIONS):
            new_position = self.explore_neighbourhood(self.current.position)
            new_value = gp.OBJECTIVE_FUNCTION(new_position)
            if new_value < self.current.value:
                self.current = Solution(new_position, new_value)
                if new_value < self.local_best.value:
                    self.local_best = copy.deepcopy(self.current)
                    if self.global_best_agent_type.value > self.local_best.value:
                        self.supervisor.update_global_best(self.local_best, self.__class__)

        self.supervisor.collect_results(self.__class__, self.local_best.value)