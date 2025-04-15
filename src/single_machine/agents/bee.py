from __future__ import annotations
import copy

import numpy as np

from .agent import ParticleAgent
from ...utils.custom_objects.data_classes import Solution
from ...utils import global_parameters as gp


class BeeAgent(ParticleAgent):
    def __init__(self, supervisor: 'Supervisor'):
        super().__init__(supervisor)
        
        self.current: Solution = Solution(np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, (gp.DIMENSIONS,)))
        self.current.value = gp.OBJECTIVE_FUNCTION(self.current.position)
        self.local_best: Solution = copy.deepcopy(self.current)

    def explore_neighbourhood(self, position):
        phi = np.random.uniform(-1, 1, gp.DIMENSIONS)
        partner = np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, gp.DIMENSIONS)
        return np.clip(position + phi * (position - partner), gp.MIN_VALUE, gp.MAX_VALUE)

    def execute(self):
        for iteration in range(gp.BEE_ITERATIONS):
            new_position = self.explore_neighbourhood(self.current.position)
            new_value = gp.OBJECTIVE_FUNCTION(new_position)
            if new_value < self.current.value:
                self.current = Solution(new_position, new_value)
                if new_value < self.local_best.value:
                    self.local_best = copy.deepcopy(self.current)
                    self.supervisor.update_global_best(self.local_best, self.__class__)

        self.supervisor.collect_results(self.__class__, self.local_best.value)