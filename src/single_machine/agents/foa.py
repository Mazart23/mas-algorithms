from __future__ import annotations

import numpy as np

from .agent import ParticleAgent
from ...utils.custom_objects.data_classes import Solution
from ...utils import global_parameters as gp


class FOAAgent(ParticleAgent):
    def __init__(self, supervisor: 'Supervisor'):
        super().__init__(supervisor)
        
        self.position = np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, gp.DIMENSIONS)
        self.local_best = Solution(self.position, gp.OBJECTIVE_FUNCTION(self.position))

    def random_fly(self):
        return self.position + np.random.normal(0, 1, gp.DIMENSIONS)

    def execute(self):
        for iteration in range(gp.FOA_ITERATIONS):
            new_pos = self.random_fly()
            new_val = gp.OBJECTIVE_FUNCTION(new_pos)
            if new_val < self.local_best.value:
                self.local_best = Solution(new_pos, new_val)
                self.position = new_pos
                self.supervisor.update_global_best(self.local_best, self.__class__)

        self.supervisor.collect_results(self.__class__, self.local_best.value)