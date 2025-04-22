from __future__ import annotations

import numpy as np

from .agent import ParticleAgent
from ...utils.custom_objects.data_classes import Solution
from ...utils import global_parameters as gp

class DEAgent(ParticleAgent):

    def __init__(self, supervisor: 'Supervisor'):
        super().__init__(supervisor)
        
        self.local_best = Solution(pos := np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, gp.DIMENSIONS), gp.OBJECTIVE_FUNCTION(pos))

    def mutate(self, r1, r2, r3):
        return np.clip(r1 + gp.DE_F * (r2 - r3), gp.MIN_VALUE, gp.MAX_VALUE)

    def crossover(self, target, mutant):
        return np.array([mutant[i] if np.random.rand() < gp.DE_CR else target[i] for i in range(gp.DIMENSIONS)])

    def execute(self):
        for iteration in range(gp.DE_ITERATIONS):
            r1, r2, r3 = [np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, gp.DIMENSIONS) for _ in range(3)]
            mutant = self.mutate(r1, r2, r3)
            trial = self.crossover(self.local_best.position, mutant)
            trial_value = gp.OBJECTIVE_FUNCTION(trial)
            if trial_value < self.local_best.value:
                self.local_best = Solution(trial, trial_value)
                if self.global_best_agent_type.value > self.local_best.value:
                    self.supervisor.update_global_best(self.local_best, self.__class__)

        self.supervisor.collect_results(self.__class__, self.local_best.value)