from __future__ import annotations

import numpy as np

from .agent import ParticleAgent
from ...utils.custom_objects.data_classes import Solution
from ...utils import global_parameters as gp

class DEAgent(ParticleAgent):
    iterations = gp.DE_ITERATIONS

    def __init__(self, supervisor: 'Supervisor'):
        super().__init__(supervisor)
        
        self.F = gp.F_DE
        self.CR = gp.CR_DE
        
        self.local_best = Solution(pos := np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, gp.DIMENSIONS), gp.OBJECTIVE_FUNCTION(pos))

    def get_childs(self):
        return [self.local_best]

    def mutate(self, r1, r2, r3) -> np.ndarray:
        return np.clip(r1 + self.F * (r2 - r3), gp.MIN_VALUE, gp.MAX_VALUE)

    def crossover(self, target, mutant) -> np.ndarray:
        return np.array([mutant[i] if np.random.rand() < self.CR else target[i] for i in range(gp.DIMENSIONS)])

    def adapt(self, exploration: int, exploitation: int) -> None:
        self.F *= exploration * exploitation
        self.CR *= exploration * exploitation
    
    def execute(self):
        for iteration in range(self.__class__.iterations):
            r1, r2, r3 = self.supervisor.get_parents(size=3)
            mutant = self.mutate(r1.position, r2.position, r3.position)
            trial = self.crossover(self.local_best.position, mutant)
            trial_value = gp.OBJECTIVE_FUNCTION(trial)
            if trial_value < self.local_best.value:
                self.local_best = Solution(trial, trial_value)
                if self.global_best_agent_type.value > self.local_best.value:
                    self.supervisor.update_global_best(self.local_best, self.__class__)

        self.supervisor.collect_results(self, self.local_best.value)
