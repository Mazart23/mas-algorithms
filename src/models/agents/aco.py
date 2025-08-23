from __future__ import annotations

import numpy as np

from .agent import ParticleAgent
from ...utils.custom_objects.data_classes import Solution
from ...utils import global_parameters as gp


class ACOAgent(ParticleAgent):
    iterations = gp.ACO_ITERATIONS
    
    def __init__(self, supervisor: 'Supervisor'):
        super().__init__(supervisor)

        self.alpha = gp.ALPHA_ACO
        self.beta = gp.BETA_ACO

        self.local_best = Solution()

    def adapt(self, exploration: int, exploitation: int):
        self.alpha *= exploitation 
        self.beta *= exploration
    
    def generate_solution(self):
        if gp.IS_DISCRETE:
            position = np.array([
                np.random.choice(
                    np.linspace(gp.MIN_VALUE, gp.MAX_VALUE, gp.DISCRETE_POINTS),
                    p=self.pheromone_prob(i)
                )
                for i in range(gp.DIMENSIONS)
            ])
        else:
            position = np.array([
                np.random.normal(self.supervisor.means[i], self.supervisor.sigmas[i])
                for i in range(gp.DIMENSIONS)
            ])
            position = np.clip(position, gp.MIN_VALUE, gp.MAX_VALUE)
        
        return Solution(position, gp.OBJECTIVE_FUNCTION(position))

    def pheromone_prob(self, i):
        tau = self.supervisor.pheromones[i] ** self.alpha
        eta = self.supervisor.heuristic[i] ** self.beta
        probs = tau * eta
        return probs / np.sum(probs)

    def execute(self) -> None:
        for iteration in range(self.__class__.iterations):
            solution = self.generate_solution()

            if self.local_best.position is None or solution.value < self.local_best.value:
                self.local_best = solution
                if self.global_best_agent_type.value > self.local_best.value:
                    self.supervisor.update_global_best(self.local_best, self.__class__)

        self.supervisor.collect_results(self, self.local_best.value)
