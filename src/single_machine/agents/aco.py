from __future__ import annotations

import numpy as np

from .agent import ParticleAgent
from ...utils.custom_objects.data_classes import Solution
from ...utils import global_parameters as gp


class ACOAgent(ParticleAgent):
    def __init__(self, supervisor: 'Supervisor'):
        super().__init__(supervisor)

        self.num_ants = 10
        self.evaporation_rate = 0.5
        self.alpha = 1.0
        self.beta = 2.0

        self.pheromones = np.ones((gp.DIMENSIONS,))
        self.heuristic = np.ones((gp.DIMENSIONS,))
        self.local_best = Solution()
        self.childs = []

    def get_childs(self):
        return self.childs

    def adapt(self, exploration: int, exploatation: int):
        self.alpha *= exploatation 
        self.beta *= exploration
    
    def _generate_solution(self):
        position = np.array([
            np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE)
            if np.random.rand() > self._pheromone_prob(i) or self.local_best.position is None
            else self.local_best.position[i]
            for i in range(gp.DIMENSIONS)
        ])
        return Solution(position, gp.OBJECTIVE_FUNCTION(position))

    def _pheromone_prob(self, i):
        tau = self.pheromones[i] ** self.alpha
        eta = self.heuristic[i] ** self.beta
        return tau * eta / (tau * eta + 1e-9)

    def _update_pheromones(self, best_solution: Solution):
        delta_pheromones = 1.0 / (1.0 + best_solution.value)
        self.pheromones = (1 - self.evaporation_rate) * self.pheromones + delta_pheromones

    def execute(self) -> None:
        self.childs = []
        for iteration in range(gp.ACO_ITERATIONS):
            solutions = [self._generate_solution() for _ in range(self.num_ants)]
            best_solution = min(solutions, key=lambda s: s.value)
            self._update_pheromones(best_solution)

            if self.local_best.position is None or best_solution.value < self.local_best.value:
                self.local_best = best_solution
                if self.global_best_agent_type.value > self.local_best.value:
                    self.supervisor.update_global_best(self.local_best, self.__class__)

            self.childs.append(best_solution)

        self.supervisor.collect_results(self.__class__, self.local_best.value)