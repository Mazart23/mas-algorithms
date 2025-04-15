from __future__ import annotations
import queue

import numpy as np

from .agent import ParticleAgent
from ...utils.custom_objects.data_classes import Solution
from ...utils import global_parameters as gp


class GAAgent(ParticleAgent):
    def __init__(self, supervisor: 'Supervisor'):
        super().__init__(supervisor)
        
        self.local_best: Solution | None = None
        self.parent1: Solution | None = None
        self.parent2: Solution | None = None
        self.offsprings_queue: queue.PriorityQueue[Solution] = queue.PriorityQueue()
        self.childs: list[Solution] = []

    def get_childs(self):
        return self.childs
    
    def get_local_best(self) -> float:
        return self.local_best.value
    
    def set_local_best(self) -> None:
        self.childs = list(self.offsprings_queue.queue)
        self.local_best = self.childs[0]

    def crossover(self) -> np.ndarray[float]:
        return np.array([parent.position[dim] for dim, parent in zip(range(gp.DIMENSIONS), np.random.choice([self.parent1, self.parent2], size=gp.DIMENSIONS))])

    def mutate(self, offspring: np.ndarray[float]) -> np.ndarray[float]:
        mutation_vector = np.random.uniform(-0.5, 0.5, (gp.DIMENSIONS,)) * (np.random.rand(gp.DIMENSIONS) < gp.MUTATION_RATE)
        return offspring + mutation_vector

    def execute(self) -> None:
        for iteration in range(gp.GA_ITERATIONS):
            self.parent1, self.parent2 = self.supervisor.get_parents()
            offspring = self.crossover()
            offspring = self.mutate(offspring)
            offspring_score = gp.OBJECTIVE_FUNCTION(offspring)
            solution = Solution(offspring, offspring_score)
            self.offsprings_queue.put(solution)
            # self.supervisor.update_childs(solution)
        self.set_local_best()
        self.supervisor.collect_results(self.__class__, self.local_best.value)