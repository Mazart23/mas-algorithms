from __future__ import annotations
import copy
import queue

import numpy as np

from .agent import ParticleAgent
from ...utils.custom_objects.data_classes import Solution
from ...utils import global_parameters as gp
from ...utils.functions import discrete


class ABCAgent(ParticleAgent):
    iterations = gp.ABC_ITERATIONS
    
    def __init__(self, supervisor: 'Supervisor'):
        super().__init__(supervisor)
        
        self.is_employeed: bool = gp.EMPLOYED_ABC_PERCENTAGE > np.random.uniform(0, 1)
        self.W = gp.W_ABC_EMPLOYEED if self.is_employeed else gp.W_ABC_SCOUT
        
        self.current: Solution = Solution(
            pos := discrete(np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, (gp.DIMENSIONS,))), 
            gp.OBJECTIVE_FUNCTION(pos)
        )
        self.local_best: Solution = copy.deepcopy(self.current)
        self.childs: list[Solution] = []
        
        self.offsprings_queue: queue.PriorityQueue[Solution] = queue.PriorityQueue()

    def get_childs(self):
        return self.childs
    
    def get_local_best(self) -> float:
        return self.local_best.value
    
    def set_local_best(self) -> None:
        self.childs = list(self.offsprings_queue.queue)
        self.local_best = self.childs[0]
    
    def explore_neighbourhood(self, position) -> np.ndarray:
        phi = np.random.uniform(-1, 1, gp.DIMENSIONS)
        partner, = self.supervisor.get_parents(size=1)
        return discrete(np.clip(position + self.W * phi * (position - partner.position), gp.MIN_VALUE, gp.MAX_VALUE))

    def determine_type(self) -> None:
        if self.supervisor.abc_border_performance:
            self.is_employeed = self.supervisor.abc_border_performance > self.local_best.value
        self.W = gp.W_ABC_EMPLOYEED if self.is_employeed else gp.W_ABC_SCOUT

    def adapt(self, exploration: int, exploitation: int):
        self.W *= exploration
    
    def execute(self):
        self.determine_type()
        if self.is_employeed:
            for iteration in range(self.__class__.iterations):
                new_position = self.explore_neighbourhood(self.current.position)
                new_value = gp.OBJECTIVE_FUNCTION(new_position)
                offspring = Solution(new_position, new_value)
                self.offsprings_queue.put(offspring)
                if new_value < self.current.value:
                    self.current = Solution(new_position, new_value)
                    if new_value < self.local_best.value:
                        self.local_best = copy.deepcopy(self.current)
                        if self.global_best_agent_type.value > self.local_best.value:
                            self.supervisor.update_global_best(self.local_best, self.__class__)
        else:
            for iteration in range(self.__class__.iterations):
                new_candidate, = self.supervisor.get_parents(size=1)
                new_position = self.explore_neighbourhood(new_candidate.position)
                offspring = Solution(new_position, gp.OBJECTIVE_FUNCTION(new_position))
                self.offsprings_queue.put(offspring)
            self.set_local_best()
            if self.local_best.value < self.global_best_agent_type.value:
                self.supervisor.update_global_best(self.local_best, self.__class__)

        self.supervisor.collect_results(self, self.local_best.value)
