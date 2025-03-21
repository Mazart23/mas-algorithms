from __future__ import annotations
import sys
import time
import threading
import copy
from dataclasses import dataclass, field
import uuid
import queue

import numpy as np

# general parameters
NUM_AGENTS = 5
ITERATIONS = 50
ADAPTATION_SPEED = 0.1

# PSO parameters
PSO_ITERATIONS = 50
W = 0.7  # Inertion
C1 = 1.5  # weight for best local position
C2 = 1.5  # weight for best global position

# GA parameters
GA_ITERATIONS = 50
GA_POPULATION = 10
CROSSOVER_PROB = 0.5
MUTATION_RATE = 0.1
PARENTS_PERCENTAGE = 0.2
CHILDREN_PERCENTAGE = 0.5

# function parameters
DIMENSIONS = 10
MIN_VALUE = -5.12
MAX_VALUE = -5.12

def rastrigin(x):
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

@dataclass(order=True)
class Solution:
    position: np.ndarray | None = field(default=None, compare=False)
    value: float = float('inf')
    solution_id: uuid.UUID = field(default_factory=uuid.uuid4, compare=False)

class SwarmSupervisor:
    def __init__(self, agents: list[ParticleAgent] = []):
        self.particle_agents: list[ParticleAgent] = agents
        
        self.global_best = Solution()
        
        self.population: list[Solution] = []
        self.childs = queue.PriorityQueue()
        
        self._lock_population = threading.Lock()
        self._lock_global_best = threading.Lock()
        self._lock_particle_agents = threading.Lock()
        
        self.performance = {"PSO": [], "GA": []}
    
    def initialize_global_best(self):
        if self.particle_agents:
            self.global_best = copy.deepcopy(min([agent.local_best for agent in self.particle_agents], key=lambda s: s.value))
            self.announce_global_best()
    
    def initialize_population(self):
        self.population = sorted([Solution(pos := np.random.uniform(MIN_VALUE, MAX_VALUE, DIMENSIONS), rastrigin(pos)) for _ in range(GA_POPULATION)])
    
    def add_agents(self, agent_obj_lst: list[ParticleAgent]):
        with self._lock_particle_agents:
            self.particle_agents += agent_obj_lst
    
    def remove_agent(self, agent_obj: ParticleAgent):
        with self._lock_particle_agents:
            self.particle_agents.remove(agent_obj)
    
    def update_global_best(self, new_best_candidate: Solution):
        if new_best_candidate.value < self.global_best.value:
            with self._lock_global_best:
                self.global_best = copy.deepcopy(new_best_candidate)
            self.announce_global_best()
            print(f"[SwarmSupervisor] New global best: {self.global_best.value:.4f} at {self.global_best.position}")
    
    def announce_global_best(self):
        with self._lock_particle_agents:
            for particle_agent in self.particle_agents:
                particle_agent.set_global_best(copy.deepcopy(self.global_best))
                
    def update_childs(self, new_child: Solution):
        self.childs.put(new_child)
    
    def create_pairs(self):
        pass
    
    def select_population(self):
        n_parents = int(PARENTS_PERCENTAGE * GA_POPULATION) + 1
        n_childs = int(CHILDREN_PERCENTAGE * GA_POPULATION) + 1
        n_random = GA_POPULATION - n_parents - n_childs

        best_ones = self.population[:n_parents] + self.childs[:n_childs]
        others = self.population[n_parents:] + self.childs[n_childs:]

        self.population = best_ones + np.ndarray.tolist(np.random.choice(others, size=n_random, replace=False))
    
    def collect_results(self, agent_type: str, best_value: float):
        self.performance[agent_type].append(best_value)
        
    def adjust_agent_ratio(self, num_pso: int, num_ga: int):
        if num_pso > 1 and num_ga > 1 and len(self.performance["PSO"]) > 0 and len(self.performance["GA"]) > 0:
            avg_pso = np.mean(self.performance["PSO"][-num_pso:])
            avg_ga = np.mean(self.performance["GA"][-num_ga:])
    
            num_to_change = int(min(num_pso, num_ga) * ADAPTATION_SPEED)
            if num_to_change == 0:
                num_to_change += 1
            
            if avg_ga > avg_pso:
                num_pso -= num_to_change
                num_ga += num_to_change
            else:
                num_ga -= num_to_change
                num_pso += num_to_change

        return num_pso, num_ga

class ParticleAgent(threading.Thread):
    def __init__(self, supervisor: SwarmSupervisor):
        self.agent_id = uuid.uuid4()
        super().__init__()
        self.supervisor: SwarmSupervisor = supervisor
    
    def set_global_best(self, global_best: Solution):
        self.global_best = global_best


class PSOAgent(ParticleAgent):
    def __init__(self, supervisor: SwarmSupervisor):
        super().__init__(supervisor)
        
        position = np.random.uniform(MIN_VALUE, MAX_VALUE, (DIMENSIONS,))
        self.velocities: np.ndarray[float] = np.random.uniform(-1, 1, (DIMENSIONS,))
        
        self.current: Solution = Solution(position, rastrigin(position))
        self.local_best: Solution = copy.deepcopy(self.current)
        self.global_best: Solution = Solution()
    
    def run(self):            
        for iteration in range(PSO_ITERATIONS):
            global_best_position = self.global_best.position
        
            self.velocities = (
                W * self.velocities +
                C1 * np.random.rand(DIMENSIONS) * (self.local_best.position - self.current.position) +
                C2 * np.random.rand(DIMENSIONS) * (global_best_position - self.current.position)
            )
            self.current.position += self.velocities
            self.current.value = rastrigin(self.current.position)
            
            if self.current.value < self.local_best.value:
                self.local_best = copy.deepcopy(self.current)
                if self.global_best.value > self.local_best.value:
                    self.supervisor.update_global_best(self.local_best)
            
            if iteration % 10 == 0:
                print(f"[AGENT {self.agent_id}] Iteration {iteration}, local best: {self.local_best.value:.4f}")
        
        self.supervisor.collect_results("PSO", self.local_best.value)
                
class GAAgent(ParticleAgent):
    def __init__(self, supervisor: SwarmSupervisor, parent1: Solution, parent2: Solution):
        super().__init__(supervisor)
        self.parent1: Solution | None = parent1
        self.parent2: Solution | None = parent2

    def pass_parents(self, parent1: Solution, parent2: Solution):
        self.parent1 = parent1
        self.parent2 = parent2
    
    def select_population(pop_init: list[Solution], pop_current) -> list[Solution]:
        pop_init = sorted(pop_init, key=lambda s: s.fitness, reverse=False)
        pop_current = sorted(pop_current, key=lambda s: s.fitness, reverse=False)

        n_parents = int(PARENTS_PERCENTAGE * GA_POPULATION) + 1
        n_children = int(CHILDREN_PERCENTAGE * GA_POPULATION) + 1
        n_random = GA_POPULATION - n_parents - n_children

        best_ones = pop_init[:n_parents] + pop_current[:n_children]
        others = pop_init[n_parents:] + pop_current[n_children:]

        next_pop = best_ones + np.ndarray.tolist(np.random.choice(others, size=n_random, replace=False))

        return next_pop

    def crossover(self, parent1, parent2):
        return [parent.position[dim] for dim, parent in zip(DIMENSIONS, np.random.choice([parent1, parent2], size=DIMENSIONS))]

    def mutate(self, offspring):
        if np.random.rand() < MUTATION_RATE:
            idx = np.random.randint(0, DIMENSIONS)
            offspring[idx] += np.random.uniform(-0.5, 0.5)
        return offspring

    def run(self):
        for _ in range(GA_ITERATIONS):
            self.select_population()
            child = self.crossover()
            offspring = self.mutate(offspring)
            offspring_score = rastrigin(offspring)

            worst_idx = np.argmax(self.scores)
            if offspring_score < self.scores[worst_idx]:
                self.population[worst_idx] = offspring
                self.scores[worst_idx] = offspring_score

        best_idx = np.argmin(self.scores)
        best_solution = Solution(self.population[best_idx], self.scores[best_idx])
        self.supervisor.update_global_best(best_solution)
        self.supervisor.collect_results("GA", best_solution.value)


if __name__ == "__main__":
    print(f"Is GIL enabled: {sys._is_gil_enabled()}\n")
    time_start = time.perf_counter()
    
    supervisor = SwarmSupervisor()
    
    num_pso = NUM_AGENTS // 2
    num_ga = NUM_AGENTS - num_pso
    
    agents = [PSOAgent(supervisor) for _ in range(num_pso)] + [GAAgent(supervisor) for _ in range(num_ga)]
    
    for i in range(ITERATIONS):
        agents = [PSOAgent(supervisor) for _ in range(num_pso)] + [GAAgent(supervisor) for _ in range(num_ga)]
        supervisor.add_agents(agents)
        supervisor.initialize_global_best()
        supervisor.initialize_population()
        
        for agent in agents:
            agent.start()
        
        for agent in agents:
            agent.join()
        
        supervisor.select_population()
            
        num_pso, num_ga = supervisor.adjust_agent_ratio(num_pso, num_ga)
    
    time_end = time.perf_counter()
    print(f'\nTime execution: {time_end - time_start}')
    print(f'Best global solution: {supervisor.global_best.value:.4f} at {supervisor.global_best.position}')