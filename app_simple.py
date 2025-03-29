from __future__ import annotations
import sys
import time
import threading
import copy
from dataclasses import dataclass, field
import uuid
from enum import Enum
import queue
import ctypes

import numpy as np


# general parameters
NUM_AGENTS = 400
ITERATIONS = 10
ADAPTATION_SPEED = 0.1
ADAPTATION_CHANGE_TOLERATION = 0.05

# PSO parameters
PSO_ITERATIONS = 200
W = 1  # Inertion
C1 = 1.5  # weight for best local position
C2 = 1  # weight for best global position

# GA parameters
GA_ITERATIONS = 200
GA_POPULATION = 10
CROSSOVER_PROB = 0.9
MUTATION_RATE = 0.1
PARENTS_PERCENTAGE = 0.2
CHILDREN_PERCENTAGE = 0.5

# function parameters
DIMENSIONS = 10
MIN_VALUE = -5.12
MAX_VALUE = 5.12


class AgentType(Enum):
    PSO = 0
    GA = 1


def rastrigin(x):
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

def sphere(x):
    return sum(xi**2 for xi in x)

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(c * xi) for xi in x)
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e

def griewank(x):
    sum1 = sum(xi**2 / 4000 for xi in x)
    prod = np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])
    return sum1 - prod + 1

def one_max(x):
    x = np.round(x).astype(int)
    return -sum(x)

OBJECTIVE_FUNCTIONS = {
    'rastrigin': rastrigin,
    'sphere': sphere,
    'ackley': ackley,
    'griewank': griewank
}

@dataclass(order=True)
class Solution:
    position: np.ndarray | None = field(default=None, compare=False)
    value: float = float('inf')
    solution_id: uuid.UUID = field(default_factory=uuid.uuid4, compare=False)

class SwarmSupervisor:
    def __init__(self, num_pso: int, num_ga: int, adapt: bool = False):
        self.num_pso: int = num_pso
        self.num_ga: int = num_ga
        self.adapt: bool = adapt

        # self.particle_agents: list[ParticleAgent] = []
        self.particle_agents_pso: list[ParticleAgent] = []
        self.particle_agents_ga: list[ParticleAgent] = []
        
        self.is_running: dict[str, bool] = {
            'PSO': False,
            'GA': False
        }
        
        self.global_best = Solution()
        
        self.population: list[Solution] = []
        self.childs: list[Solution] = []
        self.possible_pairs: int = 0
        self.probabilities: list[float] = []
        
        self._lock_population = threading.Lock()
        self._lock_probabilities = threading.Lock()
        self._lock_global_best = threading.Lock()
        # self._lock_particle_agents = threading.Lock()
        self._lock_particle_agents_pso = threading.Lock()
        self._lock_particle_agents_ga = threading.Lock()
        
        self.performance = {'PSO': [], 'GA': []}
    
    def initialize_global_best(self):
        if self.particle_agents_pso:
            self.global_best = copy.deepcopy(min([agent.local_best for agent in self.particle_agents_pso], key=lambda s: s.value))
            self.announce_global_best()
    
    def initialize_population(self):
        self.population = sorted([Solution(pos := np.random.uniform(MIN_VALUE, MAX_VALUE, DIMENSIONS), OBJECTIVE_FUNCTION(pos)) for _ in range(GA_POPULATION)])
        self.calculate_possible_pairs()
        self.calculate_probabilities()

    def initialize_agents(self, agent_obj_lst_pso: list[PSOAgent], agent_obj_lst_ga: list[GAAgent]):
        with self._lock_particle_agents_pso:
            self.particle_agents_pso += agent_obj_lst_pso
        with self._lock_particle_agents_ga:
            self.particle_agents_ga += agent_obj_lst_ga
    
    def add_agents(self, agent_obj_lst: list[ParticleAgent]):
        if agent_obj_lst:
            if isinstance(agent_obj_lst[0], PSOAgent):
                with self._lock_particle_agents_pso:
                    self.particle_agents_pso += agent_obj_lst
                for agent in agent_obj_lst:
                    agent.start()
                    agent.set_global_best(copy.deepcopy(self.global_best))
            elif isinstance(agent_obj_lst[0], GAAgent):
                with self._lock_particle_agents_ga:
                    self.particle_agents_ga += agent_obj_lst
                for agent in agent_obj_lst:
                    agent.start()
    
    def remove_agents(self, agent_type: str, num_to_remove: int):
        match agent_type:
            case 'PSO':
                with self._lock_particle_agents_pso:
                    for _ in range(num_to_remove):
                        worst_agent = max(self.particle_agents_pso)
                        self.particle_agents_pso.remove(worst_agent)
                        worst_agent.kill()
            case 'GA':
                with self._lock_particle_agents_ga:
                    for _ in range(num_to_remove):
                        worst_agent = max(self.particle_agents_ga)
                        self.particle_agents_ga.remove(worst_agent)
                        worst_agent.kill()
    
    def update_global_best(self, new_best_candidate: Solution):
        if new_best_candidate.value < self.global_best.value:
            with self._lock_global_best:
                self.global_best = copy.deepcopy(new_best_candidate)
            self.announce_global_best()
            print(f'[SwarmSupervisor] New global best: {self.global_best.value:.4f} at {self.global_best.position}')
    
    def announce_global_best(self):
        with self._lock_particle_agents_pso:
            for particle_agent in self.particle_agents_pso:
                particle_agent.set_global_best(copy.deepcopy(self.global_best))

    def fetch_childs(self):
        for agent in self.particle_agents_ga:
            agent_childs = agent.get_childs()
            self.childs += agent_childs
    
    def get_parents(self):
        with self._lock_probabilities:
            parent1, parent2 = np.random.choice(self.population, p=self.probabilities, replace=False, size=2)
        return parent1, parent2
    
    def calculate_possible_pairs(self):
        population_length = len(self.population)
        self.possible_pairs = (population_length * (population_length + 1)) / 2

    def calculate_probabilities(self):
        self.probabilities = [i / self.possible_pairs for i in range(1, len(self.population) + 1)]

    def select_population(self) -> None:
        self.fetch_childs()
        self.childs.sort()
        population_length = len(self.population)
        n_parents = int(PARENTS_PERCENTAGE * population_length) + 1
        n_childs = int(CHILDREN_PERCENTAGE * population_length) + 1
        n_random = population_length - n_parents - n_childs

        best_ones = self.population[:n_parents] + self.childs[:n_childs]
        others = self.population[n_parents:] + self.childs[n_childs:]

        self.population = sorted(best_ones + list(np.random.choice(others, size=n_random, replace=False)))
        self.childs = []
    
    def collect_results(self, agent_type: str, best_value: float):
        self.performance[agent_type].append(best_value)
        
    def adjust_agent_ratio(self):
        avg_pso = np.mean(self.performance['PSO'][-self.num_pso:])
        avg_ga = np.mean(self.performance['GA'][-self.num_ga:])

        num_to_change = int(min(self.num_pso, self.num_ga) * ADAPTATION_SPEED)
        if num_to_change == 0:
            num_to_change += 1
        
        if avg_ga > avg_pso * (1 + ADAPTATION_CHANGE_TOLERATION) and self.num_ga > 1:
            num_to_change = min(num_to_change, self.num_ga - 1)
            self.num_pso += num_to_change
            self.num_ga -= num_to_change
            self.add_agents([PSOAgent(self) for _ in range(num_to_change)])
            self.remove_agents('GA', num_to_change)
        elif avg_pso > avg_ga * (1 + ADAPTATION_CHANGE_TOLERATION) and self.num_pso > 1:
            num_to_change = min(num_to_change, self.num_pso - 1)
            self.num_ga += num_to_change
            self.num_pso -= num_to_change
            self.add_agents([GAAgent(self) for _ in range(num_to_change)])
            self.remove_agents('PSO', num_to_change)
    
    def start_agents(self):
        self.is_running['PSO'] = True
        self.is_running['GA'] = True
        for agent in self.particle_agents_pso:
            agent.go()
        for agent in self.particle_agents_ga:
            agent.go()
    
    def run_agents(self):
        for agent in self.particle_agents_pso:
            agent.start()
        for agent in self.particle_agents_ga:
            agent.start()
    
    def wait_for_agents(self):
        while True:
            if self.is_running['PSO'] and not any(agent.event.is_set() for agent in self.particle_agents_pso):
                self.is_running['PSO'] = False
                print('####### PSO STOPPED')
            if self.is_running['GA'] and not any(agent.event.is_set() for agent in self.particle_agents_ga):
                self.is_running['GA'] = False
                print('####### GA STOPPED')
            if not self.is_running['PSO'] and not self.is_running['GA']:
                break
            time.sleep(0.01)


class ParticleAgent(threading.Thread):
    def __init__(self, supervisor: SwarmSupervisor):
        self.agent_id = uuid.uuid4()
        super().__init__(daemon=True)
        self.event = threading.Event()
        self.supervisor: SwarmSupervisor = supervisor
        self.local_best: Solution | None = None
    
    def __hash__(self):
        return hash(self.agent_id)

    def __eq__(self, other):
        return self.local_best.value == other.local_best.value
    
    def __lt__(self, other):
        return self.local_best.value < other.local_best.value
    
    def __gt__(self, other):
        return self.local_best.value > other.local_best.value
    
    def execute(self) -> None:
        pass
    
    def run(self):
        while True:
            self.event.wait()
            self.execute()
            self.event.clear()
    
    def go(self):
        self.event.set()
        
    def stop(self):
        self.event.clear()
    
    def kill(self):
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(self.ident), ctypes.py_object(SystemExit)
        )

class PSOAgent(ParticleAgent):
    def __init__(self, supervisor: SwarmSupervisor):
        super().__init__(supervisor)
        
        position = np.random.uniform(MIN_VALUE, MAX_VALUE, (DIMENSIONS,))
        self.velocities: np.ndarray[float] = np.random.uniform(-1, 1, (DIMENSIONS,))
        
        self.current: Solution = Solution(position, OBJECTIVE_FUNCTION(position))
        self.local_best: Solution | None = copy.deepcopy(self.current)
        self.global_best: Solution = Solution()

    def set_global_best(self, global_best: Solution):
        self.global_best = global_best
    
    def execute(self) -> None:
        for iteration in range(PSO_ITERATIONS):
            global_best_position = self.global_best.position
        
            self.velocities = (
                W * self.velocities +
                C1 * np.random.rand(DIMENSIONS) * (self.local_best.position - self.current.position) +
                C2 * np.random.rand(DIMENSIONS) * (global_best_position - self.current.position)
            )
            self.current.position += self.velocities
            self.current.value = OBJECTIVE_FUNCTION(self.current.position)
            
            if self.current.value < self.local_best.value:
                self.local_best = copy.deepcopy(self.current)
                if self.global_best.value > self.local_best.value:
                    self.supervisor.update_global_best(self.local_best)
            
            if iteration % 10 == 0:
                print(f'[AGENT {self.agent_id}] Iteration {iteration}, local best: {self.local_best.value:.4f}')
        
        self.supervisor.collect_results('PSO', self.local_best.value)
                
class GAAgent(ParticleAgent):
    def __init__(self, supervisor: SwarmSupervisor):
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

    def crossover(self) -> list[int]:
        return [parent.position[dim] for dim, parent in zip(range(DIMENSIONS), np.random.choice([self.parent1, self.parent2], size=DIMENSIONS))]

    def mutate(self, offspring: list[int]):
        mutation_vector = list(np.random.uniform(-0.5, 0.5, size=DIMENSIONS) * (np.random.rand(DIMENSIONS) < MUTATION_RATE))
        return offspring + mutation_vector

    def execute(self) -> None:
        for iteration in range(GA_ITERATIONS):
            self.parent1, self.parent2 = self.supervisor.get_parents()
            offspring = self.crossover()
            offspring = self.mutate(offspring)
            offspring_score = OBJECTIVE_FUNCTION(offspring)
            solution = Solution(offspring, offspring_score)
            self.offsprings_queue.put(solution)
            # self.supervisor.update_childs(solution)
            if iteration % 10 == 0:
                print(f'[AGENT {self.agent_id}] Iteration {iteration}')
        self.set_local_best()
        self.supervisor.collect_results('GA', self.local_best.value)


if __name__ == '__main__':
    function_name = 'ackley'
    adapt = False
    print(f'Is GIL enabled: {sys._is_gil_enabled()}')
    print(f'Objective function: {function_name}')
    print(f'Is adaptation enabled: {adapt}', end='')
    
    OBJECTIVE_FUNCTION = OBJECTIVE_FUNCTIONS[function_name]
    
    time_start = time.perf_counter()
    
    num_pso = NUM_AGENTS // 2
    num_ga = NUM_AGENTS - num_pso

    supervisor = SwarmSupervisor(num_pso, num_ga, adapt=adapt)
        
    supervisor.initialize_agents([PSOAgent(supervisor) for _ in range(num_pso)], [GAAgent(supervisor) for _ in range(num_ga)])
    supervisor.initialize_global_best()
    supervisor.initialize_population()

    supervisor.run_agents()
    
    for i in range(ITERATIONS):
        print(f'\n\n##############\nIteration: {i}')
        print(f'Number of agents:\n\tPSO: {supervisor.num_pso}\n\tGA: {supervisor.num_ga}\n')
        supervisor.start_agents()
        supervisor.wait_for_agents()

        supervisor.select_population()
        
        if supervisor.adapt:
            supervisor.adjust_agent_ratio()
    
    time_end = time.perf_counter()
    print(f'\nTime execution: {time_end - time_start}')
    print(f'Best global solution PSO: {supervisor.global_best.value:.4f} at {supervisor.global_best.position}')
    ga_best = min(supervisor.population)
    print(f'Best global solution GA: {ga_best.value:.4f} at {ga_best.position}')
