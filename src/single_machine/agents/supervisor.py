from __future__ import annotations
import time
import threading
import copy

import numpy as np

from ...utils.custom_objects.data_classes import Solution
from ...utils.custom_objects.enums import AgentType
from ...utils import global_parameters as gp


class Supervisor:
    def __init__(self, adapt: bool = False):
        self.adapt: bool = adapt
        
        len_agent_types = len(AgentType)
        get_num_agents = (gp.NUM_AGENTS // len_agent_types + (1 if x < gp.NUM_AGENTS % len_agent_types else 0) for x in range(len_agent_types))
        self.num_agents: dict[AgentType, int] = {
            agent_type: num for agent_type, num in zip(AgentType, get_num_agents)
        }
        self.particle_agents: dict[AgentType, list['ParticleAgent']] = {
            agent: [] for agent in AgentType
        }
        self.is_running: dict[AgentType, bool] = {
            agent: False for agent in AgentType
        }
        
        self.global_best: Solution = Solution()
        self.agent_type_best: AgentType[str, Solution] = {
            agent: Solution() for agent in AgentType
        }
        
        self.population: list[Solution] = []
        self.childs: list[Solution] = []
        self.possible_pairs: int = 0
        self.probabilities: list[float] = []
        
        self._lock_population = threading.Lock()
        self._lock_probabilities = threading.Lock()
        self._lock_global_best = threading.Lock()
        self._lock_global_best_agents: dict[AgentType, threading.Lock] = {
            agent: threading.Lock() for agent in AgentType
        }
        self._lock_particle_agents: dict[AgentType, threading.Lock] = {
            agent: threading.Lock() for agent in AgentType
        }
        
        self.performance: dict[AgentType, list[float]] = {
            agent: [] for agent in AgentType
        }
    
    def initialize_global_best(self):
        curr_global_best: Solution = Solution()
        for agent_type in AgentType:
            agent_type_best_temp = copy.deepcopy(min([agent.local_best for agent in self.particle_agents[agent_type]], key=lambda s: s.value))
            self.agent_type_best[agent_type] = agent_type_best_temp
            if agent_type_best_temp.value < curr_global_best.value:
                self.global_best = agent_type_best_temp
        self.announce_global_best()
    
    def initialize_population(self):
        self.population = sorted([Solution(pos := np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, (gp.DIMENSIONS,)), gp.OBJECTIVE_FUNCTION(pos)) for _ in range(gp.GA_POPULATION)])
        self.calculate_possible_pairs()
        self.calculate_probabilities()

    def initialize_agents(self):
        for agent_type in AgentType:
            new_agent_obj_lst = [agent_type.value(self) for _ in range(self.num_agents[agent_type])]
            with self._lock_particle_agents[agent_type]:
                self.particle_agents[agent_type] += new_agent_obj_lst
    
    def add_agents(self, agent_type: AgentType, num_to_add: int):
        agent_obj_lst = [agent_type.value(self) for _ in range(num_to_add)]
        with self._lock_particle_agents[agent_type]:
            self.particle_agents[agent_type] += agent_obj_lst
        for agent in agent_obj_lst:
            agent.start()
            agent.set_global_best(self.global_best)
    
    def remove_agents(self, agent_type: AgentType, num_to_remove: int):
        with self._lock_particle_agents[agent_type]:
            for _ in range(num_to_remove):
                particle_agents_obj = self.particle_agents[agent_type]
                worst_agent = max(particle_agents_obj)
                particle_agents_obj.remove(worst_agent)
                worst_agent.kill()
    
    def update_global_best(self, new_best_candidate: Solution, agent_type: AgentType):
        if new_best_candidate.value < self.agent_type_best[agent_type].value:
            copy_solution = copy.deepcopy(new_best_candidate)
            with self._lock_global_best_agents[agent_type]:
                self.agent_type_best[agent_type] = copy_solution
            self.announce_global_best_agent_type(agent_type)
            if new_best_candidate.value < self.global_best.value:
                copy_solution = copy.deepcopy(copy_solution)
                with self._lock_global_best:
                    self.global_best = copy_solution
                self.announce_global_best()
                print(f'[Supervisor] New global best: {self.global_best.value:.4f} at {self.global_best.position}')
    
    def announce_global_best_agent_type(self, agent_type: AgentType):
        with self._lock_particle_agents[agent_type]:
            for particle_agent in self.particle_agents[agent_type]:
                particle_agent.set_global_best_agent_type(self.global_best)
    
    def announce_global_best(self):
        agent_type = AgentType.PSO
        with self._lock_particle_agents[agent_type]:
            for particle_agent in self.particle_agents[agent_type]:
                particle_agent.set_global_best(self.global_best)

    def fetch_childs(self):
        for agent in self.particle_agents[AgentType.GA]:
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
        n_parents = int(gp.PARENTS_PERCENTAGE * population_length) + 1
        n_childs = int(gp.CHILDREN_PERCENTAGE * population_length) + 1
        n_random = population_length - n_parents - n_childs

        best_ones = self.population[:n_parents] + self.childs[:n_childs]
        others = self.population[n_parents:] + self.childs[n_childs:]

        self.population = sorted(best_ones + list(np.random.choice(others, size=n_random, replace=False)))
        self.childs = []
    
    def collect_results(self, agent_type_class: type, best_value: float):
        self.performance[AgentType(agent_type_class)].append(best_value)
        
    def adjust_agent_ratio(self):
        avg_pso = np.mean(self.performance[AgentType.PSO][-self.num_pso:])
        avg_ga = np.mean(self.performance[AgentType.GA][-self.num_ga:])

        num_to_change = max(int(min(self.num_pso, self.num_ga) * gp.ADAPTATION_SPEED), 1)
        
        if avg_ga > avg_pso * (1 + gp.ADAPTATION_CHANGE_TOLERATION) and self.num_ga > 1:
            num_to_change = min(num_to_change, self.num_ga - 1)
            self.num_pso += num_to_change
            self.num_ga -= num_to_change
            self.add_agents(AgentType.PSO, num_to_change)
            self.remove_agents(AgentType.GA, num_to_change)
        elif avg_pso > avg_ga * (1 + gp.ADAPTATION_CHANGE_TOLERATION) and self.num_pso > 1:
            num_to_change = min(num_to_change, self.num_pso - 1)
            self.num_ga += num_to_change
            self.num_pso -= num_to_change
            self.add_agents([AgentType.GA.value(self) for _ in range(num_to_change)])
            self.remove_agents(AgentType.PSO, num_to_change)
    
    def start_agents(self):
        for agent_type in AgentType:
            self.is_running[agent_type] = True
            for agent in self.particle_agents[agent_type]:
                agent.go()
    
    def run_agents(self):
        for agent_type in AgentType:
            for agent in self.particle_agents[agent_type]:
                agent.start()
    
    def wait_for_agents(self):
        while True:
            for agent_type in AgentType:
                if self.is_running[agent_type] and not any(agent.event.is_set() for agent in self.particle_agents[agent_type]):
                    self.is_running[agent_type] = False
                    print(f'####### {agent_type.name} STOPPED')
                if not any(self.is_running.values()):
                    break
            time.sleep(0.01)