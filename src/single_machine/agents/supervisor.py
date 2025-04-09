from __future__ import annotations
import time
import threading
import copy

import numpy as np

from .agent import ParticleAgent
from .pso import PSOAgent
from .ga import GAAgent
from ...utils.data_classes import Solution, AgentType
from ...utils import global_parameters as gp


class Supervisor:
    def __init__(self, adapt: bool = False):
        len_agent_types = len(AgentType)
        get_num_agents = (gp.NUM_AGENTS // len_agent_types + (1 if x < gp.NUM_AGENTS % len_agent_types else 0) for x in range(len_agent_types))
        self.num_pso: int = next(get_num_agents)
        self.num_ga: int = next(get_num_agents)
        self.adapt: bool = adapt

        self.particle_agents_pso: list[ParticleAgent] = []
        self.particle_agents_ga: list[ParticleAgent] = []
        
        self.is_running: dict[str, bool] = {
            AgentType.PSO: False,
            AgentType.GA: False
        }
        
        self.global_best = Solution()
        
        self.population: list[Solution] = []
        self.childs: list[Solution] = []
        self.possible_pairs: int = 0
        self.probabilities: list[float] = []
        
        self._lock_population = threading.Lock()
        self._lock_probabilities = threading.Lock()
        self._lock_global_best = threading.Lock()
        self._lock_particle_agents_pso = threading.Lock()
        self._lock_particle_agents_ga = threading.Lock()
        
        self.performance = {AgentType.PSO: [], AgentType.GA: []}
    
    def initialize_global_best(self):
        if self.particle_agents_pso:
            self.global_best = copy.deepcopy(min([agent.local_best for agent in self.particle_agents_pso], key=lambda s: s.value))
            self.announce_global_best()
    
    def initialize_population(self):
        self.population = sorted([Solution(pos := np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, (gp.DIMENSIONS,)), gp.OBJECTIVE_FUNCTION(pos)) for _ in range(gp.GA_POPULATION)])
        self.calculate_possible_pairs()
        self.calculate_probabilities()

    def initialize_agents(self):
        agent_obj_lst_pso = [PSOAgent(self) for _ in range(self.num_pso)]
        agent_obj_lst_ga = [GAAgent(self) for _ in range(self.num_ga)]
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
            case AgentType.PSO:
                with self._lock_particle_agents_pso:
                    for _ in range(num_to_remove):
                        worst_agent = max(self.particle_agents_pso)
                        self.particle_agents_pso.remove(worst_agent)
                        worst_agent.kill()
            case AgentType.GA:
                with self._lock_particle_agents_ga:
                    for _ in range(num_to_remove):
                        worst_agent = max(self.particle_agents_ga)
                        self.particle_agents_ga.remove(worst_agent)
                        worst_agent.kill()
    
    def update_global_best(self, new_best_candidate: Solution, agent_type: str):
        if new_best_candidate.value < self.global_best.value:
            with self._lock_global_best:
                self.global_best = copy.deepcopy(new_best_candidate)
            self.announce_global_best()
            print(f'[Supervisor] New global best: {self.global_best.value:.4f} at {self.global_best.position}')
    
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
        n_parents = int(gp.PARENTS_PERCENTAGE * population_length) + 1
        n_childs = int(gp.CHILDREN_PERCENTAGE * population_length) + 1
        n_random = population_length - n_parents - n_childs

        best_ones = self.population[:n_parents] + self.childs[:n_childs]
        others = self.population[n_parents:] + self.childs[n_childs:]

        self.population = sorted(best_ones + list(np.random.choice(others, size=n_random, replace=False)))
        self.childs = []
    
    def collect_results(self, agent_type: str, best_value: float):
        self.performance[agent_type].append(best_value)
        
    def adjust_agent_ratio(self):
        avg_pso = np.mean(self.performance[AgentType.PSO][-self.num_pso:])
        avg_ga = np.mean(self.performance[AgentType.GA][-self.num_ga:])

        num_to_change = max(int(min(self.num_pso, self.num_ga) * gp.ADAPTATION_SPEED), 1)
        
        if avg_ga > avg_pso * (1 + gp.ADAPTATION_CHANGE_TOLERATION) and self.num_ga > 1:
            num_to_change = min(num_to_change, self.num_ga - 1)
            self.num_pso += num_to_change
            self.num_ga -= num_to_change
            self.add_agents([PSOAgent(self) for _ in range(num_to_change)])
            self.remove_agents(AgentType.GA, num_to_change)
        elif avg_pso > avg_ga * (1 + gp.ADAPTATION_CHANGE_TOLERATION) and self.num_pso > 1:
            num_to_change = min(num_to_change, self.num_pso - 1)
            self.num_ga += num_to_change
            self.num_pso -= num_to_change
            self.add_agents([GAAgent(self) for _ in range(num_to_change)])
            self.remove_agents(AgentType.PSO, num_to_change)
    
    def start_agents(self):
        self.is_running[AgentType.PSO] = True
        self.is_running[AgentType.GA] = True
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
            if self.is_running[AgentType.PSO] and not any(agent.event.is_set() for agent in self.particle_agents_pso):
                self.is_running[AgentType.PSO] = False
                print('####### PSO STOPPED')
            if self.is_running[AgentType.GA] and not any(agent.event.is_set() for agent in self.particle_agents_ga):
                self.is_running[AgentType.GA] = False
                print('####### GA STOPPED')
            if not self.is_running[AgentType.PSO] and not self.is_running[AgentType.GA]:
                break
            time.sleep(0.01)