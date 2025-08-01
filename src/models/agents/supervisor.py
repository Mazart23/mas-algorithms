from __future__ import annotations
import time
import threading
import copy
import heapq
import queue

import numpy as np
from matplotlib import pyplot as plt

from ...utils.custom_objects.data_classes import Solution
from ...utils.custom_objects.enums import AgentType
from ...utils import global_parameters as gp
from .adjuster import SoftmaxAdjuster


class Supervisor:
    def __init__(self, 
        adapt_num_agents: bool = False, 
        adapt_parameters: bool = False, 
        adapt_iterations: bool = False, 
        visualize_data: bool = False
    ) -> None:
        self.adapt_num_agents: bool = adapt_num_agents
        self.adapt_parameters: bool = adapt_parameters
        self.adapt_iterations: bool = adapt_iterations
        self.visualize_data: bool = visualize_data
        
        self.len_agent_types = len(AgentType)
        get_num_agents = (gp.NUM_AGENTS // self.len_agent_types + (1 if x < gp.NUM_AGENTS % self.len_agent_types else 0) for x in range(self.len_agent_types))
        self.num_agents: dict[AgentType, int] = {
            agent_type: num for agent_type, num in zip(AgentType, get_num_agents)
        }
        self.particle_agents: dict[AgentType, list['ParticleAgent']] = {
            agent: [] for agent in AgentType
        }

        self.q_agents_stopped: queue.Queue = queue.Queue()
        self.stopped_counter: dict[AgentType, int] = {
            agent_type: 0 for agent_type in AgentType
        }

        self.global_best: Solution = Solution()
        self.agent_type_best: dict[AgentType, Solution] = {
            agent: Solution() for agent in AgentType
        }
        
        self.population: list[Solution] = []
        self.childs: list[Solution] = []
        self.possible_pairs: int = 0
        self.probabilities: list[float] = []
        
        self.pheromones: np.ndarray = np.ones((gp.DIMENSIONS,))
        self.heuristic: np.ndarray = np.ones((gp.DIMENSIONS,))

        self.abc_border_performance: float = 0.0
        
        self._lock_population = threading.Lock()
        self._lock_probabilities = threading.Lock()
        self._lock_global_best = threading.Lock()
        self._lock_global_best_agents: dict[AgentType, threading.Lock] = {
            agent: threading.Lock() for agent in AgentType
        }
        self._lock_particle_agents: dict[AgentType, threading.Lock] = {
            agent: threading.Lock() for agent in AgentType
        }
        self._lock_performance = threading.Lock()
        
        self.performance: dict[AgentType, list[tuple['ParticleAgent', float]]] = {
            agent: [] for agent in AgentType
        }
        self.iteration_times: dict[AgentType, float] = {}

        self.avg_perfomance_history: list[dict[AgentType, float]] = []
        self.best_perfomance_history: list[dict[AgentType, float]] = []
        self.iteration_times_history: list[dict[AgentType, float]] = []
        self.nums_history: list[dict[AgentType, int]] = []
                
        self.num_agents_adjuster: SoftmaxAdjuster = SoftmaxAdjuster(self)
    
    def initialize_global_best(self) -> None:
        curr_global_best: Solution = Solution()
        for agent_type in AgentType:
            agent_type_best_temp = copy.deepcopy(min([agent.local_best for agent in self.particle_agents[agent_type]], key=lambda s: s.value))
            self.agent_type_best[agent_type] = agent_type_best_temp
            self.announce_global_best_agent_type(agent_type)
            if agent_type_best_temp.value < curr_global_best.value:
                self.global_best = agent_type_best_temp
        self.announce_global_best()
    
    def initialize_population(self) -> None:
        self.set_population(
            sorted([Solution(pos := np.random.uniform(gp.MIN_VALUE, gp.MAX_VALUE, (gp.DIMENSIONS,)), gp.OBJECTIVE_FUNCTION(pos)) for _ in range(gp.GA_POPULATION)])
        )
        self.calculate_possible_pairs()
        self.calculate_probabilities()

    def initialize_agents(self) -> None:
        for agent_type in AgentType:
            new_agent_obj_lst = [agent_type.value(self) for _ in range(self.num_agents[agent_type])]
            with self._lock_particle_agents[agent_type]:
                self.particle_agents[agent_type] += new_agent_obj_lst
    
    def set_population(self, population: list) -> None:
        self.population = population
        self.update_global_best(self.population[0], AgentType.GA.value)
        
    def update_pheromones(self) -> None:
        delta_pheromones = 1.0 / (1.0 + self.global_best.value)
        self.pheromones = (1 - gp.EVAPORATION_RATE_ACO) * self.pheromones + delta_pheromones

    def update_abc_border_performance(self) -> None:
        employed_agents_num = int(self.num_agents[AgentType.ABC] * gp.EMPLOYED_ABC_PERCENTAGE)
        self.abc_border_performance = [
            performance 
            for _, performance in self.performance[AgentType.ABC]
        ][employed_agents_num]
    
    def add_agents(self, agent_type: AgentType, num_to_add: int) -> None:
        agent_obj_lst = [agent_type.value(self) for _ in range(num_to_add)]
        with self._lock_particle_agents[agent_type]:
            self.particle_agents[agent_type] += agent_obj_lst
        for agent in agent_obj_lst:
            agent.start()
            agent.set_global_best(self.global_best)
    
    def remove_agents(self, agent_type: AgentType, num_to_remove: int) -> None:
        with self._lock_particle_agents[agent_type]:
            particle_agents_obj = self.particle_agents[agent_type]
            for _ in range(num_to_remove):
                worst_agent = max(particle_agents_obj)
                particle_agents_obj.remove(worst_agent)
                worst_agent.kill()
    
    def update_global_best(self, new_best_candidate: Solution, agent_class: type) -> None:
        agent_type = AgentType(agent_class)
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
    
    def announce_global_best_agent_type(self, agent_type: AgentType) -> None:
        with self._lock_particle_agents[agent_type]:
            for particle_agent in self.particle_agents[agent_type]:
                particle_agent.set_global_best_agent_type(self.agent_type_best[agent_type])
    
    def announce_global_best(self) -> None:
        for agent_type in [AgentType.PSO, AgentType.FOA]:
            with self._lock_particle_agents[agent_type]:
                for particle_agent in self.particle_agents[agent_type]:
                    particle_agent.set_global_best(self.global_best)

    def fetch_childs(self) -> None:
        for agent_type in [AgentType.GA, AgentType.ABC, AgentType.DE]:
            for agent in self.particle_agents[agent_type]:
                agent_childs = agent.get_childs()
                self.childs += agent_childs
    
    def get_parents(self, size: int = 2) -> tuple[np.ndarray]:
        with self._lock_probabilities:
            parents = np.random.choice(self.population, p=self.probabilities, replace=False, size=size)
        return tuple(parents)
    
    def calculate_possible_pairs(self) -> None:
        population_length = len(self.population)
        self.possible_pairs = (population_length * (population_length + 1)) / 2

    def calculate_probabilities(self) -> None:
        self.probabilities = [i / self.possible_pairs for i in range(1, len(self.population) + 1)]

    def select_population(self) -> None:
        self.fetch_childs()
        self.childs.sort()
        population_length = len(self.population)
        n_parents = int(gp.PARENTS_PERCENTAGE_GA * population_length) + 1
        n_childs = int(gp.CHILDREN_PERCENTAGE_GA * population_length) + 1
        n_random = population_length - n_parents - n_childs

        best_ones = self.population[:n_parents] + self.childs[:n_childs]
        others = self.population[n_parents:] + self.childs[n_childs:]

        self.set_population(sorted(best_ones + list(np.random.choice(others, size=n_random, replace=False))))
        self.childs = []

    def collect_results(self, agent_type_obj: 'ParticleAgent', best_value: float) -> None:
        with self._lock_performance:
            self.performance[AgentType(agent_type_obj.__class__)].append((agent_type_obj, best_value))
    
    def save_nums_and_clear_performance(self) -> None:
        print(f'''Number of agents:{''.join((f'\n\t{agent_type.name}: {self.num_agents[agent_type]}' for agent_type in AgentType))}\n''')
        print(f'''Inner iterations:{''.join((f'\n\t{agent_type.name}: {agent_type.value.iterations}' for agent_type in AgentType))}\n''')
        self.performance = {
            agent: [] for agent in AgentType
        }
        if not self.visualize_data:
            return
        self.nums_history.append(copy.copy(self.num_agents))
    
    def save_performance(self) -> None:
        self.avg_perfomance_history.append({
            agent_type: np.mean([
                perf for _, perf in self.performance[agent_type][-self.num_agents[agent_type]:]
            ])
            for agent_type in AgentType
        })
        self.best_perfomance_history.append(
            {agent_type: min(self.performance[agent_type][-self.num_agents[agent_type]:], key=lambda tup: tup[1])[1] for agent_type in AgentType}
        )
        self.iteration_times_history.append(copy.copy(self.iteration_times))
    
    def adapt(self) -> None:
        if self.adapt_num_agents:
            self.num_agents_adjuster.step()
        if self.adapt_parameters:
            for agent_type, performance_lst in self.performance.items():
                performance_lst_sorted = sorted(performance_lst, key=lambda tup: tup[1])
                length = len(performance_lst_sorted)
                
                exploration_count = max(1, int(gp.ADAPTATION_PARAMETERS_EXPLORATION_PERCENTAGE * length))
                exploitation_count = max(1, int(gp.ADAPTATION_PARAMETERS_EXPLOITATION_PERCENTAGE * length))
                
                worst_agents = heapq.nlargest(exploration_count, performance_lst, key=lambda tup: tup[1])
                for agent_obj, _ in worst_agents:
                    agent_obj.adapt(gp.ADAPTATION_PARAMETERS_EXPLORATION_RATE_INC, gp.ADAPTATION_PARAMETERS_EXPLOITATION_RATE_DEC)
                
                best_agents = heapq.nsmallest(exploitation_count, performance_lst, key=lambda tup: tup[1])
                for agent_obj, _ in best_agents:
                    agent_obj.adapt(gp.ADAPTATION_PARAMETERS_EXPLORATION_RATE_DEC, gp.ADAPTATION_PARAMETERS_EXPLOITATION_RATE_INC)
                
        if self.adapt_iterations:
            iteration_times = self.iteration_times_history[-1]
            mean_time = np.mean(list(iteration_times.values()))
            for agent_type, iteration_time in iteration_times.items():
                agent_type.value.iterations = int(agent_type.value.iterations * mean_time / iteration_time + 1)
    
    def start_agents(self) -> None:
        for agent_type in AgentType:
            for agent in self.particle_agents[agent_type]:
                agent.go()
    
    def run_agents(self) -> None:
        for agent_type in AgentType:
            for agent in self.particle_agents[agent_type]:
                agent.start()
    
    def agent_stopped(self, agent_class: type):
        stop_time = time.time()
        self.q_agents_stopped.put((AgentType(agent_class), stop_time))

    def wait_for_agents(self) -> None:
        start_time = time.perf_counter()
        for agent_type in AgentType:
            self.stopped_counter[agent_type] = 0
        
        for _ in range(sum(self.num_agents.values())):
            agent_type, end_time = self.q_agents_stopped.get()
            self.stopped_counter[agent_type] += 1
            if self.stopped_counter[agent_type] == self.num_agents[agent_type]:
                self.iteration_times[agent_type] = end_time - start_time
                print(f'{agent_type} STOPPED')

    def show_results(self, time_start: float, time_end: float) -> None:
        print(f'\nExecution time: {time_end - time_start:.2f} seconds\n')
        
        for agent_type in AgentType:
            if agent_type == AgentType.GA:
                best = min(self.population)
            else:
                best = self.agent_type_best[agent_type]
            print(f'{agent_type.name}: \n\tBest value: {best.value:.4f} \n\tBest position: {best.position}\n')
        
        if not self.visualize_data:
            return

        iterations = list(range(len(self.avg_perfomance_history)))

        def setup_x_axis():
            plt.xlim(min(iterations), max(iterations))
            plt.xticks(iterations)

        plt.figure(figsize=(12, 6))
        for agent_type in AgentType:
            plt.scatter(
                iterations,
                [hist[agent_type] for hist in self.avg_perfomance_history],
                label=agent_type.name,
                s=10
            )
        setup_x_axis()
        plt.title("Average Performance per Agent Type")
        plt.xlabel("Iteration")
        plt.ylabel("Average Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.figure(figsize=(12, 6))
        for agent_type in AgentType:
            plt.scatter(
                iterations,
                [hist[agent_type] for hist in self.best_perfomance_history],
                label=agent_type.name,
                s=10
            )
        setup_x_axis()
        plt.title("Best Performance per Agent Type")
        plt.xlabel("Iteration")
        plt.ylabel("Best Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.figure(figsize=(12, 6))
        for agent_type in AgentType:
            plt.scatter(
                iterations,
                [hist[agent_type] for hist in self.iteration_times_history],
                label=agent_type.name,
                s=10
            )
        setup_x_axis()
        plt.title("Iteration Times per Agent Type")
        plt.xlabel("Iteration")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.figure(figsize=(12, 6))
        for agent_type in AgentType:
            plt.scatter(
                iterations,
                [hist[agent_type] for hist in self.nums_history],
                label=agent_type.name,
                s=10
            )
        setup_x_axis()
        plt.title("Number of Agents per Agent Type Over Time")
        plt.xlabel("Iteration")
        plt.ylabel("Number of Agents")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.show()
