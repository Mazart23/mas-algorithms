from __future__ import annotations
import sys
import time
import threading
import copy
from dataclasses import dataclass
import uuid

import numpy as np

# Parametry PSO
NUM_AGENTS = 100  # Liczba wątków (agentów)
PARTICLES_PER_AGENT = 5  # Cząstek na agenta
DIMENSIONS = 10  # Liczba wymiarów
ITERATIONS = 50  # Maksymalna liczba iteracji
W = 0.7  # Współczynnik inercji
C1 = 1.5  # weight for best particle position
C2 = 1.5  # weight for best global position

def rastrigin(x):
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

@dataclass
class Solution:
    solution_id = uuid.uuid4()
    position: list[float] | None = None
    value: float = float('inf')

class SwarmSupervisor:
    def __init__(self, agents: list[ParticleAgent] = []):
        self.global_best = Solution()
        self.particle_agents: list[ParticleAgent] = agents
        self._lock_global_best = threading.Lock()
        self._lock_particle_agents = threading.Lock()
    
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
            print(f"[NADZORCA] Nowe globalne najlepsze: {self.global_best.value:.4f} w {self.global_best.position}")
    
    def announce_global_best(self):
        with self._lock_particle_agents:
            for particle_agent in self.particle_agents:
                particle_agent.set_global_best(self.global_best)

class ParticleAgent(threading.Thread):
    def __init__(self, supervisor: SwarmSupervisor):
        self.agent_id = uuid.uuid4()
        super().__init__()
        self.supervisor: SwarmSupervisor = supervisor
        
        self.particles: np.ndarray[float] = np.random.uniform(-5.12, 5.12, (PARTICLES_PER_AGENT, DIMENSIONS))
        self.velocities: np.ndarray[float] = np.random.uniform(-1, 1, (PARTICLES_PER_AGENT, DIMENSIONS))
        self.particle_best: np.ndarray[Solution] = np.array([Solution(particle.copy(), rastrigin(particle)) for particle in self.particles])
        
        local_best_index: int = np.argmin([particle.value for particle in self.particle_best])
        self.local_best: Solution = Solution(self.particle_best[local_best_index].position.copy(), self.particle_best[local_best_index].value)
        
        self.global_best: Solution = Solution()
        
        self._lock_local_best: threading.Lock = threading.Lock()

    def __eq__(self, other):
        if isinstance(other, ParticleAgent):
            return self.agent_id == other.agent_id
        return False
    
    def __hash__(self):
        return hash(self.agent_id)
    
    def set_global_best(self, global_best: Solution):
        self.global_best = global_best
    
    def run(self):
        for iteration in range(ITERATIONS):
            threads = []
            
            self.prev_local_best = copy.deepcopy(self.local_best)
            
            for i in range(PARTICLES_PER_AGENT):
                thread = threading.Thread(target=self.update_particle, args=(i,))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()
            
            if not self.prev_local_best is self.local_best:
                self.supervisor.update_global_best(self.local_best)
            
            if iteration % 10 == 0:
                print(f"[AGENT {self.agent_id}] Iteracja {iteration}, lokalne najlepsze = {self.local_best.value:.4f}")

    def update_particle(self, i):
        global_best_position = self.global_best.position
        
        if global_best_position is not None:
            self.velocities[i] = (
                W * self.velocities[i] +
                C1 * np.random.rand(DIMENSIONS) * (self.particle_best[i].position - self.particles[i]) +
                C2 * np.random.rand(DIMENSIONS) * (global_best_position - self.particles[i])
            )
            self.particles[i] += self.velocities[i]

        fitness = rastrigin(self.particles[i])
        if fitness < self.particle_best[i].value:
            self.particle_best[i] = Solution(self.particles[i].copy(), fitness)
        
        if fitness < self.local_best.value:
            with self._lock_local_best:
                self.local_best = Solution(self.particles[i].copy(), fitness)


if __name__ == "__main__":
    print(f"Typ: {sys._is_gil_enabled()}\n")
    time_start = time.perf_counter()
    
    supervisor = SwarmSupervisor()
    agents = [ParticleAgent(supervisor) for _ in range(NUM_AGENTS)]
    supervisor.add_agents(agents)
    
    for agent in agents:
        agent.start()
    
    for agent in agents:
        agent.join()
    
    time_end = time.perf_counter()
    print(f"\nTime execution: {time_end - time_start}")
    print(f"Best global solution: \n\tposition: {supervisor.global_best.position} \n\tvalue: {supervisor.global_best.value}")