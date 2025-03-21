import threading
import numpy as np
import uuid
import time
import sys

dimensions = 10
iterations = 50
w, c1, c2 = 0.7, 1.5, 1.5
num_agents = 150

def rastrigin(x):
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

class Solution:
    def __init__(self, position=None, value=float('inf')):
        self.position = position if position is not None else np.random.uniform(-5.12, 5.12, dimensions)
        self.value = value

class SwarmSupervisor:
    def __init__(self):
        self.global_best = Solution()
        self.lock = threading.Lock()
    
    def update_global_best(self, candidate):
        with self.lock:
            if candidate.value < self.global_best.value:
                self.global_best = Solution(candidate.position.copy(), candidate.value)
                print(f"[Supervisor] New global best: {self.global_best.value:.4f}")

class ParticleAgent(threading.Thread):
    def __init__(self, supervisor):
        super().__init__()
        self.supervisor = supervisor
        self.position = np.random.uniform(-5.12, 5.12, dimensions)
        self.velocity = np.random.uniform(-1, 1, dimensions)
        self.best = Solution(self.position.copy(), rastrigin(self.position))
    
    def run(self):
        for iteration in range(iterations):
            self.update_particle()
            self.supervisor.update_global_best(self.best)
            if iteration % 10 == 0:
                print(f"[Agent {uuid.uuid4()}] Iteration {iteration}, best = {self.best.value:.4f}")
    
    def update_particle(self):
        g_best = self.supervisor.global_best.position
        if g_best is not None:
            self.velocity = (w * self.velocity +
                             c1 * np.random.rand(dimensions) * (self.best.position - self.position) +
                             c2 * np.random.rand(dimensions) * (g_best - self.position))
            self.position += self.velocity
        fitness = rastrigin(self.position)
        if fitness < self.best.value:
            self.best = Solution(self.position.copy(), fitness)

if __name__ == "__main__":
    print(f"Typ: {sys._is_gil_enabled()}\n")
    time_start = time.perf_counter()
    
    supervisor = SwarmSupervisor()
    agents = [ParticleAgent(supervisor) for _ in range(num_agents)]
    
    for agent in agents:
        agent.start()
    for agent in agents:
        agent.join()
    
    time_end = time.perf_counter()
    print(f"\nTime execution: {time_end - time_start}")
    print(f"Best global solution: {supervisor.global_best.value:.4f} at {supervisor.global_best.position}")
