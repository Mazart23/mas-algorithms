import sys
import time
import threading
import numpy as np

# Parametry PSO
NUM_AGENTS = 100  # Liczba wątków (agentów)
PARTICLES_PER_AGENT = 5  # Cząstek na agenta
DIMENSIONS = 10  # Liczba wymiarów
ITERATIONS = 50  # Maksymalna liczba iteracji
W = 0.7  # Współczynnik inercji
C1 = 1.5  # Waga dla najlepszej pozycji cząstki
C2 = 1.5  # Waga dla najlepszej pozycji globalnej

# Funkcja Rastrigina
def rastrigin(x):
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

# Klasa nadzorcy
class Supervisor:
    def __init__(self):
        self.global_best = {"position": None, "value": float("inf")}
        self.lock = threading.Lock()

    def update_global_best(self, new_best):
        with self.lock:
            if new_best["value"] < self.global_best["value"]:
                self.global_best["position"] = new_best["position"].copy()
                self.global_best["value"] = new_best["value"]
                print(f"[NADZORCA] Nowe globalne najlepsze: {self.global_best['value']:.4f} w {self.global_best['position']}")

# Klasa agenta
class ParticleAgent(threading.Thread):
    def __init__(self, agent_id, supervisor):
        super().__init__()
        self.agent_id = agent_id
        self.supervisor = supervisor
        self.particles = np.random.uniform(-5.12, 5.12, (PARTICLES_PER_AGENT, DIMENSIONS))
        self.velocities = np.random.uniform(-1, 1, (PARTICLES_PER_AGENT, DIMENSIONS))
        self.pbest_positions = self.particles.copy()
        self.pbest_scores = np.array([rastrigin(p) for p in self.particles])
        
        self.local_best_index = np.argmin(self.pbest_scores)
        self.local_best_position = self.pbest_positions[self.local_best_index].copy()
        self.local_best_value = self.pbest_scores[self.local_best_index]

    def run(self):
        for iteration in range(ITERATIONS):
            threads = []
            for i in range(PARTICLES_PER_AGENT):
                thread = threading.Thread(target=self.update_particle, args=(i,))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()
            
            self.supervisor.update_global_best({"position": self.local_best_position, "value": self.local_best_value})
            
            if iteration % 10 == 0:
                print(f"[AGENT {self.agent_id}] Iteracja {iteration}, lokalne najlepsze = {self.local_best_value:.4f}")

    def update_particle(self, i):
        r1, r2 = np.random.rand(DIMENSIONS), np.random.rand(DIMENSIONS)
        global_best_position = self.supervisor.global_best["position"]
        
        if global_best_position is not None:
            self.velocities[i] = (
                W * self.velocities[i] +
                C1 * r1 * (self.pbest_positions[i] - self.particles[i]) +
                C2 * r2 * (global_best_position - self.particles[i])
            )
            self.particles[i] += self.velocities[i]

        fitness = rastrigin(self.particles[i])
        if fitness < self.pbest_scores[i]:
            self.pbest_scores[i] = fitness
            self.pbest_positions[i] = self.particles[i].copy()
        
        if fitness < self.local_best_value:
            self.local_best_value = fitness
            self.local_best_position = self.particles[i].copy()

# Uruchamianie systemu
if __name__ == "__main__":
    print(f"Typ: {sys._is_gil_enabled()}\n")
    time_start = time.perf_counter()
    
    supervisor = Supervisor()
    agents = [ParticleAgent(i, supervisor) for i in range(NUM_AGENTS)]
    
    for agent in agents:
        agent.start()
    
    for agent in agents:
        agent.join()
    
    time_end = time.perf_counter()
    print(f"\nCzas: {time_end - time_start}")
    print(f"Najlepsze znalezione rozwiązanie globalne: {supervisor.global_best['position']}")
    print(f"Minimalna wartość funkcji: {supervisor.global_best['value']}")