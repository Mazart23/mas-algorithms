import numpy as np
import random
import time

# Funkcja celu (Rastrigin)
def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Parametry globalne
DIMENSIONS = 10
NUM_BEES = 30
ELITE_RATIO = 0.2
SELECTED_RATIO = 0.5
ITERATIONS = 100
DOMAIN = (-5.12, 5.12)
NEIGHBORHOOD_RADIUS = 0.5

# Agent pszczoły
class BeeAgent:
    def __init__(self, position):
        self.position = np.array(position)
        self.fitness = rastrigin(self.position)
        self.memory = self.position.copy()
        self.memory_fitness = self.fitness

    def explore(self, radius):
        candidate = self.position + np.random.uniform(-radius, radius, size=self.position.shape)
        candidate = np.clip(candidate, *DOMAIN)
        candidate_fitness = rastrigin(candidate)
        if candidate_fitness < self.memory_fitness:
            self.memory = candidate
            self.memory_fitness = candidate_fitness

    def update_position(self):
        self.position = self.memory
        self.fitness = self.memory_fitness

# Inicjalizacja populacji agentów
def initialize_agents():
    return [BeeAgent(np.random.uniform(*DOMAIN, DIMENSIONS)) for _ in range(NUM_BEES)]

# Główna pętla MAS
def bee_algorithm_mas():
    agents = initialize_agents()
    global_best = min(agents, key=lambda bee: bee.fitness)

    for it in range(ITERATIONS):
        # Sortuj agentów wg. przystosowania
        agents.sort(key=lambda bee: bee.fitness)
        elites = agents[:int(NUM_BEES * ELITE_RATIO)]
        selected = agents[:int(NUM_BEES * SELECTED_RATIO)]

        # Pszczoły eksplorują
        new_agents = []

        # Elitarne – intensywna eksploracja
        for elite in elites:
            for _ in range(NUM_BEES // len(elites)):
                new_bee = BeeAgent(elite.position)
                new_bee.explore(NEIGHBORHOOD_RADIUS)
                new_bee.update_position()
                new_agents.append(new_bee)

        # Dobre – umiarkowana eksploracja
        for sel in selected:
            for _ in range(NUM_BEES // len(selected)):
                new_bee = BeeAgent(sel.position)
                new_bee.explore(NEIGHBORHOOD_RADIUS * 1.5)
                new_bee.update_position()
                new_agents.append(new_bee)

        # Przypadkowe nowe pszczoły
        while len(new_agents) < NUM_BEES:
            rand_bee = BeeAgent(np.random.uniform(*DOMAIN, DIMENSIONS))
            new_agents.append(rand_bee)

        agents = new_agents

        # Aktualizacja najlepszego globalnego rozwiązania
        current_best = min(agents, key=lambda bee: bee.fitness)
        if current_best.fitness < global_best.fitness:
            global_best = current_best

        print(f"[Iteracja {it+1}] Najlepszy wynik: {global_best.fitness:.4f}")

    return global_best.position, global_best.fitness

# Uruchomienie
if __name__ == "__main__":
    start = time.time()
    best_sol, best_val = bee_algorithm_mas()
    end = time.time()

    print("\nNajlepsze rozwiązanie:", best_sol)
    print("Najlepsza wartość funkcji:", best_val)
    print(f"Czas wykonania: {end - start:.2f} s")
