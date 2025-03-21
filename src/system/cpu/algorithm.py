import sys
import time
import threading
import multiprocessing

import numpy as np


# Parametry PSO
NUM_AGENTS = 1  # Liczba procesów (agentów)
PARTICLES_PER_AGENT = 50  # Cząstek na agenta
DIMENSIONS = 10  # Liczba wymiarów
ITERATIONS = 50  # Maksymalna liczba iteracji
W = 0.7  # Współczynnik inercji
C1 = 1.5  # Waga dla najlepszej pozycji cząstki
C2 = 1.5  # Waga dla najlepszej pozycji globalnej

# Funkcja Rastrigina
def rastrigin(x):
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

# Agent nadzorca – przechowuje najlepsze globalne rozwiązanie
def supervisor(best_solution_queue, global_best):
    while True:
        try:
            new_best = best_solution_queue.get(timeout=5)  # Pobiera najlepsze rozwiązanie
            if new_best is None:
                break  # Sygnał zakończenia
            if new_best["value"] < global_best["value"]:
                global_best["position"] = new_best["position"]
                global_best["value"] = new_best["value"]
                print(f"[NADZORCA] Nowe globalne najlepsze: {global_best['value']:.4f} w {global_best['position']}")
        except:
            break  # Jeśli kolejka pusta, kończymy

# Agent cząstki – aktualizuje cząstki w osobnym procesie
def particle_agent(agent_id, best_solution_queue, global_best):
    np.random.seed()  # Każdy proces musi mieć unikalny seed losowy
    
    # Inicjalizacja cząstek
    particles = np.random.uniform(-5.12, 5.12, (PARTICLES_PER_AGENT, DIMENSIONS))
    velocities = np.random.uniform(-1, 1, (PARTICLES_PER_AGENT, DIMENSIONS))
    pbest_positions = particles.copy()
    pbest_scores = np.array([rastrigin(p) for p in particles])

    local_best_index = np.argmin(pbest_scores)
    local_best_position = pbest_positions[local_best_index]
    local_best_value = pbest_scores[local_best_index]

    def update_particle(i):
        nonlocal local_best_position, local_best_value
        r1, r2 = np.random.rand(DIMENSIONS), np.random.rand(DIMENSIONS)
        
        velocities[i] = (
            W * velocities[i] +
            C1 * r1 * (pbest_positions[i] - particles[i]) +
            C2 * r2 * (global_best["position"] - particles[i] if global_best["position"] is not None else 0)
        )
        particles[i] += velocities[i]

        fitness = rastrigin(particles[i])
        if fitness < pbest_scores[i]:
            pbest_scores[i] = fitness
            pbest_positions[i] = particles[i]

        if fitness < local_best_value:
            local_best_value = fitness
            local_best_position = particles[i]

    for iteration in range(ITERATIONS):
        threads = []
        for i in range(PARTICLES_PER_AGENT):
            thread = threading.Thread(target=update_particle, args=(i,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        # Jeśli lokalne rozwiązanie jest lepsze od globalnego, zgłoś do nadzorcy
        if local_best_value < global_best["value"]:
            best_solution_queue.put({"position": local_best_position, "value": local_best_value})

        if iteration % 10 == 0:
            print(f"[AGENT {agent_id}] Iteracja {iteration}, lokalne najlepsze = {local_best_value:.4f}")

if __name__ == "__main__":
    print(f"Typ: {sys._is_gil_enabled()}\n")
    multiprocessing.freeze_support()
    time_start = time.perf_counter()
    
    # Tworzenie menedżera procesów
    with multiprocessing.Manager() as manager:
        # Tworzenie globalnych zasobów wewnątrz bloku main
        global_best = manager.dict({"position": None, "value": float("inf")})
        best_solution_queue = multiprocessing.Queue()

        # Tworzenie i uruchamianie agenta nadzorcy
        supervisor_process = multiprocessing.Process(target=supervisor, args=(best_solution_queue, global_best))
        supervisor_process.start()

        # Tworzenie i uruchamianie agentów cząstek
        agents = []
        for i in range(NUM_AGENTS):
            agent_process = multiprocessing.Process(target=particle_agent, args=(i, best_solution_queue, global_best))
            agent_process.start()
            agents.append(agent_process)

        # Czekanie na zakończenie agentów
        for agent in agents:
            agent.join()

        # Wysyłanie sygnału zakończenia dla nadzorcy
        best_solution_queue.put(None)
        supervisor_process.join()
        
        time_end = time.perf_counter()
        print(f"\nCzas: {time_end - time_start}")
        print(f"Najlepsze znalezione rozwiązanie globalne: {global_best['position']}")
        print(f"Minimalna wartość funkcji: {global_best['value']}")
