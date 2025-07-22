from __future__ import annotations
import sys
import time

from .agents.supervisor import Supervisor
from ..utils.functions import OBJECTIVE_FUNCTIONS_DICT
from ..utils import global_parameters as gp


def run():
    function_name = 'rastrigin'
    adapt_num_agents = True
    adapt_parameters = True
    adapt_itearations = True
    visualize_data = True
    print(f'Is GIL enabled: {sys._is_gil_enabled()}')
    print(f'Objective function: {function_name}')
    print(f'Is adaptation of number of agents enabled: {adapt_num_agents}')
    print(f'Is adaptation of number of agents enabled: {adapt_num_agents}')
    print(f'Is adaptation of number of iterations enabled: {adapt_itearations}', end='')
    
    gp.OBJECTIVE_FUNCTION = OBJECTIVE_FUNCTIONS_DICT[function_name]
    
    time_start = time.perf_counter()

    supervisor = Supervisor(adapt_num_agents, adapt_parameters, adapt_itearations, visualize_data)
        
    supervisor.initialize_agents()
    supervisor.initialize_global_best()
    supervisor.initialize_population()

    supervisor.run_agents()
    
    for i in range(gp.ITERATIONS):
        print(f'\n\n##############\nIteration: {i}')
        supervisor.save_nums_and_clear_performance()

        supervisor.start_agents()
        supervisor.wait_for_agents()

        supervisor.select_population()
        supervisor.update_pheromones()
        supervisor.update_abc_border_performance()
        supervisor.save_performance()
        
        supervisor.adapt()
    
    time_end = time.perf_counter()

    supervisor.show_results(time_start, time_end)
