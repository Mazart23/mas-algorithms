from __future__ import annotations
import sys
import time

from .agents.supervisor import Supervisor
from ..utils.functions import OBJECTIVE_FUNCTIONS_DICT
from ..utils import global_parameters as gp


def run():
    function_name = gp.OBJ_FUNCTION
    adapt_num_agents = gp.IS_ADAPT_NUM_AGENTS != 0
    adapt_parameters = gp.IS_ADAPT_PARAMETERS != 0
    adapt_iterations = gp.IS_ADAPT_ITERATIONS != 0
    visualize_data = gp.IS_VISUALIZE_DATA != 0
    print(f'Is GIL enabled: {sys._is_gil_enabled()}')
    print(f'Objective function: {function_name}')
    print(f'Is discrete points enabled: {gp.IS_DISCRETE != 0}')
    print(f'Is adaptation of number of agents enabled: {adapt_num_agents}')
    print(f'Is adaptation of agent paramaters enabled: {adapt_parameters}')
    print(f'Is adaptation of number of iterations enabled: {adapt_iterations}')
    print(f'Is visualize data enabled: {visualize_data}', end='')
    
    gp.OBJECTIVE_FUNCTION = OBJECTIVE_FUNCTIONS_DICT[function_name]
    
    time_start = time.perf_counter()

    supervisor = Supervisor(adapt_num_agents, adapt_parameters, adapt_iterations, visualize_data)
        
    supervisor.initialize_agents()
    supervisor.initialize_global_best()
    supervisor.initialize_population()

    supervisor.run_agents()
    
    for i in range(1, gp.ITERATIONS + 1):
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
