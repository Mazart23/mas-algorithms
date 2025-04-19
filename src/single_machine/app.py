from __future__ import annotations
import sys
import time

from .agents.supervisor import Supervisor
from ..utils.functions import OBJECTIVE_FUNCTIONS_DICT
from ..utils import global_parameters as gp
from ..utils.custom_objects.enums import AgentType


def run():
    function_name = 'ackley'
    adapt = False
    print(f'Is GIL enabled: {sys._is_gil_enabled()}')
    print(f'Objective function: {function_name}')
    print(f'Is adaptation enabled: {adapt}', end='')
    
    gp.OBJECTIVE_FUNCTION = OBJECTIVE_FUNCTIONS_DICT[function_name]
    
    time_start = time.perf_counter()

    supervisor = Supervisor(adapt=adapt,)
        
    supervisor.initialize_agents()
    supervisor.initialize_global_best()
    supervisor.initialize_population()

    supervisor.run_agents()
    
    for i in range(gp.ITERATIONS):
        print(f'\n\n##############\nIteration: {i}')
        print(f'''Number of agents:{''.join((f'\n\t{agent_type.name}: {supervisor.num_agents[agent_type]}' for agent_type in AgentType))}\n''')
        supervisor.start_agents()
        supervisor.wait_for_agents()

        supervisor.select_population()
        
        if supervisor.adapt:
            supervisor.adjust_agent_ratio()
    
    time_end = time.perf_counter()
    print(f'\nTime execution: {time_end - time_start}')
    for agent_type in AgentType:
        if agent_type == AgentType.GA:
            best = min(supervisor.population)
        else:
            best = supervisor.agent_type_best[agent_type]
        print(f'Best global solution {agent_type.name}: {best.value:.4f} at {best.position}')
