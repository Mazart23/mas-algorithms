from enum import Enum

from ...single_machine.agents.pso import PSOAgent
from ...single_machine.agents.ga import GAAgent
from ...single_machine.agents.abc import ABCAgent
from ...single_machine.agents.foa import FOAAgent
from ...single_machine.agents.de import DEAgent
from ...single_machine.agents.aco import ACOAgent

class AgentType(Enum):
    PSO = PSOAgent
    GA = GAAgent
    ABC = ABCAgent
    FOA = FOAAgent
    DE = DEAgent
    ACO = ACOAgent
