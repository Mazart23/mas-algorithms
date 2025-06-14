from enum import Enum

from ...models.agents.pso import PSOAgent
from ...models.agents.ga import GAAgent
from ...models.agents.abc import ABCAgent
from ...models.agents.foa import FOAAgent
from ...models.agents.de import DEAgent
from ...models.agents.aco import ACOAgent

class AgentType(Enum):
    PSO = PSOAgent
    GA = GAAgent
    ABC = ABCAgent
    FOA = FOAAgent
    DE = DEAgent
    ACO = ACOAgent
