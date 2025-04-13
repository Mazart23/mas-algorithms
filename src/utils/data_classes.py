import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..single_machine.agents.pso import PSOAgent
from ..single_machine.agents.ga import GAAgent
from ..single_machine.agents.bee import BeeAgent
from ..single_machine.agents.foa import FOAAgent
from ..single_machine.agents.de import DEAgent
from ..single_machine.agents.aco import ACOAgent


@dataclass(order=True)
class Solution:
    position: np.ndarray | None = field(default=None, compare=False)
    value: float = float('inf')
    solution_id: uuid.UUID = field(default_factory=uuid.uuid4, compare=False)

class AgentType(Enum):
    PSO = PSOAgent
    GA = GAAgent
    BEE = BeeAgent
    FOA = FOAAgent
    DE = DEAgent
    ACO = ACOAgent
