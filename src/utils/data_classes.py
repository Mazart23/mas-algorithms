import uuid
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

@dataclass(order=True)
class Solution:
    position: np.ndarray | None = field(default=None, compare=False)
    value: float = float('inf')
    solution_id: uuid.UUID = field(default_factory=uuid.uuid4, compare=False)

class AgentType(Enum):
    PSO = 0
    GA = 1
    BEE = 2
    FOA = 3
    DE = 4
    ACO = 5
