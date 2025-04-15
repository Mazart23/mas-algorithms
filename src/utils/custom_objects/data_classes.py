import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass(order=True)
class Solution:
    position: np.ndarray | None = field(default=None, compare=False)
    value: float = float('inf')
    solution_id: uuid.UUID = field(default_factory=uuid.uuid4, compare=False)
