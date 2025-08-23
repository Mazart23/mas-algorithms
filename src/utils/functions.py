import numpy as np

from . import global_parameters as gp

def rastrigin(x):
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

def sphere(x):
    return sum(xi**2 for xi in x)

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(c * xi) for xi in x)
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e

def griewank(x):
    sum1 = sum(xi**2 / 4000 for xi in x)
    prod = np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])
    return sum1 - prod + 1

OBJECTIVE_FUNCTIONS_DICT = {
    'rastrigin': rastrigin,
    'sphere': sphere,
    'ackley': ackley,
    'griewank': griewank,
}

def discrete(x: np.ndarray) -> np.ndarray:
    if gp.IS_DISCRETE == 0:
        return x

    step = (gp.MAX_VALUE - gp.MIN_VALUE) / (gp.DISCRETE_POINTS - 1)

    idx = np.rint((x - gp.MIN_VALUE) / step).astype(int)
    idx = np.clip(idx, 0, gp.DISCRETE_POINTS - 1)

    return gp.MIN_VALUE + idx * step
