OBJECTIVE_FUNCTION = lambda: True

# general parameters
NUM_AGENTS = 100
ITERATIONS = 10
ADAPTATION_SPEED = 0.1
ADAPTATION_CHANGE_TOLERATION = 0.05

# PSO parameters
PSO_ITERATIONS = 100
W = 1  # Inertion
C1 = 1.5  # weight for best local position
C2 = 1  # weight for best global position

# GA parameters
GA_ITERATIONS = 100
GA_POPULATION = 100
CROSSOVER_PROB = 0.9
MUTATION_RATE = 0.1
PARENTS_PERCENTAGE = 0.2
CHILDREN_PERCENTAGE = 0.5

# BEE parameters
BEE_ITERATIONS = 100
# W = 1  # Inertion
# C1 = 1.5  # weight for best local position
# C2 = 1  # weight for best global position

# FOA parameters
FOA_ITERATIONS = 100
# W = 1  # Inertion
# C1 = 1.5  # weight for best local position
# C2 = 1  # weight for best global position

# DE parameters
DE_ITERATIONS = 100
DE_F = 0.5  # Mutation factor
DE_CR = 0.7  # Crossover probability

# ACO parameters
ACO_ITERATIONS = 100
# W = 1  # Inertion
# C1 = 1.5  # weight for best local position
# C2 = 1  # weight for best global position

# function parameters
DIMENSIONS = 10
MIN_VALUE = -5.12
MAX_VALUE = 5.12