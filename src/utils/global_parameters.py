OBJECTIVE_FUNCTION = lambda: True

# general parameters
NUM_AGENTS = 100
ITERATIONS = 10

# adaptation parameters
ADAPTATION_NUM_AGENT_TAU = 0.1 # Softmax temperature
ADAPTATION_NUM_AGENT_ALPHA = 0.1 # Learning rate
ADAPTATION_NUM_AGENT_GAMMA = 0.1 # Discount factor
ADAPTATION_NUM_AGENT_MIN_CHANGE = 1 # Min number of agents to change in one cycle
ADAPTATION_NUM_AGENT_MAX_CHANGE = 10 # Max number of agents to change in one cycle
ADAPTATION_NUM_AGENT_SPEED = 5 # Scale speed of changes
ADAPTATION_NUM_AGENT_AVG_INFLUENCE = 0.2 # Percentage describing influence of mean solution for performance calculating
ADAPTATION_NUM_AGENT_BEST_INFLUENCE = 0.8 # Percentage describing influence of best solution for performance calculating
ADAPTATION_PARAMETERS_EXPLORATION_PERCENTAGE = 0.2 # Percentage of all agents from specific type to increase exploration related parameters
ADAPTATION_PARAMETERS_EXPLOATATION_PERCENTAGE = 0.2 # Percentage of all agents from specific type to increase exploatation related parameters
ADAPTATION_PARAMETERS_EXPLORATION_RATE = 1.1 # Weight of increasing exploration related parameters
ADAPTATION_PARAMETERS_EXPLOATATION_RATE = 1.1 # Weight of increasing exploration related parameters

# PSO parameters
PSO_ITERATIONS = 100
W = 1  # Inertion
C1 = 1.5  # weight for best local position
C2 = 1  # weight for best global position

# GA parameters
GA_ITERATIONS = 100
GA_POPULATION = 100
CROSSOVER_PROB_GA = 0.9
MUTATION_RATE_GA = 0.1
PARENTS_PERCENTAGE_GA = 0.2
CHILDREN_PERCENTAGE_GA = 0.5

# BEE parameters
BEE_ITERATIONS = 100
W_BEE = 1  # Weight of including random phi

# FOA parameters
FOA_ITERATIONS = 100
W_FOA = 1  # Weight of velocity of changes

# DE parameters
DE_ITERATIONS = 100
F_DE = 0.5  # Mutation factor
CR_DE = 0.7  # Crossover probability

# ACO parameters
ACO_ITERATIONS = 100
W_ACO = 1  # Inertion

# function parameters
DIMENSIONS = 10
MIN_VALUE = -5.12
MAX_VALUE = 5.12