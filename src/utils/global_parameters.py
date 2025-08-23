import yaml


with open('src/config.yaml', 'r') as file:
    data = yaml.safe_load(file)

for key, value in data.items():
    globals()[key] = value

globals()['DISCRETE_POINTS'] = max(2, data.get('DISCRETE_POINTS', 100))

# init of objective function
OBJECTIVE_FUNCTION = lambda: True
