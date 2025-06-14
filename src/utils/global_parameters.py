import yaml


with open('src/config.yaml', 'r') as file:
    data = yaml.safe_load(file)

for key, value in data.items():
    globals()[key] = value


# init of objective function
OBJECTIVE_FUNCTION = lambda: True
