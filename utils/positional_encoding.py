from math import cos, sin

import yaml

with open('./config.yaml', "r") as f:
    config = yaml.safe_load(f)
d = config["model"]["d_model"]

def positional_encoding(position):
    positions = []
    for i in range(d):
        to_be_sinusoided = position / pow(10_000, 2 * (i // 2) / d)
        if i % 2 == 0:
            positions.append(sin(to_be_sinusoided))
        else:
            positions.append(cos(to_be_sinusoided))
    return positions
