import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import math
import numpy as np


def endurance(solution):
    res = math.exp(
        -2 * (solution[1] - math.sin(solution[1])) ** 2
    ) + math.sin(
        solution[2] * solution[4]
    ) + math.cos(solution[3] * solution[5])
    return -res

def opt_func(swarm):
    # results = []
    # for sol in swarm:
    #    endur = endurance(sol)
    #    results.append(endur)
    # return results
    n_particles = swarm.shape[0]
    j = [endurance(swarm[i]) for i in range(n_particles)]
    return np.array(j)

options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

x_max = np.ones(6)
x_min = np.zeros(6)

my_bounds = (x_min, x_max)

optimizer = ps.single.GlobalBestPSO(
    n_particles=300, dimensions=6, options=options, bounds=my_bounds)

optimizer.optimize(opt_func, iters=1000)
