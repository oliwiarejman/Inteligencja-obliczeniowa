import pygad
import numpy as np
import random
import time
import math

labirynt = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

gene_space = [0, 1, 2, 3]

chromosome_length = 30

sol_per_pop = 100 
num_parents_mating = 50 
num_generations = 100 
keep_parents = 10 
mutation_percent_genes = 5

def is_move_valid(lab, position):
    x, y = position
    if lab[x, y] == 1:
        return False
    return True

def fitness_func(ga_instance, solution, solution_idx):
    x, y = 1, 1
    for move in solution:
        if move == 0:
            y -= 1 
        elif move == 1:
            y += 1
        elif move == 2:
            x -= 1
        elif move == 3:
            x += 1 
        if not is_move_valid(labirynt, (x, y)):
            return -999999 
        if (x, y) == (10, 10):
            return 999999 - len(solution) 
    distance_to_exit = abs(10 - x) + abs(10 - y)
    return -distance_to_exit 


ga_instance = pygad.GA(
    gene_space=gene_space,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=chromosome_length,
    parent_selection_type="sss",
    keep_parents=keep_parents,
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=mutation_percent_genes
)


start = time.time()
ga_instance.run()
end = time.time()


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Czas działania algorytmu: {czas}".format(czas=end-start))


x, y = 1, 1 
path = [(x, y)]
for move in solution:
    if move == 0:
        y -= 1
    elif move == 1:
        y += 1
    elif move == 2:
        x -= 1
    elif move == 3:
        x += 1
    path.append((x, y))
    if (x, y) == (10, 10):
        break

print("Found path:")
print(path)

ga_instance.plot_fitness()
