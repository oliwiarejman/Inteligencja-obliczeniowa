import pygad
import numpy
import math
import time

def endurance(x, y, z, u, v, w):
    return math.exp(-2*(y-math.sin(x))**2) + math.sin(z*u) + math.cos(v*w)


def fitness_func(ga_instance, solution, solution_idx):
    return endurance(solution[0], solution[1], solution[2], solution[3], solution[4], solution[5])


gene_space = {'low': 0.0, 'high': 1.0}

sol_per_pop = 20
num_genes = 6
num_parents_mating = 10
num_generations = 50
keep_parents = 2
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 20 

def run_ga():
    ga_instance = pygad.GA(
        gene_space=gene_space,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes
    )
    start = time.time()
    ga_instance.run()
    end = time.time()
    return ga_instance, end - start

ga_instance, run_time = run_ga()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution:", solution)
print("Fitness value of the best solution:", solution_fitness)
print("Czas działania algorytmu:", run_time, "sekund")
ga_instance.plot_fitness()

num_runs = 10
best_solutions = []
best_fitness_values = []
run_times = []

for _ in range(num_runs):
    ga_instance, run_time = run_ga()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    best_solutions.append(solution)
    best_fitness_values.append(solution_fitness)
    run_times.append(run_time)

for i in range(num_runs):
    print(f"Run {i+1}:")
    print("Best solution:", best_solutions[i])
    print("Best fitness value:", best_fitness_values[i])
    print("Run time:", run_times[i], "sekund")
    print()

average_fitness = sum(best_fitness_values) / num_runs
average_time = sum(run_times) / num_runs

print("Średnia wartość fitness:", average_fitness)
print("Średni czas działania algorytmu:", average_time, "sekund")
