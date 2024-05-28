import pygad
import numpy
import time

items = [
    (3, 100), (5, 300), (9, 400), (7, 200), (4, 350), 
    (5, 500), (6, 150), (7, 180), (3, 100), (2, 300)
]
max_weight = 25

def fitness_func(ga_instance, solution, solution_idx):
    weight = numpy.sum(solution * numpy.array([item[0] for item in items]))
    value = numpy.sum(solution * numpy.array([item[1] for item in items]))
    if weight > max_weight:
        return 0
    return value

gene_space = [0, 1]
sol_per_pop = 10
num_genes = len(items)
num_parents_mating = 5
num_generations = 30
keep_parents = 2
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10

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
        mutation_percent_genes=mutation_percent_genes,
        stop_criteria=["reach_1630"]
    )
    start = time.time()
    ga_instance.run()
    end = time.time()
    return ga_instance, end - start

best_solution_value = 1630

successful_runs = 0
total_time = 0
num_runs = 10

for _ in range(num_runs):
    ga_instance, run_time = run_ga()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    if solution_fitness == best_solution_value:
        successful_runs += 1
        total_time += run_time

print(f"Liczba udanych prób: {successful_runs}/{num_runs}")
if successful_runs > 0:
    average_time = total_time / successful_runs
    print(f"Średni czas działania algorytmu: {average_time:.2f} sekund")
else:
    print("Brak udanych prób.")

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Najlepsze rozwiązanie (chromosom):", solution)
print("Wartość najlepszego rozwiązania:", solution_fitness)
print("Waga najlepszego rozwiązania:", numpy.sum(solution * numpy.array([item[0] for item in items])))
ga_instance.plot_fitness()
