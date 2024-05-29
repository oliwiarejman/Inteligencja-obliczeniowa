import pygad
import numpy
import math

labirynth = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
             [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
             [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
             [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
             [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
             [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
             [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
             [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
             [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 3, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
printlab = ""
for arr in labirynth:
    for el in arr:
        printlab += "  " if el == 1 else "# "
    printlab += "\n"
print(printlab)

gene_space = [1, 2, 3, 4]


def move_through_labirynth(solution):
    picked_moves = set((1, 1))
    x = 1
    y = 1
    counter = 0
    punish_points = 0
    found = False
    for move in solution:
        counter += 1
        prev_x = x
        prev_y = y
        if move == 1:
            x -= 1
        elif move == 2:
            x += 1
        elif move == 3:
            y -= 1
        elif move == 4:
            y += 1
        picked_moves.add((x, y))
        if (x, y) in picked_moves:
            punish_points += 2
        if (0 <= x < len(labirynth)) and (0 <= y < len(labirynth)):
            point = labirynth[x][y]
            if point == 0:
                punish_points += 2
                x = prev_x
                y = prev_y
            elif point == 3:
                punish_points -= 10
                break
    distance_from_exit = math.sqrt(((x - 10) ** 2) + ((y - 10) ** 2))
    punish_points += distance_from_exit ** 2
    return punish_points
# dopisac odleglosc od wyjscia
# jak wejdzie na sciane to dostaje kare i cofa sie na pole
# rozwiazanie zasymulowac i wyciac nielegalne ruchy  
# 

def fitness_func(model, solution, solution_idx):
    punish = move_through_labirynth(solution)
    fitness = 1 - punish/100
    return fitness


fitness_function = fitness_func

# ile chromsomów w populacji
# ile genow ma chromosom
sol_per_pop = 200
num_genes = 30

# ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
# ile pokolen
# ilu rodzicow zachowac (kilka procent)
num_parents_mating = 30
num_generations = 100
keep_parents = 6

# jaki typ selekcji rodzicow?
# sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

# w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

# mutacja ma dzialac na ilu procent genow?
# trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(
    solution_fitness=solution_fitness))

ga_instance.plot_fitness()

solved_labirynth = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
             [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
             [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
             [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
             [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
             [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
             [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
             [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
             [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 3, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
x = 1
y = 1
counter = 0
for move in solution:
    moves = set((1,1))
    counter += 1
    x_prev = x
    y_prev = y
    if move == 1:
        x -= 1
    elif move == 2:
        x += 1
    elif move == 3:
        y -= 1
    elif move == 4:
        y += 1
    if (x, y) in moves:
        print("move taken")
        counter -= 1
    moves.add((x,y))
    if solved_labirynth[x][y] == 3:
        solved_labirynth[x][y] = 4
        break
    elif solved_labirynth[x][y] == 0:
        x = x_prev
        y = y_prev
    else:
        solved_labirynth[x][y] = 2

printlab = ""
for arr in solved_labirynth:
    printlab += "\n"
    for el in arr:
        if el==0:
            printlab += "# "
        elif el==1:
            printlab += "  "
        elif el == 2:
            printlab += "= "
        elif el == 4:
            printlab += "X "
        elif el == 3:
            printlab += "O "
print(printlab)
print(counter)
