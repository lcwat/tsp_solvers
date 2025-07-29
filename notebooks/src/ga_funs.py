import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# define the initial population


def initial_population(locations_df=pd.DataFrame, start_location=int, n_population=int):
    # create a list of the remaining locations in the arrangement
    remaining_locations = [
        *locations_df[~locations_df.location.isin([start_location])].location
    ]

    # start by creating population of permutations of the locations
    population_perms = [
        [start_location] + [*np.random.permutation(remaining_locations)] for i in range(0, n_population)
    ]

    return population_perms

# create a distance matrix between all positions that can be accessed


# def ga_euclid_distance_matrix(df=pd.DataFrame):
#     # create empty matrix of necessary size
#     n = len(df)
#     dist_matrix = np.zeros((n, n), dtype=float)

#     # prespecify the location coord list
#     location_i = []
#     location_j = []

#     # fill matrix by calculating dist between appropriate locations
#     for i in range(n):
#         for j in range(n):
#             location_i = [df.x.iloc[i], df.y.iloc[i]]
#             location_j = [df.x.iloc[j], df.y.iloc[j]]

#             # calculate euclid distance
#             dist = np.sqrt(
#                 np.pow(location_i[0] - location_j[0], 2) +
#                 np.pow(location_i[1] - location_j[1], 2)
#             )

#             dist_matrix[i][j] = dist

#     return dist_matrix

# find the value of the individual (route)


def total_distance_individual(
    individual=list, dist_matrix=np.matrix, path_len=int
):
    dists = [
        dist_matrix[individual[i]-1][individual[0]-1] if i == path_len-1 else dist_matrix[individual[i]-1][individual[i+1]-1] for i in range(0, path_len)
    ]

    return sum(dists)

# define the fitness probabilty


def fitness_probability(population, dist_matrix=np.matrix, path_len=int):
    total_distance_all_individuals = [
        total_distance_individual(path, dist_matrix=dist_matrix, path_len=path_len) for path in population
    ]

    # calculate the fitness probabilty of each individual
    # given more fitness for shorter routes, standardized into prob
    pop_fitness = max(total_distance_all_individuals) - \
        total_distance_all_individuals
    pop_fitness_probs = pop_fitness / sum(pop_fitness)

    return pop_fitness_probs

# roulette wheel selection of individuals based on probability


def roulette_wheel(population, fitness_probs):
    # output an individual by grabbing random individual from population
    bool_prob_array = fitness_probs.cumsum() < np.random.uniform(
        0, 1, 1)
    selected_individual_index = len(
        bool_prob_array[bool_prob_array == True]) - 1

    return population[selected_individual_index]

# crossover genetic material of individuals (subroutes)


def crossover(parent_1, parent_2, locations_df):
    # output two new offspring from two parents
    # determine cutpoint from sequence length
    n_locations_cut = len(locations_df) - 1
    cut = round(random.uniform(1, n_locations_cut))  # random place in mid

    offspring_1 = []
    offspring_2 = []

    # create offspring by cutting parent at selected index, fill in
    # the remaining index with other individual excluding duplicates
    offspring_1 = parent_1[0:cut]
    offspring_1 += [location for location in parent_2 if location not in offspring_1]

    offspring_2 = parent_2[0:cut]
    offspring_2 += [location for location in parent_1 if location not in offspring_2]

    return offspring_1, offspring_2

# mutation function


def mutation(offspring=list, path_len=int):
    n_location_cut = path_len - 1
    # grab two random locations from route to swap positions
    index_1 = round(random.uniform(1, n_location_cut))
    index_2 = round(random.uniform(1, n_location_cut))

    temp = offspring[index_1]
    offspring[index_1] = offspring[index_2]  # swap
    offspring[index_2] = temp  # replace

    return offspring

# the ga


def run_ga(
    locations_df=pd.DataFrame, start_location=int, path_len=int, n_population=int,
    n_generations=int, crossover_rate=float,
    mutation_rate=float, dist_matrix=np.matrix
):
    # init population and find fitness of individuals
    pop = initial_population(locations_df, start_location, n_population)
    fitness_probs = fitness_probability(pop, dist_matrix, path_len=path_len)

    parents_list = [
        roulette_wheel(pop, fitness_probs) for i in range(0, int(crossover_rate * n_population))
    ]

    offspring_list = []

    # crossover the selection of parents to create offspring, iterate
    # two at a time
    for i in range(0, len(parents_list), 2):
        # recombination
        offspring_1, offspring_2 = crossover(
            parents_list[i], parents_list[i+1], locations_df)

        # determine whether offspring should mutate
        mutate_threshold = random.random()
        if (mutate_threshold > (1-mutation_rate)):
            offspring_1 = mutation(offspring_1, path_len=path_len)

        mutate_threshold = random.random()
        if (mutate_threshold > (1-mutation_rate)):
            offspring_2 = mutation(offspring_2, path_len=path_len)

        offspring_list.append(offspring_1)
        offspring_list.append(offspring_2)

    # mix in the new offspring with parents to create new population
    mixed_offspring = parents_list + offspring_list

    # find the best solutions to create better population
    fitness_probs = fitness_probability(
        mixed_offspring, dist_matrix, path_len=path_len)
    sorted_fitness_probs = np.argsort(fitness_probs)[::-1]  # descend
    best_fitness_indices = sorted_fitness_probs[0:n_population]  # selection

    # grab those offspring
    next_generation = [
        mixed_offspring[i] for i in best_fitness_indices
    ]

    # begin generation loop
    for i in range(0, n_generations):
        # rerun the selection, recombination, and mutation processes
        fitness_probs = fitness_probability(
            next_generation, dist_matrix, path_len=path_len)

        parents_list = [
            roulette_wheel(next_generation, fitness_probs) for i in range(0, int(crossover_rate * n_population))
        ]

        offspring_list = []
        for i in range(0, len(parents_list), 2):
            offspring_1, offspring_2 = crossover(
                parents_list[i], parents_list[i+1], locations_df)
            mutate_threshold = random.random()
            if (mutate_threshold > (1-mutation_rate)):
                offspring_1 = mutation(offspring_1, path_len=path_len)

            mutate_threshold = random.random()
            if (mutate_threshold > (1-mutation_rate)):
                offspring_2 = mutation(offspring_2, path_len=path_len)

            offspring_list.append(offspring_1)
            offspring_list.append(offspring_2)

        mixed_offspring = parents_list + offspring_list
        fitness_probs = fitness_probability(
            mixed_offspring, dist_matrix, path_len=path_len)
        sorted_fitness_probs = np.argsort(fitness_probs)[::-1]  # descend
        # subset selection
        best_fitness_indices = sorted_fitness_probs[0:int(.8*n_population)]

        # grab those offspring
        best_mixed_offspring = [
            mixed_offspring[i] for i in best_fitness_indices
        ]

        # remain some old individuals (20%)
        current_gen_indices = [
            random.randint(0, (n_population - 1)) for j in range(int(.2*n_population))
        ]

        carryovers = [
            pop[i] for i in current_gen_indices
        ]

        next_generation = best_mixed_offspring + carryovers

        # shuffle them up
        random.shuffle(next_generation)

    return next_generation


def run_all_ga(
    locations_df=pd.DataFrame, path_len=int, dist_matrix=np.matrix, n_population=int, n_generations=int, crossover_rate=float, mutation_rate=float
):
    # run all variations given different starting points
    start_locations = []
    paths = []
    total_distances = []

    # # init dist matrix
    # dist_matrix = euclid_distance_matrix(locations_df)

    for location in locations_df.location:
        final_pop = run_ga(
            locations_df=locations_df, start_location=location, path_len=path_len, n_population=n_population,
            n_generations=n_generations, crossover_rate=crossover_rate,
            mutation_rate=mutation_rate, dist_matrix=dist_matrix
        )

        total_distance_all_individuals = [
            total_distance_individual(path, dist_matrix=dist_matrix, path_len=path_len) for path in final_pop
        ]

        index_minimum_dist = np.argmin(total_distance_all_individuals)
        minimum_dist = min(total_distance_all_individuals)

        path = final_pop[index_minimum_dist]

        start_locations.append(location)
        paths.append(path)
        total_distances.append(minimum_dist)

    ga_performance_dict = {
        'start_locations': start_locations,
        'paths': paths,
        'distances': total_distances,
        'population': n_population,
        'generations': n_generations,
        'crossover_rate': crossover_rate,
        'mutation_rate': mutation_rate
    }

    return ga_performance_dict


def plot_ga_path(starting_location=int, ga_dict=dict, locations_df=pd.DataFrame):
    focal_path = ga_dict['paths'][starting_location-1]
    focal_distance = ga_dict['distances'][starting_location-1]

    x_shortest = []
    y_shortest = []
    for location in focal_path:
        x_value = locations_df['x'].iloc[location - 1]
        y_value = locations_df['y'].iloc[location - 1]
        x_shortest.append(x_value)
        y_shortest.append(y_value)

    x_shortest.append(x_shortest[0])
    y_shortest.append(y_shortest[0])

    fig, ax = plt.subplots()

    for i in range(len(locations_df.x)):
        for j in range(i+1, len(locations_df.x)):
            ax.plot([locations_df.x[i], locations_df.x[j]], [
                    locations_df.y[i], locations_df.y[j]], 'k-', alpha=.09, linewidth=1)

    ax.plot(x_shortest, y_shortest, '--go', linewidth=2.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([-80, 80])
    ax.set_ylim([-80, 80])

    plt.title(label="TSP best route using a GA", fontsize=14, color='k')

    str_params = '\n'+str(ga_dict['generations'])+' Generations\n'+str(ga_dict['population'])+' Population size\n' + \
        str(ga_dict['crossover_rate'])+' Crossover rate\n' + \
        str(ga_dict['mutation_rate'])+' Mutation rate'
    plt.suptitle("Total Distance Traveled: " +
                 str(round(focal_distance, 1))+str_params, fontsize=9, y=1.047)

    for i, loc in enumerate(focal_path):
        ax.annotate(
            str(loc), (x_shortest[i] + 1.5, y_shortest[i] + 1.25), fontsize=10)

    fig.set_size_inches(6.5, 6)
    plt.show()
