import numpy as np
import random
import pandas as pd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import ga_funs
import nn_funs

# function to generate random points within polygon


def random_points_in_polygon(polygon=Polygon, num_points=int):
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    while len(points) < num_points:
        point = Point(round(np.random.uniform(min_x, max_x), 1),
                      round(np.random.uniform(min_y, max_y), 1))
        if polygon.contains(point):
            points.append(point)

    return points

# create a population of dfs


def initial_location_population(
    n_population=int, polygon=Polygon,
    n_locations=int
):
    # list to populate with dfs
    df_list = []

    # list to populate location column
    location = [*range(1, n_locations+1)]

    # create random population of locations
    for i in range(0, n_population):
        points = random_points_in_polygon(
            polygon=polygon, num_points=n_locations)

        xs = []
        ys = []

        for point in points:
            xs.append(point.coords[0][0])
            ys.append(point.coords[0][1])

        df = pd.DataFrame({
            'location': location,
            'x': xs,
            'y': ys
        })

        df_list.append(df)

    return df_list


def find_deviation_score(
    location_df=pd.DataFrame, n_population=100, n_generations=30, crossover_rate=.9, mutation_rate=.5
):
    # run each of the ga and nn
    # could run one ga since they often converge on the same answer
    # from all starting locations
    # get the distances
    dist_matrix = nn_funs.euclid_distance_matrix(location_df)
    path_len = len(location_df)

    # get the optimal and nn route distances from each location in arrangement
    ga_dist = ga_funs.run_all_ga(
        locations_df=location_df, dist_matrix=dist_matrix, path_len=path_len,
        n_population=n_population, n_generations=n_generations,
        crossover_rate=crossover_rate, mutation_rate=mutation_rate
    )['distances']
    nn_dist = nn_funs.run_all_nn(
        location_df, dist_matrix=dist_matrix, path_len=path_len
    )['distances']

    # put them together into a list
    deviations = [
        np.pow(nn_dist[i]-ga_dist[i], 2) for i in range(0, path_len)
    ]

    # return the sum
    return sum(deviations)

# create fitness probabilities for each df


def fitness_probabilities(
    population=list, max_deviation=bool
):
    # deviation list
    all_deviations = [
        find_deviation_score(df) for df in population
    ]

    if max_deviation == True:
        pop_fitness_probabilities = all_deviations / sum(all_deviations)
    else:
        pop_fitness = max(all_deviations)-all_deviations
        pop_fitness_probabilities = pop_fitness / sum(pop_fitness)

    return pop_fitness_probabilities

# fitness based selection


def roulette(population=pd.DataFrame, fitness_probs=list):
    # output an individual by grabbing random individual from population
    bool_prob_array = fitness_probs.cumsum() < np.random.uniform(
        0, 1, 1)
    selected_individual_index = len(bool_prob_array[bool_prob_array == True])

    return population[selected_individual_index]

# mix together df material into offspring dfs


def reproduce(parent_1=pd.DataFrame, parent_2=pd.DataFrame):
    # determine split location somewhere in middle of df
    cut_location = round(random.uniform(1, len(parent_1)))

    len_index = len(parent_1)+1

    offspring_1 = pd.DataFrame({
        'location': [*range(1, len_index)],
        'x': [*parent_1.x[0:cut_location]]+[*parent_2.x[cut_location:len_index]],
        'y': [*parent_1.y[0:cut_location]]+[*parent_2.y[cut_location:len_index]]
    })
    offspring_2 = pd.DataFrame({
        'location': [*range(1, len_index)],
        'x': [*parent_2.x[0:cut_location]]+[*parent_1.x[cut_location:len_index]],
        'y': [*parent_2.y[0:cut_location]]+[*parent_1.y[cut_location:len_index]]
    })

    return offspring_1, offspring_2

# swap the coord of one index with specified range


def point_mutation(offspring=pd.DataFrame, polygon=Polygon):
    cut_index = len(offspring)-1
    point_index = round(random.uniform(0, cut_index))

    # draw another sample from the polygon
    point = random_points_in_polygon(polygon=polygon, num_points=1)[0].coords

    # replace x/y of that location
    offspring.iloc[point_index, 1] = point[0][0]
    offspring.iloc[point_index, 2] = point[0][1]

    return offspring

# run the ga for the arrangement creation


def arrangement_ga(
    n_population=int, polygon=Polygon, n_locations=int, n_generations=int, crossover_rate=float, mutation_rate=float, max_deviation=True
):
    # init population of arrangements and find their fitness
    init_pop = initial_location_population(
        n_population=n_population, polygon=polygon,
        n_locations=n_locations
    )
    fitness_probs = fitness_probabilities(
        population=init_pop, max_deviation=max_deviation)

    # add dfs to become parents to reproduce
    parents = [
        roulette(population=init_pop, fitness_probs=fitness_probs) for i in range(0, int(crossover_rate*n_population))
    ]

    # generate new offspring blending parent rows
    offspring = []
    for i in range(0, len(parents)-1, 2):
        offspring_1, offspring_2 = reproduce(
            parent_1=parents[i], parent_2=parents[i+1]
        )

        # should they mutate as well?
        mutate_threshold = random.random()
        if (mutate_threshold > (1-mutation_rate)):
            offspring_1 = point_mutation(
                offspring=offspring_1, polygon=polygon
            )

        # should they mutate as well?
        mutate_threshold = random.random()
        if (mutate_threshold > (1-mutation_rate)):
            offspring_2 = point_mutation(
                offspring=offspring_2, polygon=polygon
            )

        offspring.append(offspring_1)
        offspring.append(offspring_2)

    # mix together the individuals
    mixed_pop = parents + offspring

    # reassess fitness
    fitness_probs = fitness_probabilities(
        population=mixed_pop, max_deviation=max_deviation)
    sorted_probs = np.argsort(fitness_probs)[::-1]  # descending order
    best_fitness_indices = sorted_probs[0:n_population]  # select the best

    # grab those dfs
    next_generation = [
        mixed_pop[i] for i in best_fitness_indices
    ]

    # begin generation loop
    for i in range(0, n_generations):
        # assess fitness of this generatino
        fitness_probs = fitness_probabilities(
            population=next_generation, max_deviation=max_deviation)

        # add dfs to become parents to reproduce
        parents = [
            roulette(population=next_generation, fitness_probs=fitness_probs) for i in range(0, int(crossover_rate*n_population))
        ]

        # generate new offspring blending parent rows
        offspring = []
        for i in range(0, len(parents)-1, 2):
            offspring_1, offspring_2 = reproduce(
                parent_1=parents[i], parent_2=parents[i+1]
            )

            # should they mutate as well?
            mutate_threshold = random.random()
            if (mutate_threshold > (1-mutation_rate)):
                offspring_1 = point_mutation(
                    offspring=offspring_1, polygon=polygon
                )

            # should they mutate as well?
            mutate_threshold = random.random()
            if (mutate_threshold > (1-mutation_rate)):
                offspring_2 = point_mutation(
                    offspring=offspring_2, polygon=polygon
                )

            offspring.append(offspring_1)
            offspring.append(offspring_2)

        # mix together the individuals
        mixed_pop = parents + offspring

        # reassess fitness
        fitness_probs = fitness_probabilities(
            population=mixed_pop, max_deviation=max_deviation)
        sorted_probs = np.argsort(fitness_probs)[::-1]  # descending order
        # select 80% of best
        best_fitness_indices = sorted_probs[0:int(.8*n_population)]

        # grab those dfs
        best_mixed_offspring = [
            mixed_pop[i] for i in best_fitness_indices
        ]

        # create next generation by remaining 20% of the current generation
        current_gen_indices = [
            random.randint(0, n_population-1) for j in range(int(.2*n_population))
        ]
        carryovers = [
            init_pop[i] for i in current_gen_indices
        ]

        next_generation = best_mixed_offspring + carryovers

        # shuffle
        random.shuffle(next_generation)

    # return the final generation
    return next_generation


def arrangement_pop_summary(population=list, max_deviation=True):
    # deviation list
    final_pop_deviations = [
        find_deviation_score(df) for df in population
    ]

    if max_deviation == True:
        index_best_deviation = np.argmax(final_pop_deviations)
        best_deviation_score = max(final_pop_deviations)
    else:
        index_best_deviation = np.argmin(final_pop_deviations)
        best_deviation_score = min(final_pop_deviations)

    best_deviation_df = population[index_best_deviation]

    # summary stats before and after ga
    random_pop = initial_location_population(
        n_population=50, polygon=poly, n_locations=8)
    random_pop_deviations = [
        find_deviation_score(df) for df in random_pop
    ]

    # see how it worked!
    print("Random\nMean: ", np.mean(random_pop_deviations), "\nStdev: ", np.std(random_pop_deviations),
          "\nGA Results\nMean: ", np.mean(final_pop_deviations), "\nStdev: ", np.std(final_pop_deviations))

    return best_deviation_df, best_deviation_score

# plot the arrangement within the spawn area polygon


def plot_arrangement(polygon=Polygon, arrangement_df=pd.DataFrame):
    fig, ax = plt.subplots()

    xp, yp = polygon.exterior.xy
    plt.plot(xp, yp)

    ax.plot(arrangement_df.x, arrangement_df.y, 'o')
    ax.set_xlim([-90, 70])
    ax.set_ylim([-60, 60])

    # draw all possible edge connections
    for i in range(len(arrangement_df.x)):
        for j in range(i+1, len(arrangement_df.x)):
            ax.plot([arrangement_df.x[i], arrangement_df.x[j]], [
                    arrangement_df.y[i], arrangement_df.y[j]], 'k-', alpha=.09, linewidth=1)

    for i, loc in enumerate(arrangement_df.location):
        ax.annotate(
            str(loc), (arrangement_df.x[i] + 1.5, arrangement_df.y[i] + 1.25), fontsize=10)
