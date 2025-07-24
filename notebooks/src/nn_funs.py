import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# create a distance matrix between all positions that can be accessed


def euclid_distance_matrix(df=pd.DataFrame):
    # create empty matrix of necessary size
    n = len(df)
    dist_matrix = np.zeros((n, n), dtype=float)

    # prespecify the location coord list
    location_i = []
    location_j = []

    # fill matrix by calculating dist between appropriate locations
    for i in range(n):
        for j in range(n):
            location_i = [df.x.iloc[i], df.y.iloc[i]]
            location_j = [df.x.iloc[j], df.y.iloc[j]]

            # calculate euclid distance
            dist = np.sqrt(
                np.pow(location_i[0] - location_j[0], 2) +
                np.pow(location_i[1] - location_j[1], 2)
            )

            if (dist == 0):
                # set to NA value of 999
                dist_matrix[i][j] = 999
            else:
                dist_matrix[i][j] = dist

    return dist_matrix

# find the value of the individual (route)


def total_distance_nn_route(
    individual=list, dist_matrix=np.matrix, path_len=int
):
    dists = [
        dist_matrix[individual[i]-1][individual[0]-1] if i == path_len-1 else dist_matrix[individual[i]-1][individual[i+1]-1] for i in range(0, path_len)
    ]

    return sum(dists)

# using a given location find the nearest neighbor


def find_nearest_neighbor(
    dist_matrix=np.matrix, location=int,
    previously_visited=list, path_len=int
):
    # create new df with distances along with
    df = pd.DataFrame({
        'location': [*range(1, path_len+1)],
        'distances': dist_matrix[location-1]
    },
        index=[*range(0, path_len)]
    )

    # see if this df should be filtered for previously visited locations
    if len(previously_visited) > 0:
        # remove previously visited locations
        df = df[~df.location.isin(previously_visited)]

    # sort according to distance
    df = df.sort_values('distances')

    # select next closest neighbor
    nn = [*df.location][0]

    return nn


# using a starting point on the arrangement, determine nn path


def run_nn(
    start_location=int, dist_matrix=np.matrix, path_len=int
):
    path = []

    path.append(start_location)

    for i in range(0, path_len-1):
        next_stop = find_nearest_neighbor(
            dist_matrix=dist_matrix, location=path[-1], previously_visited=path,
            path_len=path_len
        )

        path.append(next_stop)

    return path


def run_all_nn(
    locations_df=pd.DataFrame, dist_matrix=np.matrix, path_len=int
):
    # try all variations
    start_locations = [*locations_df.location]

    # # init dist matrix
    # dist_matrix = euclid_distance_matrix(locations_df)

    paths = [
        run_nn(start_location=location, dist_matrix=dist_matrix, path_len=path_len) for location in start_locations
    ]
    dists = [
        total_distance_nn_route(path, dist_matrix=dist_matrix, path_len=path_len) for path in paths
    ]

    nn_performance_dict = {
        'start_locations': start_locations,
        'paths': paths,
        'distances': dists
    }

    return nn_performance_dict


def plot_nn_path(starting_location=int, nn_dict=dict, locations_df=pd.DataFrame):
    focal_path = nn_dict['paths'][starting_location-1]
    focal_distance = nn_dict['distances'][starting_location-1]

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

    plt.title(label="TSP route using NN", fontsize=14, color='k')
    plt.suptitle("Total Distance Traveled: " +
                 str(round(focal_distance, 1)), fontsize=9)

    for i, loc in enumerate(focal_path):
        ax.annotate(
            str(loc), (x_shortest[i] + 1.5, y_shortest[i] + 1.25), fontsize=10)

    fig.set_size_inches(6.5, 6)
    plt.show()
