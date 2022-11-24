import numpy as np

INT_MAX = 2147483647


def read_file(instance_filename):
    """
    read a dat file in this project. Return a dictionary "file_args" and a numpy array "Gragh" with size (VERTICES, VERTICES, 2).
    Gragh[][][0] is cost, Gragh[][][1] is demand
    """

    """
    NAME
    VERTICES
    DEPOT
    REQUIRED EDGES
    NON-REQUIRED EDGES
    VEHICLES
    CAPACITY
    TOTAL COST OF REQUIRED EDGES
    """
    file_args = {}
    with open(instance_filename, 'r', encoding='utf-8') as instance_file:
        cnt = 0
        for line in instance_file:
            if line == 'END':
                break
            if cnt < 8:
                line = line.split(":")
                index = line[0].strip()
                value = line[1].strip()
                file_args[index] = value

            elif cnt == 8:
                n = int(file_args['VERTICES'])
                gragh = np.full((n, n, 2), INT_MAX)

            else:

                [idx_a, idx_b, cost, demand] = line.strip().replace("   ", " ").\
                    replace("   ", " ").\
                    split(" ")

                gragh[int(idx_a) - 1, int(idx_b) - 1, :] = \
                    [int(cost), int(demand)]

                gragh[int(idx_b) - 1, int(idx_a) - 1, :] = \
                    [int(cost), int(demand)]

            cnt += 1

    return file_args, gragh


def floyd(gragh):
    distance = gragh.copy()

    vertices_idx = range(len(gragh))

    for i in vertices_idx:
        distance[i, i] = 0

    for i in vertices_idx:
        i_to_cost = distance[i]
        to_i_cost = distance[:, i]
        for to_i_node in range(len(to_i_cost)):
            for i_to_node in range(len(i_to_cost)):
                if distance[to_i_node, i_to_node] > to_i_cost[to_i_node] + i_to_cost[i_to_node]:
                    distance[to_i_node, i_to_node] = to_i_cost[to_i_node] + \
                        i_to_cost[i_to_node]

    return distance
