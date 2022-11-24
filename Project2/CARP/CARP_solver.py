import sys
import numpy as np
import re


INT_MAX = 2147483647


def read_file(instance_filename):
    """
    read a dat file in this project. Return a dictionary "file_args" and a numpy array "Gragh" with size (VERTICES, VERTICES, 2).
    Gragh[][][0] is cost, Gragh[][][1] is demand
    """

    file_args = {}
    demand_edge = []
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
                graph = np.full((n, n), INT_MAX)

            else:

                # [idx_a, idx_b, cost, demand] = line.strip().replace("   ", " ").\
                #     replace("   ", " ").\
                #     split(" ")
                [idx_a, idx_b, cost, demand] = re.split(r"[ ]+", line.strip())

                graph[int(idx_a) - 1, int(idx_b) - 1] = int(cost)
                graph[int(idx_b) - 1, int(idx_a) - 1] = int(cost)

                demand = int(demand)
                if demand > 0:
                    demand_edge.append(
                        ((int(idx_a) - 1, int(idx_b) - 1), demand))

            cnt += 1

    return file_args, floyd(graph), demand_edge


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


args = sys.argv[1:]
instance_filename = args[0]
TERMINATE = args[2]
RANDOM_SEED = args[4]


file_args, graph, demand_edge = read_file(instance_filename)
# graph = list(graph)

vertices_num = int(file_args['VERTICES'])
depot_pos = int(file_args['DEPOT'])
required_edge_num = int(file_args['REQUIRED EDGES'])
non_required_edge_num = int(file_args['NON-REQUIRED EDGES'])
edge_num = required_edge_num + non_required_edge_num
capacity = int(file_args['CAPACITY'])

routes = []

for edge in demand_edge:
    route = [edge]
    routes.append(route)

cost = 0
result = ""

result += "s "

for route in routes:
    node = route[0]
    point_a = node[0][0]
    point_b = node[0][1]
    cost += graph[depot_pos, point_a]
    cost += graph[point_a, point_b]
    cost += graph[point_b, depot_pos]

for route in routes:
    result += "0,"
    for node in route:
        result += ("("+str(node[0][0] + 1) + "," +
                   str(node[0][1] + 1)+")" + ",")
    result += "0,"
result = result.strip(',')
result += "\n"
result += "q "
result += str(cost)
print(result)
