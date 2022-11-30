import numpy as np
import re


INT_MAX = 1000000000


def readData(instance_filename):
    """
    read a dat file in this project. Return a dictionary "file_args" and a numpy array "graph" with size (VERTICES, VERTICES, 2).
    Graph[][][0] is cost, Graph[][][1] is demand
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

    return file_args, graph, floyd(graph), demand_edge

def floyd(graph):
    distance = graph.copy()

    vertices_idx = range(len(graph))

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

def divide_route(demand_edge):
    routes = []
    for edge in demand_edge:
        route = [edge]
        routes.append(route)
    return routes

def cal_cost(routes, graph, distance, depot):
    """
    routes: routes the car deal
    graph: the origin graph before floyd
    distance: the graph after floyd
    depot: the depot pos
    """
    cost = 0
    

    for route in routes:
        last_point = depot
        for edge in route:
            (start_point, end_point) = edge
            cost += distance[last_point][start_point]
            cost += graph[start_point][end_point]
            last_point = end_point
        cost += distance[last_point][depot]
    return cost

def print_result(routes, cost):
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
    return(result)










