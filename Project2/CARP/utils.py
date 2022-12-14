import random
import re

import numpy as np

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
                demand_graph = np.full((n, n), INT_MAX)

            else:
                print(line, re.split(r"[ ]+", line.strip()))
                [idx_a, idx_b, cost, demand] = re.split(r"[ ]+", line.strip())

                graph[int(idx_a) - 1, int(idx_b) - 1] = int(cost)
                graph[int(idx_b) - 1, int(idx_a) - 1] = int(cost)

                demand = int(demand)
                if demand > 0:
                    demand_edge.append((int(idx_a) - 1, int(idx_b) - 1))
                    demand_edge.append((int(idx_b) - 1, int(idx_a) - 1))
                    demand_graph[int(idx_a) - 1, int(idx_b) - 1] = int(demand)
                    demand_graph[int(idx_b) - 1, int(idx_a) - 1] = int(demand)

            cnt += 1

    return file_args, graph, floyd(graph), demand_graph, demand_edge


def divide_route(depot, graph, distance, demand_graph, demand_edges, capacity, iter_num, random_seed, start, time,
                 terminate, q):
    """
    args: depot, graph, distance, demand_graph, demand_edges, capacity, iter_num, random_seed, start, time, terminate
    """

    final_routes, init_cost = path_scanning(depot, graph, distance, demand_graph, demand_edges, capacity, iter_num,
                                            random_seed, start, time,
                                            terminate)
    q.put([final_routes, init_cost])


def path_scanning(depot, graph, distance, demand_graph, demand_edges, capacity, iter_num, random_seed, start, time,
                  terminate):
    """
    args: depot, graph, distance, demand_graph, demand_edges, capacity, iter_num, random_seed, start, time, terminate
    """
    # random.seed(random_seed)
    init_cost = INT_MAX
    final_routes = []

    for _ in range(INT_MAX):
        if time.time() - start > float(terminate) - 0.25:
            break
        last_point = depot
        routes = []
        demand_edges_d = demand_edges.copy()
        while demand_edges_d:
            route = []
            carry = 0
            while carry < capacity:
                edges_to_choose = find_minimal(
                    last_point, demand_edges_d, distance)
                # edges_to_choose = demand_edges_d
                # if carry <= capacity / 2:
                #     edges_to_choose = find_maximal(depot, edges_to_choose, distance)
                # elif carry > capacity / 2:
                #     edges_to_choose = find_minimal(depot, edges_to_choose, distance)
                if not edges_to_choose:
                    break

                edges_to_choose = list(
                    filter(lambda edge: demand_graph[edge[0], edge[1]] <= capacity - carry, edges_to_choose))

                if edges_to_choose:
                    edge_choose = random.choice(edges_to_choose)

                    add_edge(edge_choose, route, demand_edges_d)
                    carry += demand_graph[edge_choose[0], edge_choose[1]]
                    last_point = edge_choose[1]

                else:
                    break

            routes.append(route)

        cost = cal_cost(routes, graph, distance, depot)
        if cost < init_cost:
            final_routes = routes
            init_cost = cost
        # print(cost, init_cost)
    # q.put([final_routes, init_cost])
    return final_routes, init_cost


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
            cost += distance[last_point, start_point]
            cost += graph[start_point, end_point]
            last_point = end_point
        cost += distance[last_point, depot]
    return cost


def print_result(routes, cost):
    result = "s "
    for route in routes:
        result += "0,"
        for edge in route:
            result += ("(" + str(edge[0] + 1) + "," +
                       str(edge[1] + 1) + ")" + ",")
        result += "0,"
    result = result.strip(',')
    result += "\n"
    result += "q "
    result += str(cost)
    print(result)


def delete_edge(edge, demand_edge):
    (a, b) = edge
    if (a, b) in demand_edge:
        demand_edge.remove((a, b))
    if (b, a) in demand_edge:
        demand_edge.remove((b, a))


def add_edge(edge, route, demand_edges):
    route.append(edge)
    delete_edge(edge, demand_edges)


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


def find_minimal(init_pos, demand_edges, distance):
    """
    return a list, which contain edges in demand_edges that have minimal distance to init_pos
    """
    min_dis = INT_MAX
    min_edge = []

    for edge in demand_edges:
        start_pos = edge[0]
        if distance[init_pos, start_pos] < min_dis:
            min_dis = distance[init_pos, start_pos]

    for edge in demand_edges:
        if distance[init_pos, edge[0]] == min_dis:
            min_edge.append(edge)

    return min_edge


def find_maximal(init_pos, demand_edges, distance):
    """
    return a list, which contain edges in demand_edges that have maximal distance to init_pos
    """
    max_dis = 0
    max_edge = []

    for edge in demand_edges:
        start_pos = edge[0]
        if distance[init_pos, start_pos] > max_dis:
            max_dis = distance[init_pos, start_pos]

    for edge in demand_edges:
        if distance[init_pos, edge[0]] == max_dis:
            max_edge.append(edge)

    return max_edge
