from utils import *

INT_MAX = 1000000000


def Genetic(depot, graph, distance, demand_graph, demand_edges, capacity, iter_num, random_seed, start, time,
            terminate):
    # init population
    mutate_rate = 0.2
    population = []
    utrial = 50
    trial = 0
    pop_size = 50
    pop_num = 0

    while trial < utrial or pop_num < pop_size:
        routes, _ = path_scanning(depot, graph, distance, demand_graph, demand_edges, capacity, 10,
                                  random_seed,
                                  start, time,
                                  terminate)
        if routes not in population:
            population.append(routes)
            pop_num += 1
            trial = 0
        else:
            trial += 1

    # 演化
    for gen in iter_num:
        next_gen = Genetic_reproduce(population, mutate_rate)
        population = Genetic_replacement()

    routes_out = None
    cost_out = INT_MAX

    for pop in population:
        cost = cal_cost(pop, graph, distance, depot)
        if cost < cost_out:
            cost_out = cost
            routes_out = pop

    return routes_out, cost_out


def Genetic_reproduce(population, mutate_rate):
    return_list = []
    for _ in range(len(population)):
        routes_1, routes_2 = Genetic_reproduce_select(population)
        son_routes = Genetic_reproduce_recombine(routes_1, routes_2)
        son_routes = Genetic_reproduce_mutate(son_routes, mutate_rate)
        return_list.append(son_routes)
    return return_list


def Genetic_reproduce_select(population, graph, distance, depot):
    # TODO
    population_scores = []
    return_list = []

    for pop in population:
        population_scores.append(Genetic_score(pop, graph, distance, depot))

    threshold = 0
    threshold_list = []
    score_sum = sum(population_scores)
    for i in population_scores:
        threshold_list.append(threshold)
        threshold += i / score_sum
    threshold_list.append(1)

    for _ in range(2):
        temp = random.randrange(0, 10) / 10
        for i in range(len(population)):
            if threshold_list[i] <= temp <= threshold_list[i + 1]:
                return_list.append(population[i])
                break

    return return_list


def Genetic_reproduce_recombine(routes1, routes2, demand_edge):
    # TODO
    demand_edge_d = demand_edge.copy()

    return routes1


def Genetic_reproduce_mutate(pop, mutate_rate):
    a = random.random()
    if a > mutate_rate:
        return pop
    else:
        # TODO
        pass
    return pop


def Genetic_replacement():
    return []


def Genetic_score(routes, graph, distance, depot):
    return 1 / cal_cost(routes, graph, distance, depot)
