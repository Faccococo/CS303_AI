from utils import *


def Genetic(depot, graph, distance, demand_graph, demand_edges, capacity, iter_num, random_seed, start, time,
            terminate):
    # init population
    population = []
    init_pop_num = 100
    mutate_rate = 0.2
    for _ in range(init_pop_num):
        routes, _ = path_scanning(depot, graph, distance, demand_graph, demand_edges, capacity, 100,
                                  random_seed,
                                  start, time,
                                  terminate)
        population.append(routes)

    # 演化
    for gen in iter_num:
        next_gen = Genetic_reduce(population, mutate_rate)
        population = Genetic_replacement(population, next_gen)

    final_routes = min(population, key=Genetic_fitness)
    cost_out = cal_cost(final_routes, graph, distance, depot)
    return final_routes, cost_out


def Genetic_reproduce(population, mutate_rate):
    # TODO
    return_list = []
    for i in range(len(population)):
        return_list.append(
            Genetic_reproduce_mutate(
                Genetic_reproduce_recombine(
                    *Genetic_reproduce_select(population)
                ),
                mutate_rate
            )
        )
    return return_list


def Genetic_reproduce_select(population):
    # TODO
    return []


def Genetic_reproduce_recombine(pop1, pop2):
    # TODO
    pass


def Genetic_reproduce_mutate(pop, mutate_rate):
    # TODO
    pass


def Genetic_replacement(old, new):
    # TODO
    entity = old + new

    return []


def Genetic_fitness(routes, graph, distance, depot):
    return cal_cost(routes, graph, distance, depot)
