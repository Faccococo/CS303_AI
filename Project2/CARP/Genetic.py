from utils import *


def Genetic(depot, graph, distance, demand_graph, demand_edges, capacity, iter_num, random_seed, start, time,
            terminate):
    # init population
    mutate_rate = 0.2

    population, _ = path_scanning(depot, graph, distance, demand_graph, demand_edges, capacity, 100,
                                  random_seed,
                                  start, time,
                                  terminate)

    # 演化
    for gen in iter_num:
        next_gen = Genetic_reproduce()
        population = Genetic_replacement()

    cost_out = cal_cost(population, graph, distance, depot)
    return population, cost_out


def Genetic_reproduce():
    # TODO

    return []


def Genetic_reproduce_select(population):
    # TODO

    return []


def Genetic_reproduce_recombine(pop1, pop2):
    # TODO
    pass


def Genetic_reproduce_mutate(pop, mutate_rate):
    # TODO
    pass


def Genetic_replacement():
    return []


def Genetic_fitness():
    return 0
