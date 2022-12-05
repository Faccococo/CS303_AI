import multiprocessing
import sys
import time

from utils import *

start = time.time()
args = sys.argv[1:]
instance_filename = args[0]
TERMINATE = args[2]
RANDOM_SEED = args[4]

file_args, graph, distance, demand_graph, demand_edges = readData(instance_filename)

vertices_num = int(file_args['VERTICES'])
depot_pos = int(file_args['DEPOT']) - 1
required_edge_num = int(file_args['REQUIRED EDGES'])
non_required_edge_num = int(file_args['NON-REQUIRED EDGES'])
edge_num = required_edge_num + non_required_edge_num
capacity = int(file_args['CAPACITY'])
iter_num = 50000
process_num = 8
# out[i] is a list, out[i] = [routes, cost]
jobs = []

q = multiprocessing.Queue()
for i in range(process_num):
    p = multiprocessing.Process(target=divide_route,
                                args=(
                                    depot_pos, graph, distance, demand_graph, demand_edges, capacity, iter_num,
                                    RANDOM_SEED, start,
                                    time,
                                    TERMINATE, q))
    jobs.append(p)
    p.start()

for p in jobs:
    p.join()

outs = [q.get() for j in jobs]

[routes, cost] = outs[0]
for out in outs:
    if out[1] < cost:
        routes = out[0]
        cost = out[1]

# routes = divide_route(demand_edges) routes, cost = path_scanning(depot_pos, graph, distance, demand_graph,
# demand_edges, capacity, iter_num, RANDOM_SEED, start, time, TERMINATE, outs[0]) print(routes)
print_result(routes, cost)
# print(time.time() - start)
