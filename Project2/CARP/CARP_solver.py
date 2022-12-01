from utils import *
import sys
import time
from multiprocessing import Process

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
iter_num = 5000
# out[i] is a list, out[i] = [routes, cost]
outs = [[], [], [], []]

p1 = Process(target=path_scanning,
             args=(
                 depot_pos, graph, distance, demand_graph, demand_edges, capacity, iter_num, RANDOM_SEED, start, time, TERMINATE, outs[0]))


p2 = Process(target=path_scanning,
             args=(
                 depot_pos, graph, distance, demand_graph, demand_edges, capacity, iter_num, RANDOM_SEED, start, time, TERMINATE, outs[1]))

p3 = Process(target=path_scanning,
             args=(
                 depot_pos, graph, distance, demand_graph, demand_edges, capacity, iter_num, RANDOM_SEED, start, time, TERMINATE, outs[2]))

p4 = Process(target=path_scanning,
             args=(
                 depot_pos, graph, distance, demand_graph, demand_edges, capacity, iter_num, RANDOM_SEED, start, time, TERMINATE, outs[3]))

p1.start()
p2.start()
p3.start()
p4.start()

p1.join()
p2.join()
p3.join()
p4.join()

[routes, cost] = outs[0]
for out in outs:
    if out[1] < cost:
        routes = out[0]
        cost = out[1]

# routes = divide_route(demand_edges)
# routes, cost = path_scanning(depot_pos, graph, distance, demand_graph, demand_edges, capacity, iter_num, RANDOM_SEED, start, time, TERMINATE, outs[0])
# print(routes)
print(time.time() - start)
print_result(routes, cost)
