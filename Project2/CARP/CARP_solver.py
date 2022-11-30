from utils import *
import sys

args = sys.argv[1:]
instance_filename = args[0]
TERMINATE = args[2]
RANDOM_SEED = args[4]

file_args, graph, distance, demand_edge = readData(instance_filename)

vertices_num = int(file_args['VERTICES'])
depot_pos = int(file_args['DEPOT']) - 1
required_edge_num = int(file_args['REQUIRED EDGES'])
non_required_edge_num = int(file_args['NON-REQUIRED EDGES'])
edge_num = required_edge_num + non_required_edge_num
capacity = int(file_args['CAPACITY'])

routes = divide_route(demand_edge)
cost = cal_cost(routes, graph, distance, depot_pos)
print_result(routes, cost)


