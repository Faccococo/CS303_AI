import sys
import numpy as np
from utils import read_file

args = sys.argv[1:]
instance_filename = args[0]
TERMINATE = args[2]
RANDOM_SEED = args[4]



file_args, graph, demand_edge = read_file(instance_filename)
graph = list(graph)

vertices_num = file_args['VERTICES']
depot_pos = file_args['DEPOT']
required_edge_num = file_args['REQUIRED EDGES']
non_required_edge_num = file_args['NON-REQUIRED EDGES']
edge_num = required_edge_num + non_required_edge_num
capacity = file_args['CAPACITY']

routes = []

for edge in demand_edge:
    route = [edge.copy()]
    routes.append(route)

print(routes)



