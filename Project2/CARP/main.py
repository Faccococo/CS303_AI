import sys
import numpy as np
from utils import read_file

args = sys.argv[1:]
instance_filename = args[0]
TERMINATE = args[2]
RANDOM_SEED = args[4]


file_args, graph, demand_edge = read_file(instance_filename)
# graph = list(graph)
print(graph)

vertices_num = int(file_args['VERTICES'])
depot_pos = int(file_args['DEPOT']) - 1
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
    cost_pre = cost
    cost += graph[depot_pos, point_a]
    cost += graph[point_a, point_b]
    cost += graph[point_b, depot_pos]
    # print(depot_pos, "  ", end="")
    print(point_a, point_b, " ", end="")
    print(cost - cost_pre)

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

# f = open("output.txt", "w")
# f.write(result)
# f.close()
