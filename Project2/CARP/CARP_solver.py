import sys
import numpy as np
from utils import read_file

args = sys.argv[1:]
instance_filename = args[0]
TERMINATE = args[2]
RANDOM_SEED = args[4]

file_args, gragh = read_file(instance_filename)

vertices_num = file_args['VERTICES']
depot_pos = file_args['DEPOT']
required_edge_num = file_args['REQUIRED EDGES']
non_required_edge_num = file_args['NON-REQUIRED EDGES']
edge_num = required_edge_num + non_required_edge_num
capacity = file_args['CAPACITY']
