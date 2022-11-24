import sys
import numpy as np
from utils import read_file, Gragh, Node

args = sys.argv[1:]
instance_filename = args[0]
TERMINATE = args[2]
RANDOM_SEED = args[4]

file_args, gragh = read_file(instance_filename)
