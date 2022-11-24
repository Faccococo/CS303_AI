import numpy as np


class Gragh:
    def __init__(self, n):
        """
        n is the vertices number in the gragh
        """
        self.vertices = []
        for i in range(n):
            self.vertices.append(Node(i))

    def addEdge(self, i, j, cost, demand):
        self.vertices[i].addEdge(self.vertices[j], cost, demand)
        self.vertices[j].addEdge(self.vertices[i], cost, demand)

    def get_vertices(self, i):
        return self.vertices[i]


class Node:

    def __init__(self, n) -> None:
        self.neighbors = []  # neighbors[i] = (node, cost, demand)
        self.idx = n

    def addEdge(self, node, cost, demand):
        self.neighbors.append((node, cost, demand))


def read_file(instance_filename):
    """
    return a dictionary "file_args" and a numpy array "Gragh" with size (VERTICES, VERTICES, 2).
    Gragh[][][0] is cost, Gragh[][][1] is demand
    """

    """
    NAME
    VERTICES
    DEPOT
    REQUIRED EDGES
    NON-REQUIRED EDGES
    VEHICLES
    CAPACITY
    TOTAL COST OF REQUIRED EDGES
    """
    file_args = {}
    with open(instance_filename, 'r', encoding='utf-8') as instance_file:
        cnt = 0
        for line in instance_file:
            if line == 'END':
                break
            if cnt < 8:
                line = line.split(":")
                index = line[0].strip()
                value = line[1].strip()
                file_args[index] = value

            elif cnt == 8:
                n = int(file_args['VERTICES'])
                gragh = Gragh(n)

            else:
                # TODO: read edges
                [idx_a, idx_b, cost, demand] = line.strip().replace("   ", " ").\
                    replace("   ", " ").\
                    split(" ")
                gragh.addEdge(int(idx_a) - 1, int(idx_b) - 1,
                              int(cost), int(demand))
            cnt += 1

    return file_args, gragh
