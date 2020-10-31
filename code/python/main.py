# -*- coding: utf-8 -*-

import argparse
import sys
from pprint import pprint
import time

def main(args):
    filename = (args.file[0].name)

    graph = {}
    prev_vertex = -1

    with open(filename, 'r') as f:
        # G = (V, E)
        # p => params
        # p edge #V #E
        for line in f:
            if(line.startswith("c")):
              print(" ".join(line.split()[1:]))
              continue
            if(line.startswith("p")):
              print(" ".join(line.split()[1:]))
              numVertices, numEdges = line.split()[2:]
            elif(line.startswith("e")):
              vertex1, vertex2 = line.split()[1:]
              if(vertex1 in graph.keys()):
                graph[vertex1].append(vertex2)
              else:
                graph[vertex1] = [vertex2]
              if(vertex2 in graph.keys()):
                graph[vertex2].append(vertex1)
              else:
                graph[vertex2] = [vertex1]

    # start = time.perf_counter_ns()
    # genetic(graph, ...)
    # end = time.perf_counter_ns()
    # print("EXECUTION TIME")
    # print(end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Uses Genetic Algorithm to find solution for VC.',
        usage='main.py [-h] [--file myinstance.txt]')

    parser.add_argument('-f', '--file', type=open, nargs=1,
                        help='a file containing the instance',
                        required=True)

    args = parser.parse_args()
    print(args.file[0].name)

    main(args)