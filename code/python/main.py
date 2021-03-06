# -*- coding: utf-8 -*-

import argparse
import sys
import time

from genetic import GeneticAlgorithm, CrossoverMethod, GeneticAlgorithm2
from typing import Dict, List
from math import floor, sqrt


def main(args):
    filename = args.file[0].name
    seed = args.seed[0]
    csv_filename = args.output[0] if args.output else 'csv_file.csv'
    color_filename = args.color[0] if args.color else 'color_file.color'
    input_filename = args.input[0].name if args.input else None
    initial_solution = None

    graph = {}
    prev_vertex = -1

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('c'):
                print(''.join(line.split()[1:]))
                continue
            elif line.startswith('p'):
                print(''.join(line.split()[1:]))
                num_vertices, num_edges = line.split()[2:]
            elif line.startswith('e'):
                vertex1, vertex2 = line.split()[1:]
                vertex1 = str(int(vertex1) - 1)
                vertex2 = str(int(vertex2) - 1)

                if vertex1 in graph.keys():
                    graph[vertex1].append(vertex2)
                else:
                    graph[vertex1] = [vertex2]

                if vertex2 in graph.keys():
                    graph[vertex2].append(vertex1)
                else:
                    graph[vertex2] = [vertex1]

    if input_filename:
        with open(input_filename, 'r') as f:
            initial_solution = [] * int(num_vertices)
            index = 0
            for line in f:
                initial_solution[index] = int(line)
                index += 1

    # genetic = GeneticAlgorithm(initial_solution=None, solution_size=int(num_vertices), elite_size=10, mutation_chance=0.5,
    #                            population_size=1000, rng_seed=seed, fitness_function=fitness, order=GeneticAlgorithm.MINIMIZE, graph=graph, num_vertices=int(num_vertices))
    
    genetic2 = GeneticAlgorithm2(population_size=50, rng_seed=seed, graph=graph, initial_solution=initial_solution)

    num_edges = int(num_edges)

    max_iterations = max(min(floor((16985.0/2059904.0)*(num_edges ** 2) - (574505.0/13552.0)*num_edges + (62282450.0/1463.0)), 37000), 4000)
    
    start = time.perf_counter_ns()
    # genetic.run(max_iterations=500, crossover_type=CrossoverMethod.UNIFORM_CROSSOVER, csv_filename=csv_filename)
    num_colors, solution = genetic2.run(max_iterations=max_iterations, csv_filename=csv_filename)
    print("num colors = {}".format(num_colors))
    end = time.perf_counter_ns()
    print(f'EXECUTION TIME: {(end-start)/(10**9)} seconds.')

    if genetic2.calc_fitness(solution) != 0:
        num_colors = 'UNDEFINED'

    open(color_filename, 'w').write(f'color={num_colors}, time={(end-start)/(10**9)}')


def fitness(solution: Dict[str, int], graph: Dict[str, List[str]], num_vertices: int) -> int:
    """
	Returns a value for a specific solution of VC problem in a graph.

	Complexity: O(v^2) where v is the number of vertices.

	Args:
		solution: A solution for the VC problem.
        graph: The graph represented with a dictionary whose keys are the nodes of the graph. For each key, 
            the corresponding value is a list containing the nodes that are connected by a direct arc from this node.
		num_vertices: The number of vertices of the graph.

	Returns:
		value: The number of colors used in this solution. 
            If the solution breaks at least one rule of this problem, then it returns 2*v (where v is the number of vertices). 
	"""
    color_set = {}

    for vertex in solution:
        color_set[solution[vertex]] = True
        for neighbor in graph[str(vertex)]:
            if solution[neighbor] == solution[vertex]:
                return num_vertices*2

    return len(color_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Uses Genetic Algorithm to find solution for VC.',
        usage='main.py [-h] [--file myinstance.txt] [--seed 1]')

    parser.add_argument('-f', '--file', type=open, nargs=1,
                        help='a file containing the instance',
                        required=True)

    parser.add_argument('-s', '--seed', type=int, nargs=1,
                        help='an integer representing the seed for random number generation',
                        required=True)
    
    parser.add_argument('-o', '--output', type=str, nargs=1,
                        help='a string representing the .csv filename to be written',
                        required=False)

    parser.add_argument('-c', '--color', type=str, nargs=1,
                        help='a string representing the .color filename to be written with the color and the time',
                        required=False)

    parser.add_argument('-i', '--input', type=open, nargs=1,
                        help='a file containing a initial solution',
                        required=False)

    args = parser.parse_args()

    main(args)
