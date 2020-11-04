import random
import copy
import numpy as np
from pprint import pprint
from typing import List, Tuple, Dict
from math import ceil
from hashlib import sha1

class Crossover:
	@staticmethod
	def single_point_crossover(solution1: List[Tuple[str, int]], solution2: List[Tuple[str, int]], point: int) -> Tuple[Tuple[str, int], Tuple[str, int]]:
		"""
		Returns two children solutions based on a single-point crossover between the two given solutions.

		Args:
			solution1: One solution to be used in crossover with size N.
			solution2: Another solution to be used in crossover with size N.
			point: An integer in [0, N] representing the point to split the solutions.

		Returns:
			new_solution1, new_solution2: A tuple with two solutions containing the two ways to split and merging the parents using the given point.
		"""
		A = solution1[:point].append(solution2[point:])
		B = solution2[:point].append(solution1[point:])

		return A, B

	@staticmethod
	def multi_point_crossover(solution1: List[Tuple[str, int]], solution2: List[Tuple[str, int]], points: List[int]) -> Tuple[Tuple[str, int], Tuple[str, int]]:
		"""
		Returns two children solutions based on a multi-point crossover between the two given solutions.

		Args:
			solution1: One solution to be used in crossover with size N.
			solution2: Another solution to be used in crossover with size N.
			points: A list with integers in [0, N] representing the points to split the solutions.

		Returns:
			new_solution1, new_solution2: A tuple with two solutions containing the two ways to split and merging the parents using the given points.
		"""
		A = solution1
		B = solution2

		for p in points:
			A, B = Crossover.single_point_crossover(A, B, p)

		return A, B

	@staticmethod
	def uniform_crossover(solution1: List[Tuple[str, int]], solution2: List[Tuple[str, int]], chances: List[float], threshold: float=0.5) -> Tuple[Tuple[str, int], Tuple[str, int]]:
		"""
		Returns two children solutions based on a uniform crossover between the two given solutions.

		Args:
			solution1: One solution to be used in crossover with size N.
			solution2: Another solution to be used in crossover with size N.
			chances: A list with floats in [0, 1] representing the chances to a point in the solution be swapped.
			threshold: A float in [0, 1] representing the number which a specific chance must be greater to not be swapped.

		Returns:
			new_solution1, new_solution2: A tuple with two solutions containing the resulting solutions after the swaps.
		"""
		A = solution1
		B = solution2

		for i in range(len(chances)):
			if chances[i] < threshold:
				temp = A[i]
				A[i] = B[i]
				B[i] = temp

		return A, B

class GeneticAlgorithm:
	UNIQUE_POINT_CROSSOVER = 1
	MULTI_POINT_CROSSOVER = 2
	UNIFORM_CROSSOVER = 3

	MAXIMIZE = 1
	MINIMIZE = 2

	def __init__(self, initial_solution, solution_size, elite_size, mutation_chance, mutation_chance_increase, population_size, rng_seed, fitness_function, order, **kwargs):
		self.initial_solution = initial_solution
		self.solution_size = solution_size
		self.elite_size = elite_size
		self.mutation_chance = mutation_chance
		self.mutation_chance_increase = mutation_chance_increase
		self.population_size = population_size
		self.seed = rng_seed
		self.fitness_function = fitness_function
		if order == GeneticAlgorithm.MAXIMIZE:
			self.order = True
		else:
			self.order = False
		self.default_fitness_args = kwargs

		random.seed(self.seed)
		np.random.seed(self.seed)

	def run(self, crossover_type, max_iterations, **kwargs):
		population = first_generation(self.population_size, self.solution_size, 0)

		for iteration in range(max_iterations):
			print(f'Iteration: {iteration}')
			values = []

			for solution in population:
				values.append((self.fitness_function(solution, **self.default_fitness_args), solution))

			elite = [x for _, x in sorted(values, key=lambda x: x[0], reverse=self.order)]
			print(f'Best value: {sorted(values,  key=lambda x: x[0], reverse=self.order)[0][0]}')
			parents = []
			index = 0

			for solution in elite:
				if index < self.elite_size:
					parents.append(solution)

				index += 1

			new_population = parents

			for index in range(ceil(self.population_size / 2)):
				solution1, solution2 = self.crossover(random.choice(parents), random.choice(parents), crossover_type, **kwargs)
				solution1, solution2 = self.mutation(solution1), self.mutation(solution2)

				new_population.append(solution1)
				new_population.append(solution2)
			
			population = new_population[:self.population_size]

	def crossover(self, solution1: Dict[str, int], solution2: Dict[str, int], crossover_type: int, **kwargs) -> Tuple[Dict[str, int], Dict[str, int]]:
		solution1_as_list = list(solution1.items())
		solution2_as_list = list(solution2.items())

		if crossover_type == GeneticAlgorithm.UNIQUE_POINT_CROSSOVER:
			try:
				point = kwargs['point']
			except KeyError:
				point = len(solution1_as_list) // 2

			solution1_as_list, solution2_as_list = Crossover.single_point_crossover(solution1_as_list, solution2_as_list, point)
			return dict(solution1_as_list), dict(solution2_as_list)
		elif crossover_type == GeneticAlgorithm.MULTI_POINT_CROSSOVER:
			try:
				points = kwargs['points']
			except KeyError:
				points = [len(solution1_as_list) // 2]

			solution1_as_list, solution2_as_list = Crossover.multi_point_crossover(solution1_as_list, solution2_as_list, points)
			return dict(solution1_as_list), dict(solution2_as_list)
		elif crossover_type == GeneticAlgorithm.UNIFORM_CROSSOVER:
			try:
				chances = kwargs['chances']
				threshold = kwargs['threshold']
			except KeyError:
				threshold = 0.5
				chances = [random.uniform(0, 1) for i in range(len(solution1_as_list))]

			solution1_as_list, solution2_as_list = Crossover.uniform_crossover(solution1_as_list, solution2_as_list, chances, threshold)
			return dict(solution1_as_list), dict(solution2_as_list)
		
		raise Exception('Crossover type not implemented')

	def mutation(self, solution, mutation_chance: float=None, seed: int=None):
		if mutation_chance == None:
			mutation_chance = self.mutation_chance

		if seed == None:
			seed = self.seed

		np.random.seed(seed)
		chances = np.random.rand(self.solution_size)

		index = 0
		for key in solution:
			random_value = np.random.randint(self.solution_size)
			while random_value == solution[key]:
				random_value = np.random.randint(self.solution_size)

			solution[key] = random_value if chances[index] > (1 - mutation_chance) else solution[key]
			index += 1

		return solution

def first_generation(population_size, vertices_number, change_percentual, base_solution = None):
	population = []

	if base_solution == None:
		base_solution: Dict[str, int] = {}
		for vertex in range(vertices_number):
			base_solution[str(vertex)] = vertex

	for _ in range(population_size):
		solution: Dict[str, int] = {}

		for vertex in range(vertices_number):
			vertex = str(vertex)
			solution[vertex] = int(np.random.normal(loc=base_solution[vertex], scale=base_solution[vertex]*change_percentual))

		population.append(solution)

	return population
