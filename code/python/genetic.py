import random
import copy
import numpy as np
from pprint import pprint
from typing import List, Tuple, Dict
from math import ceil, remainder
from hashlib import sha1
from enum import Enum
import csv 

class CrossoverMethod(Enum):
	UNIQUE_POINT_CROSSOVER = 1
	MULTI_POINT_CROSSOVER = 2
	UNIFORM_CROSSOVER = 3

class GeneticTypeFitnessFunction(Enum):
	MAXIMIZE = 1
	MINIMIZE = 2

class Crossover:
	@staticmethod
	def single_point_crossover(solution1: List[Tuple[str, int]], solution2: List[Tuple[str, int]], point: int) -> Tuple[Tuple[str, int], Tuple[str, int]]:
		"""
		Returns two children solutions based on a single-point crossover between the two given solutions.

		Complexity: O(n), with n being the solution size.

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

		Complexity: O(n^2), where n is the solution size.

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

		Complexity: O(n), where n is the solution size.

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
	MAXIMIZE = GeneticTypeFitnessFunction.MAXIMIZE
	MINIMIZE = GeneticTypeFitnessFunction.MINIMIZE

	def __init__(self, initial_solution: Dict[str, int], solution_size: int, population_size: int, elite_size: int, mutation_chance: float, rng_seed: int, order: GeneticTypeFitnessFunction, fitness_function, **kwargs):
		"""
		Returns an object of GeneticAlgorithm with all the necessary attributes to run it.

		Complexity: O(1)

		Args:
			initial_solution: The first solution to be used as mean and generate the first population.
			solution_size: The length of the solution.
			population_size: The number of solutions for generation (must be greater than 0).
			elite_size: The length of the elite (must be less or equal than population_size).
			mutation_chance: The chance to a solution mutate for generation (must be in [0, 1]).
			rng_seed: The seed to be used in random numbers generation.
			order: A flag that sets if the algorithm wants to maximize ou minimize the fitness value.
				- GeneticTypeFitnessFunction.MAXIMIZE: maximize the fitness function.
				- GeneticTypeFitnessFunction.MINIMIZE: minimize the fitness function.
			fitness_function: The function to calculate the fitness value for each solution.
			**kwargs: The arguments for the given fitness function.

		Returns:
			genetic: An object of GeneticAlgorithm.
		"""
		self.initial_solution = initial_solution
		self.solution_size = solution_size

		if population_size > 0:
			self.population_size = population_size	
		else: 
			raise Exception('Population size must be greater than 0.')

		if population_size >= elite_size:
			self.elite_size = elite_size
		else:
			raise Exception('Elite size must be less or equal than population size.')

		if 0 <= mutation_chance <= 1:
			self.mutation_chance = mutation_chance
		else:
			raise Exception('Mutation chance must be in [0, 1].')

		self.seed = rng_seed
		random.seed(self.seed)
		np.random.seed(self.seed)

		if order == GeneticTypeFitnessFunction.MAXIMIZE:
			self.order = True
		else:
			self.order = False

		self.fitness_function = fitness_function
		self.default_fitness_args = kwargs

	def run(self, max_iterations: int, crossover_type: CrossoverMethod, csv_filename=None, **kwargs) -> Dict[str, int]:
		"""
		Returns the best solution found after all iterations.

		Complexity: O(n*s + i*n*f + i*e), where n is the population size, s is the solution size, 
			i is the maximum number of iterations, f is the complexity of the fitness function and e is the elite size.

		Args:
			solution_size: Defines the number of iterations to be generate.
			crossover_type: Tells the crossover type to be used.
				- CrossoverMethod.UNIQUE_POINT_CROSSOVER: uses the unique point crossover.
				- CrossoverMethod.MULTI_POINT_CROSSOVER: uses the multi point crossover.
				- CrossoverMethod.UNIFORM_CROSSOVER: uses the uniform crossover.
			**kwargs: The arguments for the given crossover type.
				- CrossoverMethod.UNIQUE_POINT_CROSSOVER:
					• point: An integer in [0, N] representing the point to split the solutions.
				- CrossoverMethod.MULTI_POINT_CROSSOVER:
					• points: A list with integers in [0, N] representing the points to split the solutions.
				- CrossoverMethod.UNIFORM_CROSSOVER:
					• chances: A list with floats in [0, 1] representing the chances to a point in the solution be swapped.
					• threshold: A float in [0, 1] representing the number which a specific chance must be greater to not be swapped.

		Returns:
			solution: The best found solution.
		"""
		csv_file = open(csv_filename, 'w', newline='')
		csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(['best', 'mean'])
		population = first_generation(self.population_size, self.solution_size, 0, self.initial_solution)

		for iteration in range(max_iterations):
			print(f'Iteration: {iteration}')
			values = []

			for solution in population:
				values.append((self.fitness_function(solution, **self.default_fitness_args), solution))

			elite_with_values = sorted(values, key=lambda x: x[0], reverse=self.order)
			elite = [x for _, x in elite_with_values]
			print(f'Best value: {elite_with_values[0][0]}')
			print(f'Mean: {sum([x for x, _ in elite_with_values])/float(self.population_size)}')
			csv_writer.writerow([elite_with_values[0][0], sum([x for x, _ in elite_with_values])/self.population_size])

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

		return population[0]

	def crossover(self, solution1: Dict[str, int], solution2: Dict[str, int], crossover_type: int, **kwargs) -> Tuple[Dict[str, int], Dict[str, int]]:
		"""
		Returns two solutions generated based on two parents solutions.

		Complexity: O(n^2)

		Args:
			solution1: First parent solution.
			solution2: Second parent solution.
			crossover_type: Tells the crossover type to be used.
				- CrossoverMethod.UNIQUE_POINT_CROSSOVER: uses the unique point crossover.
				- CrossoverMethod.MULTI_POINT_CROSSOVER: uses the multi point crossover.
				- CrossoverMethod.UNIFORM_CROSSOVER: uses the uniform crossover.
			**kwargs: The arguments for the given crossover type.
				- CrossoverMethod.UNIQUE_POINT_CROSSOVER:
					• point: An integer in [0, N] representing the point to split the solutions.
				- CrossoverMethod.MULTI_POINT_CROSSOVER:
					• points: A list with integers in [0, N] representing the points to split the solutions.
				- CrossoverMethod.UNIFORM_CROSSOVER:
					• chances: A list with floats in [0, 1] representing the chances to a point in the solution be swapped.
					• threshold: A float in [0, 1] representing the number which a specific chance must be greater to not be swapped.

		Returns:
			new_solution1, new_solution2: The two new solutions generated.
		"""
		solution1_as_list = list(solution1.items())
		solution2_as_list = list(solution2.items())

		if crossover_type == CrossoverMethod.UNIQUE_POINT_CROSSOVER:
			try:
				point = kwargs['point']
			except KeyError:
				point = len(solution1_as_list) // 2

			solution1_as_list, solution2_as_list = Crossover.single_point_crossover(solution1_as_list, solution2_as_list, point)
			return dict(solution1_as_list), dict(solution2_as_list)
		elif crossover_type == CrossoverMethod.MULTI_POINT_CROSSOVER:
			try:
				points = kwargs['points']
			except KeyError:
				points = [len(solution1_as_list) // 2]

			solution1_as_list, solution2_as_list = Crossover.multi_point_crossover(solution1_as_list, solution2_as_list, points)
			return dict(solution1_as_list), dict(solution2_as_list)
		elif crossover_type == CrossoverMethod.UNIFORM_CROSSOVER:
			try:
				chances = kwargs['chances']
				threshold = kwargs['threshold']
			except KeyError:
				threshold = 0.5
				chances = [random.uniform(0, 1) for i in range(len(solution1_as_list))]

			solution1_as_list, solution2_as_list = Crossover.uniform_crossover(solution1_as_list, solution2_as_list, chances, threshold)
			return dict(solution1_as_list), dict(solution2_as_list)
		
		raise Exception('Crossover type not implemented')

	def mutation(self, solution: Dict[str, int], mutation_chance: float=None, seed: int=None):
		"""
		Returns a solution generated based on the given solution with some changes in its values.

		Complexity: O(n) usually, but in a specific case it doesn't stop (almost 0 chance).

		Args:
			solution: Solution to be used as base.
			mutation_chance: Chance to change value for each key in the solution.
			seed: The seed to be used in random numbers generation.

		Returns:
			new_solution: A solution with a few changes based on mutation chance.
		"""
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
	"""
	Returns a population of solutions generated based on the given solution with some changes in its values.
	If base solution is None, then it will be set to have one color for vertex (fitness value is equal than the number of vertices).

	Complexity: O(n*v) where n is the population size and v is the number of vertices.

	Args:
		population_size: The number of solutions to be generated.
		vertices_number: The number of vertices of the graph.
		change_percentual: How much a value will be changed for every key in the base solution.
		base_solution: Solution to be used as base.

	Returns:
		population: A population with a few changes based on the base solution.
	"""
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

class GeneticAlgorithm2:
	def __init__(self, population_size: int, rng_seed: int, graph):

		self.graph = graph
		self.num_vertices = len(graph.keys())
		self.max_colors = 1

		if population_size > 0:
			self.population_size = population_size	
			self.elite_size = population_size // 2
		else: 
			raise Exception('Population size must be greater than 0.')

		self.seed = rng_seed
		random.seed(self.seed)
		np.random.seed(self.seed)

		self.random1 = np.random.RandomState(rng_seed)
		self.random2 = np.random.RandomState(rng_seed * 2 + rng_seed // 2)

	def run(self, max_iterations: int, csv_filename: str = 'csv_file') -> Dict[str, int]:
		"""
		Returns the best solution found after all iterations.
		"""
		csv_file = open(csv_filename, 'w', newline='')
		csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(['best', 'mean'])
		population = self.gen_first_generation(self.population_size, self.num_vertices)
		only_valid = False
		best_so_far = self.num_vertices

		for iteration in range(max_iterations):
			print(f'Iteration: {iteration}')

			population_fitness = []
			sorted_fitness = []

			for p in population:
				fitness = self.calc_fitness(p)
				population_fitness.append([fitness, p])

			new_population = []
			# ascending sort by fitness value and number of colors used
			sorted_fitness = sorted(population_fitness, key = lambda x: (x[0], len(set(x[1]))))

			csv_writer.writerow([sorted_fitness[0][0], sum([x for x, _ in sorted_fitness])/self.population_size])
			# print('Best so far = {}'.format(sorted_fitness[0]))

			best_individuals = [sorted_fitness[i][1] for i in range(self.elite_size)]

			new_population = list(best_individuals)

			if(sorted_fitness[0][0] >= 4 and iteration % 50 == 0 ):
				# if(sorted_fitness[0][0] >= 4):
				self.max_colors += 1
				print(self.max_colors)
				print(sorted_fitness[0][0])
			# if(sorted_fitness[0][0] == 0):
			# 	print("solucao valida")
			# 	print(sorted_fitness[0][1])
			
			parent1, parent2 = self.parentSelection1(new_population)
			child1, child2 = self.crossover(parent1, parent2)
			child1 = self.mutation1(child1)
			child2 = self.mutation1(child2)

			new_population.append(child1)
			new_population.append(child2)
			# else:
			# 		parent1, parent2 = self.parentSelection2(new_population)
			# 		child1, child2 = self.crossover(parent1, parent2)
			# 		child1 = self.mutation2(child1)
			# 		child2 = self.mutation2(child2)
			# 		new_population.append(child1)
			# 		new_population.append(child2)

			new_population.extend(self.gen_first_generation(self.elite_size, self.num_vertices))
			population = list(new_population)

		print(sorted_fitness[0][0])
		print(len(set(sorted_fitness[0][1])))

		return len(set(sorted_fitness[0][1])), sorted_fitness[0][1]

	def parentSelection1(self, population):
		parent1Temp_index, parent2Temp_index = self.random1.choice(len(population), 2)
		# print("CHOICES 1 = {} {}".format(parent1Temp_index, parent2Temp_index))
		# shuffled_pop = random.sample(population, len(population))
		parent1Temp = population[parent1Temp_index]
		parent2Temp = population[parent2Temp_index]
		parent1_fitness = self.calc_fitness(parent1Temp)
		parent2_fitness = self.calc_fitness(parent2Temp)

		parent1 = parent1Temp if parent1_fitness > parent2_fitness else parent2Temp

		parent1Temp_index, parent2Temp_index = self.random2.choice(len(population), 2)
		# print("CHOICES 2 = {} {}".format(parent1Temp_index, parent2Temp_index))
		# shuffled_pop = random.sample(population, len(population))
		parent1Temp = population[parent1Temp_index]
		parent2Temp = population[parent2Temp_index]
		parent1_fitness = self.calc_fitness(parent1Temp)
		parent2_fitness = self.calc_fitness(parent2Temp)
		
		parent2 = parent1Temp if parent1_fitness > parent2_fitness else parent2Temp
		
		return parent1, parent2
		
	def parentSelection2(self, population):
		best_individual = population[0]

		return best_individual, best_individual

	def crossover(self, parent1, parent2):
		cross_point = self.random1.randint(0, len(parent1))
		# print("crosspoint = {}".format(cross_point))
		child1 = []
		child2 = []
		child1.extend(parent1[:cross_point])
		child1.extend(parent2[cross_point:])
		child2.extend(parent2[:cross_point])
		child2.extend(parent1[cross_point:])
		
		return child1, child2

	def mutation1(self, individual):
		individual_copy = list(individual)
		all_colors = set(individual_copy)
		for vertex in range(len(individual_copy)):
			all_colors = set(individual_copy)
			adjacent_colors = set()
			for neighbohr in self.graph[str(vertex)]:
				adjacent_colors.add(individual_copy[int(neighbohr)])
			for neighbohr in self.graph[str(vertex)]:
				if(individual_copy[vertex] == individual_copy[int(neighbohr)]):
					valid_colors = all_colors.difference(adjacent_colors)
					if(len(valid_colors) == 0):
						return individual_copy
					colors_list = list(valid_colors)
					new_color_index = self.random1.choice(len(colors_list))
					new_color = colors_list[new_color_index]
					individual_copy[vertex] = new_color
		return individual_copy

	def mutation2(self, individual):
		all_colors = set(individual)
		for vertex in range(len(individual)):
			adjacent_colors = set()
			for neighbohr in self.graph[str(vertex)]:
				adjacent_colors.add(individual[int(neighbohr)])
			for neighbohr in self.graph[str(vertex)]:
				if(individual[vertex] == individual[int(neighbohr)]):
					colors_list = list(all_colors)
					new_color_index = self.random1.choice(len(colors_list))
					new_color = colors_list[new_color_index]
					individual[vertex] = new_color
		return individual
		
	def calc_fitness(self, individual):
		broken_restrictions = 0
		for vertex in range(len(individual)):
			individual_color = individual[vertex]
			for neighbohr in self.graph[str(vertex)]:
				if(individual_color == individual[int(neighbohr)]):
					broken_restrictions += 1
		# number_of_colors = len(set(individual))

		return broken_restrictions // 2

	def gen_first_generation(self, population_size, vertices_number):
		"""
		Returns a population of solutions.

		Complexity: O(n*v) where n is the population size and v is the number of vertices.

		Args:
			population_size: The number of solutions to be generated.
			vertices_number: The number of vertices of the graph.
		Returns:
			population: A random population with the specified size.
		"""
		population = []

		for _ in range(population_size):
			solution = [self.random1.randint(0, self.max_colors) for _ in range(vertices_number)]
			population.append(solution)

		return population