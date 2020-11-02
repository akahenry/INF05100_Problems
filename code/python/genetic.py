# -*- coding: utf-8 -*-

import random
import copy
import numpy as np
from pprint import pprint


def BFS:
  # Mark all the vertices as not visited 
  visited = [False] * (len(self.graph)) 

  # Create a queue for BFS 
  queue = [] 

  # Mark the source node as  
  # visited and enqueue it 
  queue.append(s) 
  visited[s] = True

  while queue: 

      # Dequeue a vertex from  
      # queue and print it 
      s = queue.pop(0) 
      print (s, end = " ") 

      # Get all adjacent vertices of the 
      # dequeued vertex s. If a adjacent 
      # has not been visited, then mark it 
      # visited and enqueue it 
      for i in self.graph[s]: 
          if visited[i] == False: 
              queue.append(i) 
              visited[i] = True

def genetic(graph, best_n, mutation_chance, random_seed, initial_pop_size, max_generations=20, num_vertices, initial_solution):
  
  np.random.seed(random_seed)
  
  mutation_chance_increase = 0

  initial_population = generate_first_generation(random_seed, num_vertices, initial_solution, pop_size)
  # print("INITIAL POPULATION = {}".format(initial_population))

  population_best = []
  parents = []
  parents_fitness = []
  past_generation_best = [0]

  get_individual = lambda a : [b[1] for b in a]

  for g in range(max_generations):
    print("G = {}".format(g))
    if(g == 0):
      parents = initial_population[:]
      for p in parents:
        # calculates individual value
        p_fitness = calculate_fitness(p, graph)
        # print("p = {} => {}".format(p, p_fitness))
        parents_fitness.append([p_fitness, p])
      # print("FIRST GEN FITNESS")
      # print(parents_fitness)
      print("FIRST_GEN_AVERAGE = {}".format(sum([f[0] for f in parents_fitness])/len(parents_fitness)))
      print("FIRST_GEN_BEST = {}".format(sorted(parents_fitness, key=lambda x: x[0], reverse=True)[0]))
      parents = get_individual(parents_fitness)
    else:
      # print(population_best)
      print("PAST_GEN_AVERAGE = {}".format(sum([f[0] for f in population_best])/len(population_best)))
      print("PAST GENERATION BEST = {}".format(population_best[0]))
      past_generation_best = population_best[0]
      parents_fitness = population_best[:best_n]
      # print("BEST FROM COPY = {}".format(parents_fitness[0]))
      parents = get_individual(parents_fitness)[:best_n]
      # print("BEST FROM PARENTS = {}".format(parents[0]))

    # 2 children per couple

    children = []
    parents_indexes = [x for x in range(len(parents))]
    # print(parents_indexes)
    for i in range(len(parents) // 2):
      parent1, parent2 = random.sample(parents, 2)
      
      first_child = []
      
      for gene_index in range(len(parents[0])):
        if(gene_index % 2 == 0):
          first_child.append(parent1[gene_index])
        else:
          first_child.append(parent2[gene_index])

      
      will_mutate = random.random() <= mutation_chance + mutation_chance_increase
      if(will_mutate):
        mutation_index = random.randrange(len(first_child))
        first_child[mutation_index] = random.uniform(-0.3, 0.3) * first_child[mutation_index]

      second_child = []
      
      for w in range(len(parents[0])):
        if(w % 2 == 0):
          second_child.append(parent2[w])
        else:
          second_child.append(parent1[w])
      
      will_mutate = random.random() <= mutation_chance + mutation_chance_increase
      if(will_mutate):
        mutation_index = random.randrange(len(second_child))
        second_child[mutation_index] = random.random() * second_child[mutation_index]

      children.append(first_child)
      children.append(second_child)

    children_fitness = []
    
    for c in children:
      # calculates individual value
      c_fitness = calculate_fitness(c, graph)
      children_fitness.append([c_fitness, c])
    
    total_population = parents_fitness + children_fitness

    population_best = sorted(total_population,key = lambda x: x[0], reverse=True)[:best_n]
    
    if(population_best[0][0] <= past_generation_best[0] and mutation_chance + mutation_chance_increase < 0.20):
      print("increasing mutation_chance by 1%")
      mutation_chance_increase += 0.01
      print("new mutation_chance_increase = {}".format(mutation_chance_increase))
    elif(population_best[0][0] > past_generation_best[0] and mutation_chance_increase > 0):
      mutation_chance_increase = 0

  print("LAST AVERAGE = {}".format(sum([f[0] for f in population_best])/len(population_best)))
  print("LAST BEST = {}".format(population_best[0]))
  print('BEST {}'.format(population_best[0]))
  return population_best[0][1]

def generate_first_generation(random_seed, num_vertices, initial_solution, pop_size):
  np.random.seed(random_seed)

  population = []
  for i in range(pop_size):
    disturbed_values = []
    for s in initial_solution:
      disturbed_values.append((s + random.randint(0, num_vertices)) % num_vertices)

    population.append(disturbed_values)

    # sobrescreve o primeiro valor gerado com os valores originais
    population[0] = initial_solution[:]

  return population

def calculate_fitness(solution, graph):
  
  # a solution is expected to be in the form
  # {
  #   node: color
  # }
  
  # count the number of colors and the number of broken restrictions
  
  # count broken restrictions
  broken_restrictions = 0
  respect_restrictions = 0
  
  for v in graph:
    solution_color = solution[v]
    for adjacent_node in graph[v]:
      if(solution_color == solution[adjacent_node]):
        broken_restrictions += 1
      else:
        respect_restrictions += 1
  
  # number of colors in solution
  number_of_colors = len(set(solution.values()))
  
  return [number_of_colors, respect_restrictions - broken_restrictions]