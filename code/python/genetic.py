import random
import copy
import numpy as np
from pprint import pprint

def genetic(best_n, mutation_chance, weights, pop_size, run_episode_fn):
  # indice de corte
  # crossover_index = random.randrange(len(weights))
  # crossover_index = len(weights) // 2
  # print("crossover_index = {}".format(crossover_index))
  
  mutation_chance_increase = 0

  initial_population = generate_first_generation(weights, pop_size)
  # print("INITIAL POPULATION = {}".format(initial_population))
  
  # pprint("initial population")
  # print(weights)
  # for p in initial_population:
  #   print(p)

  population_best = []
  parents = []
  parents_fitness = []
  past_generation_best = [0]

  get_individual = lambda a : [b[1] for b in a]

  # 20 generations
  for g in range(100):
    print("G = {}".format(g))
    if(g == 0):
      parents = initial_population[:]
      for p in parents:
        # calculates individual value
        p_fitness = run_episode_fn(p)
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

    # loop gera filhos
    # 2 children per couple

    children = []
    parents_indexes = [x for x in range(len(parents))]
    # print(parents_indexes)
    for i in range(len(parents) // 2):
      # print("i = {}".format(i))

      # parent1 = total_population[random.randrange(len(parents))]
      # parent2 = total_population[random.randrange(len(parents))]

      # parent1, parent2 = random.sample(parents, 2)
      parent1_index = np.random.choice(parents_indexes, size=None, replace=False,p=None)
      parents_indexes.remove(parent1_index)
      parent2_index = np.random.choice(parents_indexes, size=None, replace=False,p=None)
      parents_indexes.remove(parent2_index)
      parent1 = parents[parent1_index]
      parent2 = parents[parent2_index]
      # print("{}\n{}".format(parent1_index, parent2_index))
      
      
      # if(i == 0):
      # print("p1 = {}".format(parent1))
      # print("p2 = {}".format(parent2))

      # parent1_first_half_genes = parent1[:crossover_index]
      # parent1_second_half_genes = parent1[crossover_index:]
      # parent2_first_half_genes = parent2[:crossover_index]
      # parent2_second_half_genes = parent2[crossover_index:]

      first_child = []
      # first_child.extend(parent1_first_half_genes)
      # first_child.extend(parent2_second_half_genes)
      for w in range(len(parents[0])):
        if(w % 2 == 0):
          first_child.append(parent1[w])
        else:
          first_child.append(parent2[w])

      # if(i==0):
      #   print("first_child = {}".format(first_child))

      will_mutate = random.random() <= mutation_chance + mutation_chance_increase
      if(will_mutate):
        # print('MUTATE FIRST CHILD')
        mutation_index = random.randrange(len(first_child))
        first_child[mutation_index] = random.uniform(-0.3, 0.3) * first_child[mutation_index]

      second_child = []
      # second_child.extend(parent2_first_half_genes)
      # second_child.extend(parent1_second_half_genes)
      for w in range(len(parents[0])):
        if(w % 2 == 0):
          second_child.append(parent2[w])
        else:
          second_child.append(parent1[w])
      
      # if(i==0):
      #   print("second_child = {}".format(second_child))

      will_mutate = random.random() <= mutation_chance + mutation_chance_increase
      if(will_mutate):
        # print('MUTATE SECOND CHILD')
        mutation_index = random.randrange(len(second_child))
        second_child[mutation_index] = random.random() * second_child[mutation_index]

      children.append(first_child)
      children.append(second_child)

    children_fitness = []
    
    for c in children:
      # calculates individual value
      c_fitness = run_episode_fn(c)
      # print("p = {} => {}".format(p, p_fitness))
      children_fitness.append([c_fitness, c])
    
    # print("\n\nPAIS = {}\n".format(parents_fitness))
    # print("FILHOS = {}\n\n".format(children_fitness))
    
    total_population = parents_fitness + children_fitness

    # print("TOTAL_POPULATION = {}".format(total_population))

    population_best = sorted(total_population,key = lambda x: x[0], reverse=True)[:best_n]
    # print("population_best SORTED = {}".format(population_best))
    
    # print("current best = {}\nprevious best = {}".format(population_best[0], past_generation_best))


    if(population_best[0][0] <= past_generation_best[0] and mutation_chance + mutation_chance_increase < 0.20):
      # print("current best = {}\nprevious best = {}".format(population_best[0], past_generation_best))
      print("increasing mutation_chance by 1%")
      mutation_chance_increase += 0.01
      print("new mutation_chance_increase = {}".format(mutation_chance_increase))
      # break
    elif(population_best[0][0] > past_generation_best[0] and mutation_chance_increase > 0):
      mutation_chance_increase = 0

    # print("BEST POP: {}".format(population_best))

  print("LAST AVERAGE = {}".format(sum([f[0] for f in population_best])/len(population_best)))
  print("LAST BEST = {}".format(population_best[0]))
  print('BEST {}'.format(population_best[0]))
  return population_best[0][1]

def generate_first_generation(values, pop_size):

  # values é inicialmente random ou vem de arquivo
  # gerar perturbações nesses valores para gerar a população inicial

  population = []
  for i in range(pop_size):
    disturbed_values = []
    for v in values:
      # altera todos os valores da solução em até 30%
      if(-0.0001 <= v <= 0.0001):
        initial_v = v + random.uniform(-1, 1)
      else:
        initial_v = v
      disturbed_values.append(v + initial_v * random.uniform(-0.3, 0.3))

    population.append(disturbed_values)
    
    # sobrescreve o primeiro valor gerado com os valores originais
    # population[0] = np.ndarray.tolist(values)
    population[0] = values[:]

  return population