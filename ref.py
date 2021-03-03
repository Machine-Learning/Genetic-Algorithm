import numpy
import re
import json
import math
import random
from client import *

# Inputs of the equation.
# equation_inputs = [4,-2,3.5,5,-11,-4.7]

# Number of the weights we are looking to optimize.
num_weights = 11

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 10
num_parents_mating = 2
total_api_calls = 140
train_data_wieght = 0.4
p = 0.8

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

try:
    with open('./output.txt','r') as prev:
        old_generation = json.load(prev)
        old_generation = json.loads(old_generation)
        data = list(old_generation)
        new_population = numpy.array(data)
    # print("inside try 1 ")
except:
    initial_inputs = []
    # print('Going to except******************************')
    # open file and read the content in a list
    try:
        with open('./overfit.txt','r') as overfit:
            line = overfit.read()
            tmp = re.split(', |\[|\]|\n', line)
        # print("Inside try 2")
        for i in tmp:
            if i != '':
                initial_inputs.append(float(i))
        # print(initial_inputs,'**********')
                
        new_population = []
        # new_population = numpy.random.uniform(low=-10.0, high=10.0, size=pop_size)
        new_population.append(initial_inputs)
        rng = 0.25
        const_addition = min(initial_inputs)
        for i in range(sol_per_pop-1):
            ls = []
            for vec in initial_inputs:
                num = numpy.random.uniform(-rng,rng)
                num = vec*(1.01+num)
                if(not num):
                    num += const_addition
                ls.append(num)
            new_population.append(ls)
        new_population = numpy.array(new_population)
        # print(new_population,' type:  ',type(new_population))
        
    except:
        #Creating the initial population.
        # print("inside last except ")
        new_population = numpy.random.uniform(low=-10.0, high=10.0, size=pop_size)
    

# print(new_population)

def fitness_function(fitness):
    # power = 5
    # if fitness[0][0]>=1:
    #     power = 1/power
    for e in fitness:
        e[0] = train_data_wieght*e[0] + (1-train_data_wieght)*e[1]
    print("Fitness = ",fitness)
    return fitness

def cal_pop_fitness(pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = []
    i = 1
    for p in pop:
        # print('Vector: ',list(p))
        fitness.append(get_errors(SECRET_KEY, list(p)))
        # fitness.append([i,i])

    return fitness

def select_parents(pop, fitness):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    # dividing fitness values into probability ranges
    total = 0
    for e in fitness:
        total = total + e[0]
    percent = []
    for e in fitness:
        percent.append(1 - e[0]/total)
    # percent = 100 - percent
    total = 0
    for e in percent:
        total = total + e
    # roulette wheel for 100-percent values
    roulette = [0]
    val = 0
    for e in percent:
        # cumulative sum of fitness
        val = val + e
        roulette.append(val/total)
    
    # selecting parents according to value of a random number
    # print("select_parents's pop ",pop)
    # parents = numpy.empty(pop.shape)
    parents = []
    # print("parents1 : ",parents)
    for p in pop:
        num = numpy.random.uniform(0,1)
        id = 1
        while id < len(roulette) and (roulette[id] - num) < 1e-20:
            id = id + 1
        val = pop[id-1]
        parents.append(val)
    parents = numpy.array(parents)
    return parents

def crossover(parents, num_parents_mating,fitness):
    offspring = numpy.empty(parents.shape)
    n = offspring.shape[0]
    i = 0
    # uses Whole Arithmatic Recombination
    while i < n:
        # prob = p
        # coeff = []
        # for idx in range(i, i+num_parents_mating):
        #     coeff.append(prob)
        #     prob *= (1-p)
            # for j in range(0,offspring.shape[1]):
        num = random.sample(range(1,offspring.shape[1]),num_parents_mating-1)
        num.sort()
        num.append(offspring.shape[1])
        for idx in range(i, i+num_parents_mating):
            offspring[idx][0:num[0]] = parents[idx][0:num[0]]
            for k in range(0,len(num)-1):
                offspring[idx][num[k]:num[k+1]] = parents[i+(idx+k+1)%num_parents_mating][num[k]:num[k+1]]
                
        i = i + num_parents_mating
    
    return offspring

def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        for j in range(offspring_crossover.shape[1]):
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            if(random_value > -0.2 and random_value < 0.2 ): # question do we have to do mutation for every generation
                # flag2 = numpy.random.randint(0,10)
                # flag1 = numpy.random.randint(0,10)
                mut = numpy.random.uniform(0.05,0.15)
                s = numpy.random.choice([-1,1])
                offspring_crossover[idx, j] = offspring_crossover[idx, j]*(1+s*mut)
            # temp  = offspring_crossover[idx, flag1]
            # offspring_crossover[idx, flag1] = offspring_crossover[idx, flag2]
            # offspring_crossover[idx, flag2] = temp 
    return offspring_crossover

num_generations = total_api_calls//sol_per_pop
for generation in range(num_generations):
    print("Generation : ", generation)
    # Measing the fitness of each chromosome in the population.
    print('New population: ',new_population,end='\n')
    fitness = cal_pop_fitness(new_population)
    var_fitness = [[fitness[x][y] for y in range(len(fitness[0]))] for x in range(len(fitness))]
    for f in var_fitness:
        f[0] = "{:e}".format(f[0])
        f[1] = "{:e}".format(f[1])
    print('Errors: ',var_fitness,end='\n\n')

    # average fitness
    fitness = fitness_function(fitness)
    # Selecting the best parents in the population for mating.
    parents = select_parents(new_population, fitness)

    # Generating next generation using crossover.
    offspring_crossover = crossover(parents,num_parents_mating,fitness)
    # print("offspring_crossover : ",offspring_crossover)

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = mutation(offspring_crossover)

    # Creating the new population based on the parents and offspring.
    # new_population[0:parents.shape[0], :] = parents
    new_population = offspring_mutation

    # The best result in the current iteration.
    # print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))

answer = json.dumps(new_population.tolist())
with open('./output.txt','w+') as write_file:
    json.dump(answer, write_file)
print('************************************')
# # Getting the best solution after iterating finishing all generations.
# #At first, the fitness is calculated for each solution in the final generation.
# fitness = cal_pop_fitness(equation_inputs, new_population)
# # Then return the index of that solution corresponding to the best fitness.
# best_match_idx = numpy.where(fitness == numpy.max(fitness))

# print("Best solution : ", new_population[best_match_idx, :])
# print("Best solution fitness : ", fitness[best_match_idx])
