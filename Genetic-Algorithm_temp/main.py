import numpy
import re
import json
import random
from client import *

# Inputs of the equation.
# equation_inputs = [4,-2,3.5,5,-11,-4.7]
initial_inputs = []

# open file and read the content in a list
################## use JSON here ########################
with open('./overfit.txt','r') as overfit:
    line = overfit.read()
    tmp = re.split(', |\[|\]|\n', line)
# print("Inside try 2")
for i in tmp:
    if i != '':
        initial_inputs.append(float(i))


pop = [[ 0.00000000e+00, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11,
  -1.75214813e-10, -1.78647510e-15,  8.37487117e-16,  2.17040738e-05,
  -2.23105544e-06, -1.59792834e-08,  9.98214034e-10],
 [-2.04721003e-06, -1.14596759e-12, -2.66817632e-13,  4.94743930e-11,
  -1.54610976e-10, -2.25444384e-15,  1.02537689e-15,  2.73406628e-05,
  -2.34845940e-06, -1.45238413e-08,  8.52802022e-10],
 [ 0.00000000e+00, -1.45799022e-12, -2.28980078e-13 , 4.62010753e-11,
  -1.75214813e-10, -1.78647510e-15,  8.37487117e-16 , 2.17040738e-09,
  -2.23105544e-06, -1.59792834e-08,  9.98214034e-10],
 [-2.04721003e-06, -1.14596759e-12, -2.66817632e-13,  4.94743930e-19,
  -1.54610976e-10, -2.25444384e-15,  1.02537689e-15,  2.73406628e-05,
  -2.34845940e-06, -1.45238413e-08,  8.52802022e-10]]

fitness = [9.574044e+12, 2.591813e+13, 3.574044e+13, 2.545344e+13]
# , 3.574044e+12, 2.545344e+13, 3.574044e+12, 2.591813e+13, 3.574044e+12, 2.545344e+13]
#  [ 0.00000000e+00 -1.45799022e-12 -2.28980078e-13  4.62010753e-11
#   -1.75214813e-10 -1.78647510e-15  8.37487117e-16  2.17040738e-05
#   -2.23105544e-06 -1.59792834e-08  9.98214034e-10]
#  [-2.04721003e-06 -1.14596759e-12 -2.66817632e-13  4.94743930e-11
#   -1.54610976e-10 -2.25444384e-15  1.02537689e-15  2.73406628e-05
#   -2.34845940e-06 -1.45238413e-08  8.52802022e-10]
#  [ 0.00000000e+00 -1.36296126e-12 -2.28980078e-13  4.62010753e-11
#   -1.75214813e-10 -1.78647510e-15  8.37487117e-16  2.17040738e-05
#   -2.23105544e-06 -1.59792834e-08  9.98214034e-10]
#  [-2.04721003e-06 -1.14596759e-12 -2.66817632e-13  4.94743930e-11
#   -1.54610976e-10 -2.25444384e-15  1.02537689e-15  2.73406628e-05
#   -2.34845940e-06 -1.45238413e-08  8.52802022e-10]
#  [ 0.00000000e+00 -1.45799022e-12 -2.28980078e-13  4.62010753e-11
#   -1.75214813e-10 -1.78647510e-15  8.37487117e-16  2.17040738e-05
#   -2.23105544e-06 -1.54782117e-08  9.98214034e-10]
#  [-2.04721003e-06 -1.14596759e-12 -2.66817632e-13  4.94743930e-11
#   -1.54610976e-10 -2.25444384e-15  1.02537689e-15  2.73406628e-05
#   -2.34845940e-06 -1.45238413e-08  8.52802022e-10]]


# parents = numpy.array(parents)
# offspring = numpy.empty(parents.shape)
# n = offspring.shape
# i=0
# while i < n:
#     # prob = p
#     # coeff = []
#     # for idx in range(i, i+num_parents_mating):
#     #     coeff.append(prob)
#     #     prob *= (1-p)
#     num = random.sample(range(1,11),2-1)
#     num.sort()
#     num.append(11)
#     print(num,",type = ",type(num))
#     for idx in range(i, i+2):
#         # for j in range(0,offspring.shape[1]):
#         offspring[idx][0:num[0]] = parents[idx][0:num[0]]
#         for k in range(0,len(num)-1):
#             offspring[idx][num[k]:num[k+1]] = parents[i+(idx+k+1)%2][num[k]:num[k+1]]
#             # for c in range(0,len(coeff)):
#             #     offspring[idx][j] = offspring[idx][j] + coeff[c]*parents[(idx+c)%num_parents_mating][j]
            
#     i = i + 2
# print("Parents = ", parents,"\n\n")
# print("Offspring = ",offspring)


# def select_parents(pop, fitness):
#     # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
#     # dividing fitness values into probability ranges
#     total = 0
#     for e in fitness:
#         total = total + e
#     percent = []
#     for e in fitness:
#         percent.append(1 - e/total)
#     # percent = 100 - percent
#     total = 0
#     for e in percent:
#         total = total + e
#     # roulette wheel for 100-percent values
#     roulette = [0]
#     val = 0
#     for e in percent:
#         # cumulative sum of fitness
#         val = val + e
#         roulette.append(val/total)
#     print("Roulette = ",roulette)
#     # selecting parents according to value of a random number
#     # print("select_parents's pop ",pop)
#     # parents = numpy.empty(pop.shape)
#     parents = []
#     # print("parents1 : ",parents)
#     for p in pop:
#         num = numpy.random.uniform(0,1)
#         print()
#         print("num = ",num)
#         id = 1
#         while id < len(roulette) and (roulette[id] - num) < 1e-20:
#             id = id + 1
#         l = pop[id-1]
#         print("p = ",l)
#         parents.append(l)
#     parents = numpy.array(parents)
#     print("parents2 : ",parents)
#     return parents

# pop = numpy.array(pop)
# parent = select_parents(pop,fitness)
# print("Parents = ",parent)


def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        for j in range(offspring_crossover.shape[1]):
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            if(random_value > -0.25 and random_value < 0.25 ): # question do we have to do mutation for every generation
                # flag2 = numpy.random.randint(0,10)
                # flag1 = numpy.random.randint(0,10)
                # mut = numpy.random.uniform(0.05,0.15)
                mut = numpy.random.uniform(1.5,2)
                s = numpy.random.choice([-1,1])
                offspring_crossover[idx, j] = offspring_crossover[idx, j]*(1+s*mut)
            # temp  = offspring_crossover[idx, flag1]
            # offspring_crossover[idx, flag1] = offspring_crossover[idx, flag2]
        
        # if(random_value > -0.4 and random_value < 0.4 ): # question do we have to do mutation for every generation
        #     # flag2 = numpy.random.randint(0,10)
        #     flag1 = numpy.random.randint(0,10)
        #     s = numpy.random.choice([-1,1])
        #     offspring_crossover[idx, flag1] = offspring_crossover[idx, flag1]*(1+s*mut)
        #     # temp  = offspring_crossover[idx, flag1]
        #     # offspring_crossover[idx, flag1] = offspring_crossover[idx, flag2]
        #     # offspring_crossover[idx, flag2] = temp 
    return offspring_crossover

print("Parents = ", pop,"\n\n")
offspring = numpy.array(pop)
offspring = mutation(offspring)
print("Offspring = ",offspring)

# error = get_errors(SECRET_KEY, list(initial_inputs))
# print(error)

# except:
# print("not in try")
# print(equation_inputs)
# exit()
# for i in tmp:
#     if i != '':
#         equation_inputs.append(float(i)) 
# # numpy.resize(equation_inputs,(11,0))

# # Number of the weights we are looking to optimize.
# num_weights = 11

# """
# Genetic algorithm parameters:
#     Mating pool size
#     Population size
# """
# sol_per_pop = 8
# num_parents_mating = 10

# # Defining the population size.
# pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
# #Creating the initial population.
# new_population = numpy.random.uniform(low=-10.0, high=10.0, size=pop_size)
# # print(new_population)

# def cal_pop_fitness(equation_inputs):
#     # Calculating the fitness value of each solution in the current population.
#     # The fitness function caulcuates the sum of products between each input and its corresponding weight.
#     fitness = get_errors(SECRET_KEY, equation_inputs)
#     return fitness[0]

# fitness = cal_pop_fitness(equation_inputs)
# print(fitness)
# print(equation_inputs)
