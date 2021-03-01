import numpy
import re
import json
from client import *

# Inputs of the equation.
# equation_inputs = [4,-2,3.5,5,-11,-4.7]
equation_inputs = []

# open file and read the content in a list
################## use JSON here ########################
with open('./output.txt','r') as prev:
    old_generation = json.load(prev)
    old_generation = json.loads(old_generation)
    data = list(old_generation)
    equation_inputs = numpy.array(data)
    print("inside try 1 ")
# except:
# print("not in try")
print(equation_inputs)
exit()
for i in tmp:
    if i != '':
        equation_inputs.append(float(i)) 
# numpy.resize(equation_inputs,(11,0))

# Number of the weights we are looking to optimize.
num_weights = 11

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 8
num_parents_mating = 10

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
new_population = numpy.random.uniform(low=-10.0, high=10.0, size=pop_size)
# print(new_population)

def cal_pop_fitness(equation_inputs):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = get_errors(SECRET_KEY, equation_inputs)
    return fitness[0]

fitness = cal_pop_fitness(equation_inputs)
print(fitness)
# print(equation_inputs)
