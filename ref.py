import numpy
import json
import math
import random
from client import *

NUM_WEIGHTS = 11
SOL_PER_POP = 10
NUM_PARENTS_MATING = 2
TRAIN_DATA_WEIGHT = 0.4
MUTATION_PROB = 0.3
MUTATION_CHANGE = 0.3

POP_SIZE = (SOL_PER_POP,NUM_WEIGHTS) 
TOTAL_GEN = 90
TOATL_API_CALLS = 10*TOTAL_GEN
try:
    with open('./output.txt','r') as prev:
        old_generation = json.load(prev)
        old_generation = json.loads(old_generation)
        data = list(old_generation)
        new_population = numpy.array(data)
except:
    initial_inputs = []
    try:
        with open('./overfit.txt','r') as overfit:
            tmp = json.load(overfit)
        for i in tmp:
            if i != '':
                initial_inputs.append(float(i))
        new_population = []
        new_population.append(initial_inputs)
        rng = 0.1
        const_addition = min(initial_inputs) 
        for i in range(SOL_PER_POP-1):
            ls = []
            for vec in initial_inputs:
                num = numpy.random.uniform(-rng,rng)
                num = vec*(1.01+num)
                if(not num):
                    num += const_addition
                ls.append(num)
            new_population.append(ls)
        new_population = numpy.array(new_population)
    except:
        new_population = numpy.random.uniform(low=-10.0, high=10.0, size=POP_SIZE)

def cal_error(pop):
    fitness = []
    i = 1
    for p in pop:
        fitness.append(get_errors(SECRET_KEY, list(p)))
    return fitness

def cal_error_weight(fitness):
    for e in fitness:
        e[0] = TRAIN_DATA_WEIGHT*e[0] + (1-TRAIN_DATA_WEIGHT)*e[1]
    print("Fitness = ",fitness)
    return fitness

def cal_fitness(fitness):
    total = 0
    for e in fitness:
        total = total + e[0]
    fitness_val = []
    # for best results keep EXP between 0.5 and 1
    EXP = 0.85
    for e in fitness:
        fitness_val.append(math.pow(total/e[0],EXP))
    return fitness_val

def select_parents(pop, fitness):
    total = 0
    for e in fitness:
        total = total + e
    roulette = [0]
    val = 0
    for e in fitness:
        val = val + e
        roulette.append(val/total)
    parents = []
    for p in pop:
        num = numpy.random.uniform(0,1)
        id = 1
        while id < len(roulette) and (roulette[id] - num) < 1e-20:
            id = id + 1
        val = pop[id-1]
        parents.append(val)
    parents = numpy.array(parents)
    return parents

def crossover(parents, NUM_PARENTS_MATING):
    offspring = numpy.empty(parents.shape)
    n = offspring.shape[0]
    i = 0
    while i < n:
        num = random.sample(range(1,offspring.shape[1]),NUM_PARENTS_MATING-1)
        num.sort()
        num.append(offspring.shape[1])
        for idx in range(i, i+NUM_PARENTS_MATING):
            offspring[idx][0:num[0]] = parents[idx][0:num[0]]
            for k in range(0,len(num)-1):
                offspring[idx][num[k]:num[k+1]] = parents[i+(idx+k+1)%NUM_PARENTS_MATING][num[k]:num[k+1]]
        i = i + NUM_PARENTS_MATING
    return offspring

def mutation(offspring_crossover):
    for idx in range(offspring_crossover.shape[0]):
        for j in range(offspring_crossover.shape[1]):
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            if(random_value > -MUTATION_PROB and random_value < MUTATION_PROB ):
                mut = numpy.random.uniform(-MUTATION_CHANGE,MUTATION_CHANGE)
                s = numpy.random.choice([-1,1])
                offspring_crossover[idx, j] = offspring_crossover[idx, j]*(1+s*mut)
    return offspring_crossover

NUM_GENERATIONS = TOATL_API_CALLS//SOL_PER_POP
for generation in range(NUM_GENERATIONS):
    print("Generation : ", generation)
    print('New population: ',list(new_population),end='\n')
    
    # FITNESS
    fitness = cal_error(new_population)
    var_fitness = [[fitness[x][y] for y in range(len(fitness[0]))] for x in range(len(fitness))]
    for f in var_fitness:
        f[0] = "{:e}".format(f[0])
        f[1] = "{:e}".format(f[1])
    print('Errors: ',var_fitness,end='\n\n')
    fitness = cal_error_weight(fitness)
    fitness = cal_fitness(fitness)
    
    # SELECTION
    parents = select_parents(new_population, fitness)
    
    # CROSSOVER
    offspring_crossover = crossover(parents,NUM_PARENTS_MATING)
    
    # MUTATION
    offspring_mutation = mutation(offspring_crossover)
    new_population = offspring_mutation

answer = json.dumps(new_population.tolist())
with open('./output.txt','w+') as write_file:
    json.dump(answer, write_file)
print('************************************')
# end here
