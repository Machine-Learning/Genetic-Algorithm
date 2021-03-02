import numpy
import re
import json
import math
from client import *

num_weights = 11
sol_per_pop = 10
num_parents_mating = 2
const_addition = 1e-18
total_api_calls = 50
pop_size = (sol_per_pop,num_weights) 

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
            line = overfit.read()
            tmp = re.split(', |\[|\]|\n', line)
        for i in tmp:
            if i != '':
                initial_inputs.append(float(i))
        new_population = []
        rng = 1.0
        for i in range(sol_per_pop):
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
        new_population = numpy.random.uniform(low=-1e-16, high=1e-7, size=pop_size)

def cal_pop_fitness(pop):
    fitness = []
    i = 1
    for p in pop:
        fitness.append(get_errors(SECRET_KEY, list(p)))
    return fitness

def select_parents(pop, fitness):
    total = 0
    for e in fitness:
        total = total + e[0]
    roulette = [0]
    val = 0
    for e in fitness:
        val = val + e[0]
        roulette.append(val/total)
    parents = pop
    for p in parents:
        num = numpy.random.uniform(0,1)
        id = 1
        while id < len(roulette) and (roulette[id] - num) < 1e-20:
            id = id + 1
        p = pop[id-1]
    return parents

def crossover(parents, num_parents_mating,fitness):
    offspring = numpy.empty(parents.shape)
    n = offspring.shape[0]
    i = 0
    while i < n:
        total = 0
        for idx in range(i, i+num_parents_mating):
            total = total + fitness[idx][0]
        coeff = []
        for idx in range(i, i+num_parents_mating):
            coeff.append(fitness[idx][0]/total)
        for idx in range(i, i+num_parents_mating):
            for j in range(0,offspring.shape[1]):
                offspring[idx][j] = 0.0
                for c in range(0,len(coeff)):
                    offspring[idx][j] = offspring[idx][j] + coeff[c]*parents[(idx+c)%n][j]
        i = i + num_parents_mating
    return offspring

def mutation(offspring_crossover):
    for idx in range(offspring_crossover.shape[0]):
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        if(random_value > -0.4 and random_value < 0.4 ):
            flag1 = numpy.random.randint(0,10)
            mut = numpy.random.uniform(-0.1,0.1)
            offspring_crossover[idx, flag1] = offspring_crossover[idx, flag1]*(1+mut)
    return offspring_crossover

num_generations = total_api_calls//sol_per_pop

for generation in range(num_generations):
    print("Generation : ", generation)
    print('New population: ',new_population,end='\n')
    fitness = cal_pop_fitness(new_population)
    var_fitness = [[fitness[x][y] for y in range(len(fitness[0]))] for x in range(len(fitness))]
    for f in var_fitness:
        f[0] = "{:e}".format(f[0])
        f[1] = "{:e}".format(f[1])
    print('Fitness: ',var_fitness,end='\n\n')
    parents = select_parents(new_population, fitness)
    offspring_crossover = crossover(parents,num_parents_mating,fitness)
    print("offspring_crossover : ",offspring_crossover)
    offspring_mutation = mutation(offspring_crossover)
    new_population = offspring_mutation

answer = json.dumps(new_population.tolist())
with open('./output.txt','w+') as write_file:
    json.dump(answer, write_file)
print('************************************')
