import numpy
import json
import math
import random
from client import *

num_weights = 11
sol_per_pop = 10
num_parents_mating = 2
total_api_calls = 20
train_data_wieght = 0.4
p = 0.8
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
            tmp = json.load(overfit)
        for i in tmp:
            if i != '':
                initial_inputs.append(float(i))
        new_population = []
        new_population.append(initial_inputs)
        rng = 0.1
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
    except:
        new_population = numpy.random.uniform(low=-10.0, high=10.0, size=pop_size)

def fitness_function(fitness):
    for e in fitness:
        e[0] = train_data_wieght*e[0] + (1-train_data_wieght)*e[1]
    print("Fitness = ",fitness)
    return fitness

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
    percent = []
    for e in fitness:
        percent.append(1 - e[0]/total)
    total = 0
    for e in percent:
        total = total + e
    roulette = [0]
    val = 0
    for e in percent:
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

def crossover(parents, num_parents_mating,fitness):
    offspring = numpy.empty(parents.shape)
    n = offspring.shape[0]
    i = 0
    while i < n:
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
    for idx in range(offspring_crossover.shape[0]):
        for j in range(offspring_crossover.shape[1]):
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            if(random_value > -0.2 and random_value < 0.2 ): 
                mut = numpy.random.uniform(0.05,0.15)
                s = numpy.random.choice([-1,1])
                offspring_crossover[idx, j] = offspring_crossover[idx, j]*(1+s*mut)
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
    print('Errors: ',var_fitness,end='\n\n')
    fitness = fitness_function(fitness)
    parents = select_parents(new_population, fitness)
    offspring_crossover = crossover(parents,num_parents_mating,fitness)
    offspring_mutation = mutation(offspring_crossover)
    new_population = offspring_mutation

answer = json.dumps(new_population.tolist())
with open('./output.txt','w+') as write_file:
    json.dump(answer, write_file)
print('************************************')
# end here

