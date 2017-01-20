from random import *
from operator import attrgetter
import numpy as np
import tensorflow as tf
#from soma import soma
from cnn import cnn, file_results
from copy import copy

nb_epochs = 10
mutation_rate = 0.3
crossover_rate = 0.7
nb_population = 6
individual_size = 3


class Individual:
    def __init__(self, individual="", fitness=0.0):
        self.individual = individual
        self.fitness = fitness

    def length(self):
        return len(self.individual)


def get_random_individuals(length):
    result = []
    for i in xrange(length):
        bit = randint(2, 100)
        result.append(bit)

    return result


def print_population(population):
    for i in xrange(nb_population):
        print (population[i].individual, population[i].fitness)

    print


def get_fitness(population):
    total_fitness = 0.0
    for i in xrange(nb_population):
        if population[i].fitness == 0.0:
            population[i].fitness = cnn(population[i].individual)
        total_fitness += population[i].fitness

    return total_fitness


def roulette_select(total_fitness, population):
    fitness_slice = random() * total_fitness
    fitness_so_far = 0.0

    for i in xrange(nb_population):
        fitness_so_far += population[i].fitness

        if fitness_so_far >= fitness_slice:
            return population[i].individual

    return None


def crossover(ind1, ind2):
    if random() < crossover_rate:
        i = randint(0, 1)
        if i == 0:
            temp_bit = ind1[0]
            ind1[0] = ind2[0]
            ind2[0] = temp_bit
            return True
        else:
            temp_bit = ind1[2]
            ind1[2] = ind2[2]
            ind2[2] = temp_bit
            return True

    return False


def mutation(ind):
    if random() < mutation_rate:
        i = randint(0, 2)
        mutate = gauss(0.0, 5.0)
        ind[i] = int(round(ind[i] + mutate))
        if ind[i] < 1.0:
            ind[i] = 1
        return True

    return False


def elitism(population, temp_population):
    population = sorted(population, key=attrgetter('fitness'), reverse=True)
    sample_population = sample(temp_population, 4)
    current_population = np.hstack((population[:2], sample_population))

    return current_population


def main():

    solution_found = False
    population = []
    epoch = 1

    for i in xrange(nb_population):
        ind = Individual(get_random_individuals(individual_size))
        population.append(ind)

    while not solution_found:

        print("Epoch: " + str(epoch))

        total_fitness = get_fitness(population)

        print_population(population)

        for i in xrange(nb_population):
            if population[i].fitness > 295:
                solution_found = True
                break

        if solution_found:
            break

        temp_population = []

        pop = 0

        while pop < nb_population:
            ind1 = None
            while ind1 is None:
                ind1 = roulette_select(total_fitness, population)

            ind2 = None
            while ind2 is None:
                ind2 = roulette_select(total_fitness, population)

            ind1, ind2 = copy(ind1), copy(ind2)

            crossover(ind1, ind2)

            mutation(ind1)

            mutation(ind2)

            temp_population.append(Individual(ind1))

            temp_population.append(Individual(ind2))

            pop += 2

        population = elitism(population, temp_population)

        epoch += 1

        if epoch > nb_epochs:
            break

    return True

if __name__ == '__main__':

    main()
    file_results.close()