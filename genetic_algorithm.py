'''
    @author: Amer Sulieman
    @version: 09/15/2019
'''
from Chromosome import Chromosome
import random
import math
import numpy as np
import os.path as path
from custom_config import ask_user_for_their_choice as user_choice
from custom_config import get_config_values_from_user as user_data


def calc_num_elites(pop_size, elites_rate):
    '''
        Ensure that there are at least 1 elite in the population
    '''
    num_elites = int(pop_size * elites_rate)
    if num_elites < 1:
        return 1
    return num_elites


def calc_fitness(chromosome):
    '''
        Fitness function is f = 1 + [summantion (xi^2/4000)] - [Pi(cos(xi/sqrt(i)))]
    '''
    summation = 0
    for beta in chromosome:
        # shifts my beta to be in range -1024 to 1023
        beta_in_range = (beta - 2**(NUM_BITS-1))/32
        res = (beta_in_range**2)/4000
        summation += res

    pi = 1
    for i in range(0, NUM_BETAS):
        beta = chromosome[i]
        # shifts my beta to be in range -1024 to 1023
        beta_in_range = (beta - 2**(NUM_BITS-1))/32
        root_i = (i+1)**0.5
        res = math.cos(beta/root_i)
        pi *= res

    fitness = 1 + summation - pi
    return fitness


def create_population():
    '''
        Creates a population of chromosome objects
    '''
    return [Chromosome() for value in range(population_size)]


def initialize():
    '''
        For the initial round, fill
    '''
    population = create_population()
    for chromosome in population:
        chromosome.generate_random_betas(NUM_BETAS, NUM_BITS)
        chromosome.fitness = calc_fitness(chromosome.vector)
    return population


def copy_elites(num_elites, previous_population, next_population):
    '''
        Copies the elites from previous population to the new one
    '''
    for index in range(num_elites):
        next_population[index] = previous_population[index]


def roulette_wheel(population):
    '''
        Calculate the probability for each chromosome to be picked
        For crossover.
        Add up all the fitnesses and dived each chromosome by total
        to get probability
    '''
    all_fitnesses = [chromosome.fitness for chromosome in population]
    total = sum(all_fitnesses)
    # probabilities in order for best to worst, as population is sorted
    probabilities = [fitness/total for fitness in all_fitnesses]
    return probabilities


def __swap_beta_partially(bit_location, chromA_beta, chromeB_beta):
    '''
        !!!!!!!Not the entire beta fully is crossed!!!!!!
        This is a helper function for swapping bits within a beta.
        For example if the random bit picked for swapping is within
        a beta range, like bit 5 . Then for chromosomeA beta's bits after 5
        And chromosomeB beta's bits after 5 should be swapped.
    '''
    # mask to help us know which bits are not swapped
    mask_size = NUM_BITS - bit_location - 1
    mask = (2**mask_size) - 1
    '''
        Get the bits we do not swap
        Example,if binary of 6 -> 0110
        We swap at location 2 <- this is index of bits
        then 011 | 0
        the last bit is not swapped
    '''
    chromoA_bits_not_swap = chromA_beta & mask
    chromoB_bits_not_swap = chromeB_beta & mask
    # Xor the previous two variables to retrive later the swapped bits
    a_and_b_unswapped_bits_mask = chromoA_bits_not_swap ^ chromoB_bits_not_swap
    # chromosome A after swapping partially with B
    new_a = chromeB_beta ^ a_and_b_unswapped_bits_mask
    # chromosome B after swapping partially with A
    new_b = chromA_beta ^ a_and_b_unswapped_bits_mask
    return new_a, new_b


def __two_chromosomes_cross_over(bit_location, chromA, chromB):
    '''
        A function that handles crossover of two picked chromosomes
    '''
    new_chromo_A = chromA[:]
    new_chromo_B = chromB[:]
    # Find which beta the crossover starts at
    beta_location = int(bit_location/NUM_BITS)
    # swap betas from specified location between two chromosed
    for beta in range(beta_location, NUM_BETAS):
        new_chromo_A[beta] = chromB[beta]
        new_chromo_B[beta] = chromA[beta]
    # Get at which bit the cross happend and if needed to cross
    # two betas partially
    specific_bit_cross = beta_location % NUM_BITS
    beta_last_bit = NUM_BETAS-1
    # if the bit is not last one then a beta has to be crossed partially, not fully
    if specific_bit_cross != beta_last_bit:
        A_beta = chromA[beta_location]
        B_beta = chromB[beta_location]
        new_A_beta, new_B_beta = __swap_beta_partially(
            specific_bit_cross, A_beta, B_beta)
        new_chromo_A[beta_location] = new_A_beta
        new_chromo_B[beta_location] = new_B_beta
    return new_chromo_A, new_chromo_B


def cross_over(elite_location, num_of_cross_overs, prev_pop, next_pop):
    '''
        Main cross over function to cross for specific rate of the population
    '''
    # Add the new chromosomes after the elites
    next_population_index = elite_location + 1
    randomness_probability = roulette_wheel(prev_pop)
    # loop to cross over two chromosomes for specific rate, step size is two
    for i in range(0, num_of_cross_overs, 2):
        # pick two chromosomes to cross
        chromos_to_cross = np.random.choice(
            prev_pop, size=2, p=randomness_probability)
        chromA = chromos_to_cross[0].vector
        chromB = chromos_to_cross[1].vector
        # pick any bit location from 0 to 159 <- index of bits
        cross_over_location = random.randint(0, chromosome_size-1)
        new_chromo1, new_chromo2 = __two_chromosomes_cross_over(
            cross_over_location, chromA, chromB)

        # add the new chromosomes to next population
        next_pop[next_population_index].vector = new_chromo1[:]
        next_population_index += 1
        next_pop[next_population_index].vector = new_chromo2[:]
        next_population_index += 1


def mutation(mutations, chromosome_size, population_size, population):
    '''
        A function that picks random bit location to mutate
    '''
    index_after_elites = num_of_elites
    for i in range(mutations):
        # Pick any chromosome that is not an elite
        random_chromo = random.randint(index_after_elites, population_size-1)
        # Pick random bit location to change
        random_bit_location = random.randint(0, chromosome_size-1)
        # which beta in chromosome will have its bit mutated
        beta_location = int(random_bit_location / NUM_BITS)
        # The location of the bit in that beta to be mutated
        bit_to_change = random_bit_location % NUM_BITS
        chromosome = population[random_chromo]
        # xor bit with 1 to flip it
        chromosome.vector[beta_location] ^= (1 << bit_to_change)


def genetics_algorithm():

    # used to track how many generations's best fitness does not change
    no_change_tracker = 0
    population = initialize()
    population.sort(reverse=True)
    result = ""
    result += "Generation= 0, Fitness= {}\n".format(population[0])
    generation = 1
    while generation < (generations+1) and no_change_tracker < num_no_change_condition:
        next_population = create_population()
        copy_elites(num_of_elites, population, next_population)
        elite_location = num_of_elites - 1
        cross_over(elite_location, num_cross_overs,
                   population, next_population)
        rest_population_to_fill = num_of_elites + num_cross_overs
        # fill rest population with new chromosomes
        for chromosomeLeft in range(rest_population_to_fill, population_size):
            next_population[chromosomeLeft].generate_random_betas(
                NUM_BETAS, NUM_BITS)
        # mutate the newly created population
        mutation(mutations, chromosome_size, population_size, next_population)
        # calculate the rest of the population fitness after elites
        for i in range(num_of_elites, population_size):
            chromosome = next_population[i]
            chromosome.fitness = calc_fitness(chromosome.vector)
        next_population.sort(reverse=True)
        if population[0] == next_population[0]:
            no_change_tracker += 1
        else:
            no_change_tracker = 0
        population = next_population[:]
        result += "Generation= {}, Fitness= {}\n".format(
            generation, population[0])
        generation += 1
    return result


# The following all are necessary data for the algorithm
generations = 100
iterations = 5
# default values
population_size = 200
elites_rate = 0.05
cross_over_rate = 0.80
mutation_rate = 0.10
NUM_BETAS = 10
NUM_BITS = 16
chromosome_size = NUM_BETAS * NUM_BITS
mutations = int(mutation_rate * population_size)
num_cross_overs = int(population_size * cross_over_rate)
num_of_elites = calc_num_elites(population_size, elites_rate)
num_no_change_condition = 0.10 * population_size
# check if user want to change default values
user_want_to_change = user_choice()
if user_want_to_change:
    data = user_data()
    population_size = data[0]
    elites_rate = data[1]
    cross_over_rate = data[2]
    mutation_rate = data[3]
###########################################################
# call the algorithm
# call the algorithm
result = genetics_algorithm()
output_file = "Output_SGA.txt"
with open(output_file, "w") as out_file:
    out_file.write(result)
