'''
    @author: Amer Sulieman
    @version: 09/15/2019
'''

import random


class Chromosome:
    '''
     A class that represent a single chromosome vector
     I care mainly about the fitness, when objects are compared
     They are object to their fitness
    '''

    def __init__(self):
        self.vector = []
        self.fitness = None

    # fills up the chromosome with random numbers for each beta
    def generate_random_betas(self, num_betas, num_bits):
        for value in range(num_betas):
            random_beta = random.randint(0, (2**num_bits)-1)
            self.vector.append(random_beta)

    # Object less than method for comparison
    def __lt__(self, other):
        return self.fitness < other.fitness

    # Object less than or equal method for comparison
    def __le__(self, other):
        if self.fitness == other.fitness or self.fitness < other.fitness:
            return True
        return False

    # Object greater than method for comparison
    def __gt__(self, other):
        return self.fitness > other.fitness

    # Object greater than or equal method for comparison
    def __ge__(self, other):
        if self.fitness > other.fitness or self.fitness == other.fitness:
            return True
        return False

    # Object equal method for comparison
    def __eq__(self, other):
        return self.fitness == other.fitness

    # Object not equal method for comparison
    def __ne__(self, other):
        return self.fitness != other.fitness

    # Object representation method for comparison
    def __str__(self):
        return str(self.fitness)
