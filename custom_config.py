'''
    A file that handles user choice and reading exit conditions
'''
import re as regex


def ask_user_for_their_choice():
    while True:
        user_choice = input("The default values are as follow:\n" +
                            "population_size = 200\nElites_rate = 0.05\n" +
                            "cross_over_rate = 0.80\nmutation_rate = 0.10\n" +
                            "Would you like to change them?\n" +
                            "Enter Y or N : ")
        if user_choice.lower() == "y":
            print("\n\n")
            return True
        elif user_choice.lower() == "n":
            return False
        else:
            print("\n\nYour answer should be Y or N")


def get_config_values_from_user():
    while True:
        print("Enter population size as positive Integer!")
        population_size = input("Population size = ")
        try:
            population_size = int(population_size)
        except:
            continue
        if population_size < 0:
            print("\nPopulation size must be positive number")
        else:
            break

    while(True):
        print(
            "\nEnter your rates as positive decimals between 0.0 and 1.0!! Example : 0.9 or 0.5")
        elites_rate = input("Elite rate: ")
        cross_over_rate = input("Crossover rate: ")
        mutation_rate = input("Mutation rate: ")
        try:
            elites_rate = float(elites_rate)
            cross_over_rate = float(cross_over_rate)
            mutation_rate = float(mutation_rate)
        except:
            continue
        if elites_rate < 0 or cross_over_rate < 0 or mutation_rate < 0:
            continue
        elif elites_rate > 1 or cross_over_rate > 1 or mutation_rate > 1:
            continue
        else:
            break
    return population_size, elites_rate, cross_over_rate, mutation_rate
