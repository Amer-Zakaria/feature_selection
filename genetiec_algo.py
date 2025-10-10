import random
import copy
import numpy as np
import pandas as pd
from typing import List, Tuple
from model_evaluation import evaluate


class FeatureSelectionGeneticAlgorithm:
    """
    Implements a genetic algorithm for feature selection.

    The algorithm evolves a population of feature subsets (individuals)
    to find an optimal subset that maximizes a fitness score, which is
    based on model performance (MSE) and the number of selected features.
    """

    def __init__(
        self,
        population_size: int = 10,
        generations: int = 15,
        mutation_rate: float = 0.1,
        X_train: pd.DataFrame = pd.DataFrame(),
        X_test: pd.DataFrame = pd.DataFrame(),
        y_train: pd.Series = pd.Series(),
        y_test: pd.Series = pd.Series(),
    ):
        """
        Initializes the FeatureSelectionGeneticAlgorithm.

        Args:
            population_size (int): The number of individuals in each generation.
            generations (int): The total number of generations to evolve.
            mutation_rate (float): The probability of a gene being mutated.
            train_df (pd.DataFrame): Training data features.
            test_df (pd.DataFrame): Testing data features.
            train_target (pd.Series): Training data target variable.
            test_target (pd.Series): Testing data target variable.
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.mse_base = evaluate(
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        )

    def _create_individual(self) -> List[int]:
        """
        Creates a random individual representing a feature subset.

        An individual is a binary list where 1 indicates feature selection
        and 0 indicates exclusion. The number of selected features is
        randomly chosen between 10% and 90% of the total features for each individual.
        And that would ensure diversity.

        Returns:
            List[int]: A binary list representing the feature subset.
        """
        num_features = len(self.X_train.columns)
        min_ones = max(1, int(0.1 * num_features))
        max_ones = max(1, int(0.9 * num_features))
        num_ones = random.randint(min_ones, max_ones)

        individual = [0] * num_features
        ones_indices = random.sample(range(num_features), num_ones)
        for i in ones_indices:
            individual[i] = 1

        return individual

    def _calculate_fitness(self, individual: List[int]) -> float:
        """minimize MSE only."""
        mask = np.array(individual, dtype=bool)

        if sum(individual) == 0:
            return 0.0

        mse = evaluate(
            self.X_train.loc[:, mask],
            self.X_test.loc[:, mask],
            self.y_train,
            self.y_test,
        )

        if np.isnan(mse) or np.isinf(mse):
            return 0.0

        # Lower MSE = Higher fitness
        return 1.0 / (1.0 + mse)

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Performs single-point crossover between two parents to create an offspring.

        The child inherits the first half of its genes from parent1 and the
        second half from parent2.

        Args:
            parent1 (List[int]): The first parent individual.
            parent2 (List[int]): The second parent individual.

        Returns:
            List[int]: The offspring individual.
        """
        half = len(parent1) // 2
        child = parent1[:half] + parent2[half:]
        return child

    def _mutate(self, individual: List[int]) -> List[int]:
        """
        Mutates an individual by flipping random bits (features).

        For each gene in the individual, there's a `self.mutation_rate`
        probability of flipping its value (0 to 1, or 1 to 0).

        Args:
            individual (List[int]): The individual to be mutated.

        Returns:
            List[int]: The mutated individual.
        """
        mutated = copy.deepcopy(individual)
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip the bit (0 to 1, or 1 to 0)
        return mutated

    def _select_parents(
        self, population: List[List[int]], fitness_scores: List[float]
    ) -> Tuple[List[int], List[int]]:
        """
        Selects two parents from the population using tournament selection.

        A small group of individuals (tournament_size) is randomly chosen,
        and the individual with the highest fitness from this group is selected.
        This process is repeated twice to get two parents.

        Args:
            population (List[List[int]]): The current population of individuals.
            fitness_scores (List[float]): The fitness scores corresponding to the population.

        Returns:
            Tuple[List[int], List[int]]: A tuple containing the two selected parent individuals.
        """

        def tournament_select():
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]

            winner_idx = tournament_indices[
                tournament_fitness.index(max(tournament_fitness))
            ]
            return population[winner_idx]

        parent1 = tournament_select()
        parent2 = tournament_select()
        return parent1, parent2

    def solve(self) -> Tuple[List[int], float]:
        """
        Executes the main genetic algorithm loop to find the best feature subset.

        The algorithm initializes a population, then iteratively evolves it
        through selection, crossover, and mutation over a specified number
        of generations. It tracks and returns the best individual (feature subset)
        and its fitness score found throughout the evolution.

        Returns:
            Tuple[List[int], float]: A tuple containing the best feature subset
                                     (as a binary list) and its fitness score.
        """
        population = [self._create_individual() for _ in range(self.population_size)]

        best_individual = [0] * len(self.X_train.columns)
        best_fitness = float(0)

        for _ in range(self.generations):
            fitness_scores = [self._calculate_fitness(ind) for ind in population]

            # Update best solution found so far
            for ind, fitness in zip(population, fitness_scores):
                if fitness > best_fitness:
                    best_individual = ind
                    best_fitness = fitness

            # Elitism: Carry over the best individual found so far directly to the new population
            new_population = [best_individual]

            # Fill the rest of the new population with offspring
            while len(new_population) < self.population_size:
                parent1, parent2 = self._select_parents(population, fitness_scores)
                child = self._crossover(parent1, parent2)
                mutated_child = self._mutate(child)
                new_population.append(mutated_child)

            population = new_population

        return (best_individual, best_fitness)
