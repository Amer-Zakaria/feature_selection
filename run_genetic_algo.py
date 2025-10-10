from data_preprocessing import data_preprocessing
from genetiec_algo import FeatureSelectionGeneticAlgorithm
from model_evaluation import evaluate
import datetime
import numpy as np


def run_genetic_algorithm(csv_url="data.csv"):
    try:
        result = data_preprocessing(csv_url=csv_url)

        # Initialize the genetic algorithm
        ga = FeatureSelectionGeneticAlgorithm(
            population_size=10,
            generations=20,
            mutation_rate=0.1,
            X_train=result.X_train,
            X_test=result.X_test,
            y_train=result.y_train,
            y_test=result.y_test,
        )

        (best_individual, best_fitness) = ga.solve()

        # Obtain time & MSE on the original data
        start_time = datetime.datetime.now()
        original_mse = evaluate(
            result.X_train,
            result.X_test,
            result.y_train,
            result.y_test,
        )
        elapsed = datetime.datetime.now() - start_time
        original_time = int(elapsed.total_seconds() * 1000)

        # Obtain time & MSE after feature selection
        start_time = datetime.datetime.now()
        mask = np.array(best_individual, dtype=bool)
        GA_mse = evaluate(
            result.X_train.loc[:, mask],
            result.X_test.loc[:, mask],
            result.y_train,
            result.y_test,
        )
        elapsed = datetime.datetime.now() - start_time
        GA_time = int(elapsed.total_seconds() * 1000)

        message_parts = [
            f"Number of selected features (GA): {sum(best_individual)} out of {result.X_train.shape[1]}.",
            f"Time & MSE of the original data: {original_time}ms & {original_mse}.",
            f"Time & MSE after Genetic Algorithm feature selection: {GA_time}ms & {GA_mse}.",
        ]

        return {
            "success": True,
            "messages": message_parts,
        }
    except Exception as e:
        return {"success": False, "message": f"Error: {e}"}
