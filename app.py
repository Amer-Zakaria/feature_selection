import gradio as gr
import datetime
import numpy as np
import pandas as pd
from data_preprocessing import data_preprocessing
from genetiec_algo import FeatureSelectionGeneticAlgorithm
from model_evaluation import evaluate
from statnew import Pearson, Distance
from classicalmethods import perform_rfe


def run_all_feature_selection(
    csv_url="data.csv", population_size=10, generations=18, mutation_rate=0.1
):
    output_messages = []

    # Preprocessing
    data_result = data_preprocessing(csv_url=csv_url)

    # Original Data Evaluation
    start_time = datetime.datetime.now()
    original_mse = evaluate(
        data_result.X_train,
        data_result.X_test,
        data_result.y_train,
        data_result.y_test,
    )
    elapsed = datetime.datetime.now() - start_time
    original_eval_time = int(elapsed.total_seconds() * 1000)
    output_messages.append(f"\n--- Original Data Evaluation ---")
    output_messages.append(f"Model Evaluation Time: {original_eval_time}ms")
    output_messages.append(f"MSE: {original_mse}")

    # Genetic Algorithm
    output_messages.append(f"\n--- Genetic Algorithm Feature Selection ---")
    ga_selection_start_time = datetime.datetime.now()
    ga = FeatureSelectionGeneticAlgorithm(
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        X_train=data_result.X_train,
        X_test=data_result.X_test,
        y_train=data_result.y_train,
        y_test=data_result.y_test,
    )
    (best_individual, best_fitness) = ga.solve()
    ga_selection_elapsed = datetime.datetime.now() - ga_selection_start_time
    ga_selection_time = int(ga_selection_elapsed.total_seconds() * 1000)

    ga_eval_start_time = datetime.datetime.now()
    mask = np.array(best_individual, dtype=bool)
    ga_mse = evaluate(
        data_result.X_train.loc[:, mask],
        data_result.X_test.loc[:, mask],
        data_result.y_train,
        data_result.y_test,
    )
    ga_eval_elapsed = datetime.datetime.now() - ga_eval_start_time
    ga_eval_time = int(ga_eval_elapsed.total_seconds() * 1000)

    output_messages.append(
        f"Number of selected features: {sum(best_individual)} out of {data_result.X_train.shape[1]}"
    )
    output_messages.append(f"Feature Selection Process Time: {ga_selection_time}ms")
    output_messages.append(f"Model Evaluation Time: {ga_eval_time}ms")
    output_messages.append(f"MSE: {ga_mse}")

    # Statistical Methods (Pearson, Distance)
    X_combined = pd.concat([data_result.X_train, data_result.X_test])
    y_combined = pd.concat([data_result.y_train, data_result.y_test])
    num_features_for_statistical_methods = 10

    # Pearson Correlation
    output_messages.append(f"\n--- Pearson Correlation Feature Selection ---")
    pearson_selection_start_time = datetime.datetime.now()
    pearson_results = Pearson(X_combined, y_combined)
    pearson_selected_features = pearson_results.index.tolist()[
        :num_features_for_statistical_methods
    ]
    pearson_selection_elapsed = datetime.datetime.now() - pearson_selection_start_time
    pearson_selection_time = int(pearson_selection_elapsed.total_seconds() * 1000)

    pearson_eval_start_time = datetime.datetime.now()
    pearson_mse = evaluate(
        data_result.X_train.loc[:, pearson_selected_features],
        data_result.X_test.loc[:, pearson_selected_features],
        data_result.y_train,
        data_result.y_test,
    )
    pearson_eval_elapsed = datetime.datetime.now() - pearson_eval_start_time
    pearson_eval_time = int(pearson_eval_elapsed.total_seconds() * 1000)

    output_messages.append(
        f"Number of selected features: {len(pearson_selected_features)}"
    )
    output_messages.append(
        f"Feature Selection Process Time: {pearson_selection_time}ms"
    )
    output_messages.append(f"Model Evaluation Time: {pearson_eval_time}ms")
    output_messages.append(f"MSE: {pearson_mse}")

    # Distance Correlation
    output_messages.append(f"\n--- Distance Correlation Feature Selection ---")
    distance_selection_start_time = datetime.datetime.now()
    distance_results = Distance(X_combined, y_combined)
    distance_selected_features = distance_results.index.tolist()[
        :num_features_for_statistical_methods
    ]
    distance_selection_elapsed = datetime.datetime.now() - distance_selection_start_time
    distance_selection_time = int(distance_selection_elapsed.total_seconds() * 1000)

    distance_eval_start_time = datetime.datetime.now()
    distance_mse = evaluate(
        data_result.X_train.loc[:, distance_selected_features],
        data_result.X_test.loc[:, distance_selected_features],
        data_result.y_train,
        data_result.y_test,
    )
    distance_eval_elapsed = datetime.datetime.now() - distance_eval_start_time
    distance_eval_time = int(distance_eval_elapsed.total_seconds() * 1000)

    output_messages.append(
        f"Number of selected features: {len(distance_selected_features)}"
    )
    output_messages.append(
        f"Feature Selection Process Time: {distance_selection_time}ms"
    )
    output_messages.append(f"Model Evaluation Time: {distance_eval_time}ms")
    output_messages.append(f"MSE: {distance_mse}")

    # Classical Method (RFE)
    output_messages.append(f"\n--- Recursive Feature Elimination (RFE) ---")
    no_of_features = 30
    rfe_selection_start_time = datetime.datetime.now()
    rfe_results = perform_rfe(
        split_data=data_result, n_features_to_select=no_of_features, step=1
    )
    rfe_selection_elapsed = datetime.datetime.now() - rfe_selection_start_time
    rfe_selection_time = int(rfe_selection_elapsed.total_seconds() * 1000)

    rfe_eval_start_time = datetime.datetime.now()
    rfe_mse = evaluate(
        data_result.X_train.loc[:, rfe_results.selected_features],
        data_result.X_test.loc[:, rfe_results.selected_features],
        data_result.y_train,
        data_result.y_test,
    )
    rfe_eval_elapsed = datetime.datetime.now() - rfe_eval_start_time
    rfe_eval_time = int(rfe_eval_elapsed.total_seconds() * 1000)

    output_messages.append(
        f"Number of selected features: {no_of_features} out of {data_result.X_train.shape[1]}"
    )
    output_messages.append(f"Feature Selection Process Time: {rfe_selection_time}ms")
    output_messages.append(f"Model Evaluation Time: {rfe_eval_time}ms")
    output_messages.append(f"MSE: {rfe_mse}")

    return "\n".join(output_messages)


def process_csv(file="data.csv", population_size=10, generations=18, mutation_rate=0.1):
    return run_all_feature_selection(file, population_size, generations, mutation_rate)


# Create Gradio interface
demo = gr.Interface(
    fn=process_csv,
    inputs=[
        gr.Text(
            label="Dataset URL (CSV)",
            placeholder="Keep our dataset URL or replace it with your own",
            value="https://raw.githubusercontent.com/Amer-Zakaria/600-features-dataset/refs/heads/main/data.csv",
        ),
        gr.Slider(
            minimum=10,
            maximum=50,
            value=10,
            step=1,
            label="GA Population Size",
            info="Choose the population size for the Genetic Algorithm (10 is well-tested on our dataset)",
        ),
        gr.Slider(
            minimum=5,
            maximum=100,
            value=18,
            step=1,
            label="GA Generations",
            info="Choose the number of generations for the Genetic Algorithm (18 is well-tested on our dataset)",
        ),
        gr.Slider(
            minimum=0.01,
            maximum=0.5,
            value=0.1,
            step=0.01,
            label="GA Mutation Rate",
            info="Choose the mutation rate for the Genetic Algorithm (0.1 is a common starting point)",
        ),
    ],
    outputs=gr.Textbox(label="Results", lines=30),
    title="Feature Selection Methods Comparison",
    description="Provide the URL for the CSV file to compare Genetic Algorithm, Statistical (Pearson, Distance), and Classical (RFE) feature selection methods. THE TARGET COLUMN IS THE FIRST COLUMN OF THE DATASET AFTER THE PREPROCESSING STEP",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
