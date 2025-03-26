import os
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sissopp import Inputs, FeatureSpace, SISSORegressor
from sissopp.postprocess.load_models import load_model

# Define required columns that must NOT be dropped
REQUIRED_COLUMNS = {'E_RS - E_ZB', '# Material', 'EA_A'}


def create_random_drop_datasets(data_path, save_path, num_drops):
    """
    Create modified datasets by randomly dropping columns and save them to a specified path.
    Ensure that required columns are not dropped.
    """
    data = pd.read_csv(data_path)
    os.makedirs(save_path, exist_ok=True)

    for i in range(num_drops):
        # Identify columns that can be dropped (excluding required ones)
        drop_candidates = [col for col in data.columns if col not in REQUIRED_COLUMNS]

        # Randomly determine how many columns to drop (at least 1)
        num_cols_to_drop = np.random.randint(1, len(drop_candidates) + 1)

        # Randomly select columns to drop
        drop_columns = np.random.choice(drop_candidates, num_cols_to_drop, replace=False)

        # Create modified dataset by dropping selected columns
        modified_data = data.drop(columns=drop_columns)

        # Save the modified dataset
        modified_file_path = os.path.join(save_path, f'modified_data_random_drop_{i+1}.csv')
        modified_data.to_csv(modified_file_path, index=False)

        print(f"Iteration {i+1}: Dropped {num_cols_to_drop} columns. Data saved to {modified_file_path}.")


create_random_drop_datasets(
    data_path="/home/u28/rahulbangad/data.csv",
    save_path="/home/u28/rahulbangad/outputhai/",
    num_drops=5
)

def update_sisso_json(original_sisso_path, save_path, num_iterations):
    """
    Updates sisso.json files for each iteration with the modified data file name.
    """
    with open(original_sisso_path, 'r') as file:
        sisso_content = json.load(file)

    for iteration in range(1, num_iterations + 1):
        iteration_folder_path = os.path.join(save_path, f'Iteration_{iteration}')
        os.makedirs(iteration_folder_path, exist_ok=True)

        sisso_content["data_file"] = f'modified_data_random_drop_{iteration}.csv'
        new_sisso_file_path = os.path.join(iteration_folder_path, 'sisso.json')

        with open(new_sisso_file_path, 'w') as new_file:
            json.dump(sisso_content, new_file, indent=4)

        print(f'Updated sisso.json saved to {new_sisso_file_path}')


update_sisso_json(
    original_sisso_path="/home/u28/rahulbangad/sisso.json",
    save_path="/home/u28/rahulbangad/sissodatarows/",
    num_iterations=5
)

def create_models(base_sisso_folder_path, models_base_path, output_csv_path, num_iterations):
    """
    Copies sisso.json and corresponding CSV files, processes them, and fits models for each iteration.
    """
    for iteration in range(1, num_iterations + 1):
        iteration_source_path = os.path.join(base_sisso_folder_path, f'Iteration_{iteration}')
        iteration_model_path = os.path.join(models_base_path, f'Iteration_{iteration}')
        csv_file_path = os.path.join(output_csv_path, f'modified_data_random_drop_{iteration}.csv')

        os.makedirs(iteration_model_path, exist_ok=True)

        sisso_file_path = os.path.join(iteration_source_path, 'sisso.json')
        if os.path.exists(sisso_file_path):
            shutil.copy(sisso_file_path, os.path.join(iteration_model_path, 'sisso.json'))
        else:
            print(f"sisso.json not found in {iteration_source_path}")
            continue

        if os.path.exists(csv_file_path):
            shutil.copy(csv_file_path, os.path.join(iteration_model_path, f'modified_data_random_drop_{iteration}.csv'))
        else:
            print(f"CSV file {csv_file_path} not found.")
            continue

        os.chdir(iteration_model_path)

        try:
            inputs = Inputs('sisso.json')
            feature_space = FeatureSpace(inputs)
            sisso = SISSORegressor(inputs, feature_space)
            sisso.fit()
            print(f"Model fitted and saved for iteration {iteration}.")
        except Exception as e:
            print(f"Error fitting model for iteration {iteration}: {e}")


create_models(
    base_sisso_folder_path="/home/u28/rahulbangad/sissodatarows/",
    models_base_path="/home/u28/rahulbangad/models/rowsmodel/",
    output_csv_path="/home/u28/rahulbangad/outputhai/",
    num_iterations=5
)


def process_file_set(it_number, base_path):
    """
    Processes and evaluates models for a given holdout set.
    """
    num_iterations = 5
    holdout_file_path = os.path.join(base_path, f"Iteration_{it_number}/modified_data_random_drop_{it_number}.csv")
    model_files = [os.path.join(base_path, f"Iteration_{i}/models/train_dim_4_model_0.dat") for i in range(1, num_iterations + 1)]

    if not os.path.exists(holdout_file_path) or any(not os.path.exists(file) for file in model_files):
        print(f"Missing files for iteration {it_number}.")
        return

    data_predict = pd.read_csv(holdout_file_path)
    features = data_predict.drop(columns=REQUIRED_COLUMNS, errors='ignore')
    features_dict = {col: features[col].values for col in features.columns}

    models = [load_model(file) for file in model_files]
    predictions = [model.eval_many(features_dict) for model in models]
    
    data_predict['Average Prediction'] = np.mean(predictions, axis=0)
    data_predict['Std Deviation'] = np.std(predictions, axis=0)

    plt.figure(figsize=(12, 7))
    plt.scatter(data_predict['EA_A'], data_predict['Average Prediction'], alpha=0.5)
    plt.errorbar(data_predict['EA_A'], data_predict['Average Prediction'], yerr=data_predict['Std Deviation'], fmt='o', alpha=0.5)
    plt.xlabel('Actual EA_A')
    plt.ylabel('Predicted EA_A')
    plt.title(f'Holdout Set {it_number} - Predictions')
    plt.show()

# Function to evaluate all holdout sets
def process_all_holdout_sets(base_path, num_iterations=5):
    for it_number in range(1, num_iterations + 1):
        process_file_set(it_number, base_path)

# Run evaluation
process_all_holdout_sets(base_path="/home/u28/rahulbangad/models/rowsmodel/", num_iterations=5)
