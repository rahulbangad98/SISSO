import pandas as pd
import numpy as np
import os

def create_training_splits(data_path, save_path, num_iterations, train_splits_per_iteration):
    """
    Create hold-out sets and multiple training datasets for each iteration.
    Print the number of rows dropped while creating each hold-out set.
    """
    data = pd.read_csv(data_path)
    os.makedirs(save_path, exist_ok=True)

    for iteration in range(1, num_iterations + 1):
        # Create hold-out set
        holdout_size = int(0.2 * len(data))
        holdout_indices = np.random.choice(data.index, size=holdout_size, replace=False)
        holdout_set = data.loc[holdout_indices]
        holdout_set.to_csv(f"{save_path}/holdout_set_{iteration}.csv", index=False)

        # Create remaining data
        remaining_data = data.drop(holdout_indices)

        # Print details about the hold-out set
        print(f"Iteration {iteration}: Hold-out set created with {holdout_size} rows dropped. "
              f"Remaining rows for training: {len(remaining_data)}.")

        # Create multiple training subsets
        for split in range(1, train_splits_per_iteration + 1):
            train_indices = np.random.choice(remaining_data.index, size=int(0.8 * len(remaining_data)), replace=False)
            train_split = remaining_data.loc[train_indices]
            train_split.to_csv(f"{save_path}/train_split_{iteration}_{split}.csv", index=False)

        print(f"Iteration {iteration}: Created {train_splits_per_iteration} training splits.")

# Example usage
create_training_splits(
    data_path="data.csv", 
    save_path="splits/", 
    num_iterations=5, 
    train_splits_per_iteration=3
)
#first
# import json
# import os

# def update_sisso_json(original_sisso_path, save_path, num_iterations):
#     """
#     Updates sisso.json files for each iteration with the specific training file name
#     and saves them in a new folder for each iteration.
#     """
#     # Load the original sisso.json content
#     with open(original_sisso_path, 'r') as file:
#         sisso_content = json.load(file)
    
#     for iteration in range(1, num_iterations + 1):
#         # Define the iteration-specific folder path
#         iteration_folder_path = os.path.join(save_path, f'Iteration_{iteration}')
        
#         # Ensure the iteration-specific folder exists
#         os.makedirs(iteration_folder_path, exist_ok=True)
        
#         # Update the data_file path in sisso.json with the appropriate train_split file
#         sisso_content["data_file"] = f'train_split_{iteration}_1.csv'
        
#         # Define the new sisso.json file path
#         new_sisso_file_path = os.path.join(iteration_folder_path, 'sisso.json')
        
#         # Save the updated sisso.json file
#         with open(new_sisso_file_path, 'w') as new_file:
#             json.dump(sisso_content, new_file, indent=4)
        
#         print(f'Updated sisso.json for Iteration {iteration} saved to {new_sisso_file_path}')

# # Example usage
# update_sisso_json(
#     original_sisso_path="/home/u28/rahulbangad/sisso.json",
#     save_path="/home/u28/rahulbangad/sissodatarows/",
#     num_iterations=5
# )
import json
import os

def update_sisso_json(original_sisso_path, save_path, num_iterations, train_splits_per_iteration):
    """
    Updates sisso.json files for each iteration with specific training file names
    and saves them in a new folder for each iteration.
    """
    # Load the original sisso.json content
    with open(original_sisso_path, 'r') as file:
        sisso_content = json.load(file)
    
    for iteration in range(1, num_iterations + 1):
        # Define the iteration-specific folder path
        iteration_folder_path = os.path.join(save_path, f'Iteration_{iteration}')
        
        # Ensure the iteration-specific folder exists
        os.makedirs(iteration_folder_path, exist_ok=True)
        
        # Create sisso.json for each training split
        for split in range(1, train_splits_per_iteration + 1):
            sisso_content["data_file"] = f'train_split_{iteration}_{split}.csv'
            # Define the new sisso.json file path
            new_sisso_file_path = os.path.join(iteration_folder_path, f'sisso_split_{split}.json')
            
            # Save the updated sisso.json file
            with open(new_sisso_file_path, 'w') as new_file:
                json.dump(sisso_content, new_file, indent=4)
            
            print(f'Updated sisso_split_{split}.json for Iteration {iteration} saved to {new_sisso_file_path}')

# Example usage
update_sisso_json(
    original_sisso_path="/home/u28/rahulbangad/sisso.json",
    save_path="/home/u28/rahulbangad/sissodatarows/",
    num_iterations=5,
    train_splits_per_iteration=3
)

#first

# import os
# import shutil
# import os
# import shutil
# import pandas as pd
# import random
# from sissopp import Inputs, FeatureSpace, SISSORegressor
# from sissopp.postprocess.load_models import load_model

# def create_models(base_sisso_folder_path, models_base_path, output_csv_path, num_iterations):
#     """
#     Copies sisso.json and corresponding CSV files, processes them, and fits models for each iteration.
#     """
#     for iteration in range(1, num_iterations + 1):
#         # Define paths for the current iteration
#         iteration_source_path = os.path.join(base_sisso_folder_path, f'Iteration_{iteration}')
#         iteration_model_path = os.path.join(models_base_path, f'Iteration_{iteration}')
#         csv_file_path = os.path.join(output_csv_path, f'train_split_{iteration}_1.csv')
        
#         # Ensure the destination folder for the model exists
#         os.makedirs(iteration_model_path, exist_ok=True)

#         # Copy sisso.json file
#         sisso_file_path = os.path.join(iteration_source_path, 'sisso.json')
#         if os.path.exists(sisso_file_path):
#             destination_sisso_path = os.path.join(iteration_model_path, 'sisso.json')
#             shutil.copy(sisso_file_path, destination_sisso_path)
#             print(f"Copied {sisso_file_path} to {destination_sisso_path}")
#         else:
#             print(f"sisso.json not found in {iteration_source_path}")
#             continue  # Skip to the next iteration if sisso.json is missing

#         # Copy the corresponding CSV file
#         if os.path.exists(csv_file_path):
#             destination_csv_path = os.path.join(iteration_model_path, f'train_split_{iteration}_1.csv')
#             shutil.copy(csv_file_path, destination_csv_path)
#             print(f"Copied {csv_file_path} to {destination_csv_path}")
#         else:
#             print(f"CSV file {csv_file_path} not found in output folder.")
#             continue  # Skip to the next iteration if the CSV file is missing

#         # Change to the iteration model directory
#         original_working_directory = os.getcwd()  # Save the current working directory
#         try:
#             os.chdir(iteration_model_path)  # Move into the model directory
#             print(f"Changed working directory to: {iteration_model_path}")

#             # Load inputs and fit the model
#             print(f"Loading inputs from: {destination_sisso_path}")
#             inputs = Inputs(destination_sisso_path)  # Assuming Inputs class exists
#             feature_space = FeatureSpace(inputs)  # Assuming FeatureSpace class exists
#             sisso = SISSORegressor(inputs, feature_space)  # Assuming SISSORegressor class exists

#             print("Fitting the model...")
#             sisso.fit()  # Fit the model
#             print(f"Model fitted for iteration {iteration} and saved in {iteration_model_path}")
#         except Exception as e:
#             print(f"Error fitting model for iteration {iteration}: {e}")
#         finally:
#             # Change back to the original working directory
#             os.chdir(original_working_directory)
#             print(f"Returned to original working directory: {original_working_directory}")
    
#     print(f"All models processed and saved in: {models_base_path}")

# # Example usage
# create_models(
#     base_sisso_folder_path="/home/u28/rahulbangad/sissodatarows/",
#     models_base_path="/home/u28/rahulbangad/models/rowsmodel/",
#     output_csv_path="/home/u28/rahulbangad/splits/",
#     num_iterations=5
# )


#try
import os
import shutil
from sissopp import Inputs, FeatureSpace, SISSORegressor

def create_models(base_sisso_folder_path, models_base_path, output_csv_path, num_iterations, train_splits_per_iteration):
    """
    Copies sisso_split_{i}.json and corresponding CSV files, processes them, and fits models for each training split.
    """
    for iteration in range(1, num_iterations + 1):
        # Define paths for the current iteration
        iteration_source_path = os.path.join(base_sisso_folder_path, f'Iteration_{iteration}')
        iteration_model_path = os.path.join(models_base_path, f'Iteration_{iteration}')
        os.makedirs(iteration_model_path, exist_ok=True)

        # Process each training split for the current iteration
        for split in range(1, train_splits_per_iteration + 1):
            # Copy CSV files
            csv_file_path = os.path.join(output_csv_path, f'train_split_{iteration}_{split}.csv')
            destination_csv_path = os.path.join(iteration_model_path, f'train_split_{iteration}_{split}.csv')
            if os.path.exists(csv_file_path):
                shutil.copy(csv_file_path, destination_csv_path)
                print(f"Copied {csv_file_path} to {destination_csv_path}")
            else:
                print(f"CSV file {csv_file_path} not found in output folder. Skipping split {split}.")
                continue

            # Copy sisso.json files
            sisso_file_path = os.path.join(iteration_source_path, f'sisso_split_{split}.json')
            destination_sisso_path = os.path.join(iteration_model_path, f'sisso_split_{split}.json')
            if os.path.exists(sisso_file_path):
                shutil.copy(sisso_file_path, destination_sisso_path)
                print(f"Copied {sisso_file_path} to {destination_sisso_path}")
            else:
                print(f"sisso_split_{split}.json not found in {iteration_source_path}. Skipping split {split}.")
                continue

            # Train and save the model
            try:
                print(f"Processing training split {split} for iteration {iteration}...")
                print(destination_sisso_path)
                # Load inputs and initialize SISSO model
                inputs = Inputs(destination_sisso_path)
                feature_space = FeatureSpace(inputs)
                sisso = SISSORegressor(inputs, feature_space)

                # Fit the model
                print(f"Fitting the model for training split {split} of iteration {iteration}...")
                sisso.fit()

                # Save the model in the respective folder
                model_save_path = os.path.join(iteration_model_path, f"model_split_{split}.dat")
                sisso.save(model_save_path)
                print(f"Model saved for training split {split} at {model_save_path}")

            except Exception as e:
                print(f"Error fitting model for iteration {iteration}, training split {split}: {e}")

    print(f"All models and sisso.json files processed and saved in: {models_base_path}")

# Example usage
create_models(
    base_sisso_folder_path="/home/u28/rahulbangad/sissodatarows/",
    models_base_path="/home/u28/rahulbangad/models/rowsmodel/",
    output_csv_path="/home/u28/rahulbangad/splits/",
    num_iterations=5,
    train_splits_per_iteration=3
)

#predicton and visualization

# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sissopp.postprocess.load_models import load_model

# def strip_units(column_name):
#     """
#     Removes units (e.g., text in parentheses) from a column name.
#     """
#     return column_name.split('(')[0].strip()

# def process_file_set(it_number, base_path, holdout_path):
#     """
#     Processes and evaluates models for a given holdout set.
#     """
#     model_files = [
#         os.path.join(base_path, f"Iteration_{it_number}/models/train_dim_4_model_0.dat")
#     ]
#     holdout_file_path = os.path.join(holdout_path, f"holdout_set_{it_number}.csv")
    
#     # Check if the holdout file exists
#     if not os.path.exists(holdout_file_path):
#         print(f"Holdout set file not found: {holdout_file_path}")
#         return
    
#     # Check if model files exist
#     missing_models = [file for file in model_files if not os.path.exists(file)]
#     if missing_models:
#         print(f"Model files not found: {missing_models}")
#         return
    
#     print(f"Processing Holdout Set: {holdout_file_path}")
    
#     # Load the holdout dataset
#     data_predict = pd.read_csv(holdout_file_path)
    
#     # Strip units from column names
#     data_predict.columns = [strip_units(col) for col in data_predict.columns]
    
#     # Prepare features by dropping unnecessary columns
#     features = data_predict.drop(['E_RS - E_ZB', '# Material'], axis=1)
#     features_dict = {col: features[col].values for col in features.columns}
    
#     # Load models and predict
#     models = [load_model(file) for file in model_files]
#     predictions = [model.eval_many(features_dict) for model in models]
    
#     # Calculate average predictions and standard deviation
#     average_predictions = np.mean(predictions, axis=0)
#     std_predictions = np.std(predictions, axis=0)
    
#     # Add predictions and error to the DataFrame
#     data_predict['Average Predicted EA_A'] = average_predictions
#     data_predict['Error'] = data_predict['EA_A'] - data_predict['Average Predicted EA_A']
#     data_predict['Std Deviation'] = std_predictions
    
#     # Print the DataFrame with errors
#     print(f"\nPrediction Results for Holdout Set {it_number}:")
#     print(data_predict[['# Material', 'EA_A', 'Average Predicted EA_A', 'Error']])
    
#     # Plot results
#     visualize_predictions(data_predict, it_number)

# def visualize_predictions(data_predict, iteration):
#     """
#     Plots actual vs predicted values for the holdout set.
#     """
#     plt.figure(figsize=(12, 7))
#     plt.scatter(data_predict['EA_A'], data_predict['Average Predicted EA_A'], color='blue', alpha=0.5, label='Predicted vs Actual')
#     plt.errorbar(data_predict['EA_A'], data_predict['Average Predicted EA_A'], yerr=data_predict['Std Deviation'], fmt='o', color='gray', alpha=0.5, label='Error Bars')
#     plt.plot([data_predict['EA_A'].min(), data_predict['EA_A'].max()], 
#              [data_predict['EA_A'].min(), data_predict['EA_A'].max()], 'r--', label='Perfect Prediction')
#     plt.title(f'Actual vs Predicted EA_A with Std Dev for Holdout Set {iteration}')
#     plt.xlabel('Actual EA_A')
#     plt.ylabel('Predicted EA_A')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def process_all_holdout_sets(base_path, holdout_path, num_iterations=5):
#     """
#     Processes and plots predictions for all holdout sets from 1 to num_iterations.
#     """
#     for it_number in range(1, num_iterations + 1):
#         process_file_set(it_number, base_path, holdout_path)

# # Example usage
# models_base_path = "/home/u28/rahulbangad/models/rowsmodel/"
# holdout_sets_path = "/home/u28/rahulbangad/splits/"
# process_all_holdout_sets(base_path=models_base_path, holdout_path=holdout_sets_path, num_iterations=5)


# #new pand v
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sissopp.postprocess.load_models import load_model

# def strip_units(column_name):
#     """
#     Removes units (e.g., text in parentheses) from a column name.
#     """
#     return column_name.split('(')[0].strip()

# def process_file_set(it_number, base_path, holdout_path):
#     """
#     Processes and evaluates models for a given holdout set.
#     """
#     model_files = [
#         os.path.join(base_path, f"Iteration_{it_number}/models/train_dim_4_model_0.dat")
#     ]
#     holdout_file_path = os.path.join(holdout_path, f"holdout_set_{it_number}.csv")
    
#     # Check if the holdout file exists
#     if not os.path.exists(holdout_file_path):
#         print(f"Holdout set file not found: {holdout_file_path}")
#         return
    
#     # Check if model files exist
#     missing_models = [file for file in model_files if not os.path.exists(file)]
#     if missing_models:
#         print(f"Model files not found: {missing_models}")
#         return
    
#     print(f"Processing Holdout Set: {holdout_file_path}")
    
#     # Load the holdout dataset
#     data_predict = pd.read_csv(holdout_file_path)
    
#     # Strip units from column names
#     data_predict.columns = [strip_units(col) for col in data_predict.columns]
    
#     # Prepare features by dropping unnecessary columns, keeping EA_A
#     features = data_predict.drop(['E_RS - E_ZB', '# Material'], axis=1)
#     features_dict = {col: features[col].values for col in features.columns}
    
#     # Load models and predict
#     models = [load_model(file) for file in model_files]
#     predictions = [model.eval_many(features_dict) for model in models]
    
#     # Calculate average predictions and standard deviation
#     average_predictions = np.mean(predictions, axis=0)
#     std_predictions = np.std(predictions, axis=0)
    
#     # Add predictions and error to the DataFrame
#     data_predict['Average Predicted E_RS - E_ZB'] = average_predictions
#     data_predict['Error'] = data_predict['E_RS - E_ZB'] - data_predict['Average Predicted E_RS - E_ZB']
#     data_predict['Std Deviation'] = std_predictions
    
#     # Print the DataFrame with errors
#     print(f"\nPrediction Results for Holdout Set {it_number}:")
#     print(data_predict[['# Material', 'E_RS - E_ZB', 'EA_A', 'Average Predicted E_RS - E_ZB', 'Error']])
    
#     # Plot results
#     visualize_predictions(data_predict, it_number)

# def visualize_predictions(data_predict, iteration):
#     """
#     Plots actual vs predicted values for the holdout set.
#     """
#     plt.figure(figsize=(12, 7))
#     plt.scatter(data_predict['E_RS - E_ZB'], data_predict['Average Predicted E_RS - E_ZB'], color='blue', alpha=0.5, label='Predicted vs Actual')
#     plt.errorbar(data_predict['E_RS - E_ZB'], data_predict['Average Predicted E_RS - E_ZB'], yerr=data_predict['Std Deviation'], fmt='o', color='gray', alpha=0.5, label='Error Bars')
#     plt.plot([data_predict['E_RS - E_ZB'].min(), data_predict['E_RS - E_ZB'].max()], 
#              [data_predict['E_RS - E_ZB'].min(), data_predict['E_RS - E_ZB'].max()], 'r--', label='Perfect Prediction')
#     plt.title(f'Actual vs Predicted E_RS - E_ZB with Std Dev for Holdout Set {iteration}')
#     plt.xlabel('Actual E_RS - E_ZB')
#     plt.ylabel('Predicted E_RS - E_ZB')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def process_all_holdout_sets(base_path, holdout_path, num_iterations=5):
#     """
#     Processes and plots predictions for all holdout sets from 1 to num_iterations.
#     """
#     for it_number in range(1, num_iterations + 1):
#         process_file_set(it_number, base_path, holdout_path)

# # Example usage
# models_base_path = "/home/u28/rahulbangad/models/rowsmodel/"
# holdout_sets_path = "/home/u28/rahulbangad/splits/"
# process_all_holdout_sets(base_path=models_base_path, holdout_path=holdout_sets_path, num_iterations=5)


#print std d

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sissopp.postprocess.load_models import load_model

def strip_units(column_name):
    """
    Removes units (e.g., text in parentheses) from a column name.
    """
    return column_name.split('(')[0].strip()

def process_file_set(it_number, base_path, holdout_path):
    """
    Processes and evaluates models for a given holdout set.
    """
    model_files = [
        os.path.join(base_path, f"Iteration_{it_number}/models/train_dim_4_model_0.dat")
    ]
    holdout_file_path = os.path.join(holdout_path, f"holdout_set_{it_number}.csv")
    
    # Check if the holdout file exists
    if not os.path.exists(holdout_file_path):
        print(f"Holdout set file not found: {holdout_file_path}")
        return
    
    # Check if model files exist
    missing_models = [file for file in model_files if not os.path.exists(file)]
    if missing_models:
        print(f"Model files not found: {missing_models}")
        return
    
    print(f"Processing Holdout Set: {holdout_file_path}")
    
    # Load the holdout dataset
    data_predict = pd.read_csv(holdout_file_path)
    
    # Strip units from column names
    data_predict.columns = [strip_units(col) for col in data_predict.columns]
    
    # Prepare features by dropping unnecessary columns
    features = data_predict.drop(['E_RS - E_ZB', '# Material'], axis=1)
    features_dict = {col: features[col].values for col in features.columns}
    
    # Load models and predict
    models = [load_model(file) for file in model_files]
    predictions = [model.eval_many(features_dict) for model in models]
    
    # Calculate average predictions and standard deviation
    average_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)
    
    # Add predictions and error to the DataFrame
    data_predict['Average Predicted E_RS - E_ZB'] = average_predictions
    data_predict['Error'] = data_predict['E_RS - E_ZB'] - data_predict['Average Predicted E_RS - E_ZB']
    data_predict['Std Deviation'] = std_predictions
    
    # Print standard deviation values for debugging
    print(f"\nStandard Deviation Values for Holdout Set {it_number}:")
    print(data_predict[['# Material', 'E_RS - E_ZB', 'Average Predicted E_RS - E_ZB', 'Std Deviation']])
    
    # Plot results
    visualize_predictions(data_predict, it_number)

def visualize_predictions(data_predict, iteration):
    """
    Plots actual vs predicted values for the holdout set.
    """
    plt.figure(figsize=(12, 7))
    plt.scatter(data_predict['E_RS - E_ZB'], data_predict['Average Predicted E_RS - E_ZB'], color='blue', alpha=0.5, label='Predicted vs Actual')
    plt.errorbar(data_predict['E_RS - E_ZB'], data_predict['Average Predicted E_RS - E_ZB'], yerr=data_predict['Std Deviation'], fmt='o', color='gray', alpha=0.5, label='Error Bars')
    plt.plot([data_predict['E_RS - E_ZB'].min(), data_predict['E_RS - E_ZB'].max()], 
             [data_predict['E_RS - E_ZB'].min(), data_predict['E_RS - E_ZB'].max()], 'r--', label='Perfect Prediction')
    plt.title(f'Actual vs Predicted E_RS - E_ZB with Std Dev for Holdout Set {iteration}')
    plt.xlabel('Actual E_RS - E_ZB')
    plt.ylabel('Predicted E_RS - E_ZB')
    plt.legend()
    plt.grid(True)
    plt.show()

def process_all_holdout_sets(base_path, holdout_path, num_iterations=5):
    """
    Processes and plots predictions for all holdout sets from 1 to num_iterations.
    """
    for it_number in range(1, num_iterations + 1):
        process_file_set(it_number, base_path, holdout_path)

# Example usage
models_base_path = "/home/u28/rahulbangad/models/rowsmodel/"
holdout_sets_path = "/home/u28/rahulbangad/splits/"
process_all_holdout_sets(base_path=models_base_path, holdout_path=holdout_sets_path, num_iterations=5)


