import os
import shutil
from sissopp import Inputs, FeatureSpace, SISSORegressor  # Ensure you have these classes implemented
import json
# Base directory for SISSO data
base_sisso_folder_path = '/home/u28/rahulbangad/sissoData/'

# Base directory for saving models
models_base_path = '/home/u28/rahulbangad/models/'

# Loop through each subdirectory in the base directory
for holdout_dir in os.listdir(base_sisso_folder_path):
    holdout_path = os.path.join(base_sisso_folder_path, holdout_dir)
    # Check if it's a directory
    if os.path.isdir(holdout_path):
        # Create holdout model directory if it doesn't exist
        holdout_model_path = os.path.join(models_base_path, holdout_dir)
        os.makedirs(holdout_model_path, exist_ok=True)

        # Iterate through each iteration folder within the holdout directory
        for iteration_dir in os.listdir(holdout_path):
            iteration_path = os.path.join(holdout_path, iteration_dir)
            if os.path.isdir(iteration_path):
                r = f'sisso.json'
                print(r)
                sisso_file_path = os.path.join(iteration_path, 'sisso.json')
                # Check if the sisso.json file exists in this directory
                if os.path.exists(sisso_file_path):
                    # Create a unique directory for this iteration's model
                    iteration_model_path = os.path.join(holdout_model_path, iteration_dir)
                    os.makedirs(iteration_model_path, exist_ok=True)

                    # Change the working directory to the unique iteration directory
                    original_working_directory = os.getcwd()
                    os.chdir(iteration_model_path)

                    # Load inputs from sisso.json
                    inputs = Inputs(r)
                    feature_space = FeatureSpace(inputs)
                    sisso = SISSORegressor(inputs, feature_space)

                    # Fit the model
                    sisso.fit()

                    # Change back to the original working directory
                    os.chdir(original_working_directory)
