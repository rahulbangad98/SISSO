#holdoutCreation

import pandas as pd
import numpy as np
import os

# Define the path where files will be saved
#save_path = '/Users/rahulbangad/Desktop/TryMe/'
save_path = '/home/u28/rahulbangad/holdOuts/'
# Check if the directory exists, if not, create it
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load the data from CSV file
#data = pd.read_csv('/Users/rahulbangad/Downloads/data copy.csv')
data = pd.read_csv('/home/u28/rahulbangad/data.csv')
# Total number of holdout sets
num_holdouts = 5

# Total number of iterations per holdout set
num_iterations = 10

# Proportion of rows to remove per iteration
fraction_to_remove = 0.25

# Proportion of rows to act as a holdout set
holdout_fraction = 0.2  # Example: 20% of the data as holdout set

for holdout in range(num_holdouts):
    # Determine the number of rows to reserve for the holdout set
    holdout_size = int(len(data) * holdout_fraction)
    
    # Create a random sample of row indices for the holdout set
    holdout_indices = np.random.choice(data.index, holdout_size, replace=False)
    holdout_set = data.loc[holdout_indices]
    working_set = data.drop(holdout_indices)
    
    # Save the holdout set to a CSV file in the specified directory
    holdout_set.to_csv(f'{save_path}holdout_set_{holdout+1}.csv', index=False)
    
    for i in range(num_iterations):
        # Determine the number of rows to drop from the working set
        num_rows_to_drop = int(len(working_set) * fraction_to_remove)
        
        # Create a random sample of row indices to drop
        drop_indices = np.random.choice(working_set.index, num_rows_to_drop, replace=False)
        
        # Drop the rows
        modified_data = working_set.drop(drop_indices)
        
        # Save the modified data to a new CSV file in the specified directory
        modified_data.to_csv(f'{save_path}modified_data_holdout_{holdout+1}_iteration_{i+1}.csv', index=False)

        print(f'Holdout {holdout+1}, Iteration {i+1}: Data saved to {save_path}modified_data_holdout_{holdout+1}_iteration_{i+1}.csv')


#Updating sisso.json code

import json
import os

# Path to the original sisso.json file
#sisso_file_path = "/Users/rahulbangad/Downloads/sisso.json"
sisso_file_path = "/home/u28/rahulbangad/sisso.json"
# Load the original sisso.json content
with open(sisso_file_path, 'r') as file:
    sisso_content = json.load(file)

# Base path where modified data files are stored
#base_save_path = '/Users/rahulbangad/Downloads/sissoData/'
base_save_path = '/home/u28/rahulbangad/sissoDataNew/'
# Ensure the base save path directory exists
if not os.path.exists(base_save_path):
    os.makedirs(base_save_path)

# Parameters
num_holdouts = 5
num_iterations = 10

# Loop through each holdout and iteration to create new sisso.json files
for holdout in range(1, num_holdouts + 1):
    for iteration in range(1, num_iterations + 1):
        # Define the iteration specific folder path
        iteration_folder_path = os.path.join(base_save_path, f'Holdout_{holdout}', f'Iteration_{iteration}')
        
        # Ensure the iteration specific folder exists
        if not os.path.exists(iteration_folder_path):
            os.makedirs(iteration_folder_path)

        # Update the data_file path
        #sisso_content["data_file"] = os.path.join(iteration_folder_path, f'modified_data_holdout_{holdout}_iteration_{iteration}.csv')
        sisso_content["data_file"] = f'modified_data_holdout_{holdout}_iteration_{iteration}.csv'

        #sisso_content["data_file"] = modified_data_holdout_{holdout}_iteration_{iteration}.csv
        # Define the new sisso.json file path
        new_sisso_file_path = os.path.join(iteration_folder_path, 'sisso.json')
        
        # Save the updated sisso.json file
        with open(new_sisso_file_path, 'w') as new_file:
            json.dump(sisso_content, new_file, indent=4)

        print(f'Updated sisso.json saved to {new_sisso_file_path}')

#model creation


import os
import shutil
from sissopp import Inputs, FeatureSpace, SISSORegressor  # Ensure you have these classes implemented
import json

# Base directory for SISSO data
base_sisso_folder_path = '/home/u28/rahulbangad/sissoDataNew/'

# Base directory for saving models
models_base_path = '/home/u28/rahulbangad/models/'

additional_files_path = '/home/u28/rahulbangad/holdOuts'

# Loop through each subdirectory in the base directory
for holdout_dir in os.listdir(base_sisso_folder_path):
    holdout_path = os.path.join(base_sisso_folder_path, holdout_dir)
    # Check if it's a directory
    if os.path.isdir(holdout_path):
        print(f"Processing holdout directory: {holdout_dir}")
        # Create holdout model directory if it doesn't exist
        holdout_model_path = os.path.join(models_base_path, holdout_dir)
        os.makedirs(holdout_model_path, exist_ok=True)

        # Iterate through each iteration folder within the holdout directory
        for iteration_dir in os.listdir(holdout_path):
            iteration_path = os.path.join(holdout_path, iteration_dir)
            if os.path.isdir(iteration_path):
                print(f"Processing iteration directory: {iteration_dir}")
                sisso_file_path = os.path.join(iteration_path, 'sisso.json')
                # Check if the sisso.json file exists in this directory
                if os.path.exists(sisso_file_path):
                    print(f"Found sisso.json in: {iteration_dir}")
                    # Create a unique directory for this iteration's model
                    iteration_model_path = os.path.join(holdout_model_path, iteration_dir)
                    print(iteration_model_path)
                    os.makedirs(iteration_model_path, exist_ok=True)

                    # Change the working directory to the unique iteration directory
                    original_working_directory = os.getcwd()
                    print(original_working_directory)
                    os.chdir(iteration_model_path)
                    
                    # Copy the sisso.json file from iteration_path to iteration_model_path
                    destination_file_path = os.path.join(iteration_model_path, 'sisso.json')
                    shutil.copy(sisso_file_path, destination_file_path)
                    print(f"Copied {sisso_file_path} to {destination_file_path}")


                    # Copy the additional CSV file from TryMe directory to iteration_model_path
                    additional_file_name = f'modified_data_{holdout_dir.lower()}_{iteration_dir.lower()}.csv'
                    additional_file_source_path = os.path.join(additional_files_path, additional_file_name)
                    additional_file_destination_path = os.path.join(iteration_model_path, additional_file_name)
                    if os.path.exists(additional_file_source_path):
                        shutil.copy(additional_file_source_path, additional_file_destination_path)
                        print(f"Copied {additional_file_source_path} to {additional_file_destination_path}")
                    else:
                        print(f"Additional file {additional_file_source_path} does not exist")



                    # Load inputs from the copied sisso.json
                    inputs = Inputs(destination_file_path)
                    feature_space = FeatureSpace(inputs)
                    sisso = SISSORegressor(inputs, feature_space)

                    # Fit the model
                    sisso.fit()

                    # Change back to the original working directory
                    os.chdir(original_working_directory)
                else:
                    print(f"sisso.json not found in: {iteration_dir}")
print(f"All models saved in: {models_base_path}")

#copying holdout files which will be used for prediction



import shutil
import os
import glob

# Source directory containing the holdout files
source_directory = '/home/u28/rahulbangad/holdOuts/'

# Destination directory where holdout set folders are located
destination_directory = '/home/u28/rahulbangad/models/'

# Find all files matching the pattern holdout_set_{holdoutnumber}.csv
file_pattern = os.path.join(source_directory, 'holdout_set_*.csv')
holdout_files = glob.glob(file_pattern)

# Copy each file to its respective Holdout_{holdoutnumber} folder in the destination directory
for file in holdout_files:
    try:
        # Extract the holdout number from the filename
        filename = os.path.basename(file)
        holdout_number = filename.split('_')[2].split('.')[0]
        
        # Determine the holdout set folder in the destination directory
        holdout_folder = os.path.join(destination_directory, f'Holdout_{holdout_number}')
        
        # Ensure the holdout set folder exists
        os.makedirs(holdout_folder, exist_ok=True)
        
        # Copy the file to the holdout set folder
        shutil.copy(file, holdout_folder)
        print(f"Successfully copied {file} to {holdout_folder}")
    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred while copying {file}: {e}")

#Prediction 

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from sissopp.postprocess.load_models import load_model

def strip_units(column_name):
    """ Remove text within parentheses (assuming these are units) from column names. """
    return re.sub(r'\s*\([^)]*\)', '', column_name).strip()

def get_data_dct(df):
    """Convert a dataframe into a dict with numpy arrays of type float for numeric columns."""
    dct = {}
    for key in df.columns:
        if key == '# Material':  # Skip conversion for the Material column
            dct[key] = df[key].values
        else:
            try:
                # Attempt to convert column to floats
                dct[key] = df[key].astype(float).to_numpy()
            except ValueError as e:
                # Handle columns that contain non-convertible strings
                print(f"Error converting {key}: {e}")
                # Replace non-convertible values with NaN
                dct[key] = pd.to_numeric(df[key], errors='coerce').to_numpy()
    return dct

def process_file_set(it_number, base_path):
    # Generate the file paths
    #model_files = [os.path.join(base_path, f"models/it{it_number}/{i}/train_dim_4_model_0.dat") for i in range(1, 11)]
    model_files = [os.path.join(base_path, f"models/Holdout_{it_number}/Iteration_{i}/models/train_dim_4_model_0.dat") for i in range(1, 11)]
    file_path = os.path.join(base_path, f"models/Holdout_{it_number}/holdout_set_{it_number}.csv")
    print(file_path)
    print(model_files)
    # Load all models
    models = [load_model(file) for file in model_files]

    # Load the test data
    data_predict = pd.read_csv(file_path)

    # Clean column names to remove any potential whitespace and units
    data_predict.columns = [strip_units(col) for col in data_predict.columns]

    # Assuming 'E_RS - E_ZB (eV)' is the target variable and should not be included in the features
    features = data_predict.drop(['EA_A', '# Material'], axis=1)  # Exclude Material from features if it exists

    # Convert the DataFrame of features into a dictionary using the get_data_dct function
    features_dict = get_data_dct(features)

    # Use each model to predict and store predictions
    predictions = [model.eval_many(features_dict) for model in models]

    # Calculate the average of the predictions from the models
    average_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)

    # Add the average predictions back to the DataFrame
    data_predict['Average Predicted EA_A'] = average_predictions
    data_predict['Error'] = data_predict['EA_A'] - data_predict['Average Predicted EA_A']
    data_predict['Std Deviation'] = std_predictions

    # Calculate statistics for error
    std_deviation = data_predict['Error'].std()
    mean_error = data_predict['Error'].mean()

    # Define a threshold for large errors
    large_error_threshold = mean_error + std_deviation

    # Filter the dataset for large errors
    large_errors = data_predict[data_predict['Error'].abs() > large_error_threshold]

    # Print the materials with large prediction errors
    print(f"Materials with Large Prediction Errors for iteration {it_number}:")
    print(large_errors[['# Material', 'EA_A', 'Average Predicted EA_A', 'Error']])

    # Plotting the actual vs. predicted values along with error bars for standard deviation
    plt.figure(figsize=(12, 7))

    # Scatter plot for actual vs. predicted values
    scatter = plt.scatter(data_predict['EA_A'], data_predict['Average Predicted EA_A'], s=50, color='blue', alpha=0.5, label='Predicted vs Actual')
    plt.title(f'Comparison of Actual and Predicted Values with Std Dev Values for iteration {it_number}')
    plt.xlabel('Actual EA_A')
    plt.ylabel('Predicted EA_A')
    plt.grid(True)

    # Adding a line of perfect prediction
    max_value = max(data_predict['EA_A'].max(), data_predict['Average Predicted EA_A'].max())
    min_value = min(data_predict['EA_A'].min(), data_predict['Average Predicted EA_A'].min())
    plt.plot([min_value, max_value], [min_value, max_value], 'r--', label='Perfect Prediction')

    # Overlaying error bars on the scatter plot using the standard deviation
    plt.errorbar(data_predict['EA_A'], data_predict['Average Predicted EA_A'], yerr=std_predictions, fmt='o', color='gray', alpha=0.5, label='Std Dev Error Bars')

    # Annotating standard deviation values
    for i, txt in enumerate(data_predict['Std Deviation']):
        plt.annotate(f"{txt:.2f}",
                     (data_predict['EA_A'].iloc[i], data_predict['Average Predicted EA_A'].iloc[i]),
                     textcoords="offset points",
                     xytext=(0, 10 + 5 * i % 3),
                     ha='center',
                     color='darkred',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', edgecolor='black', alpha=0.5))

    plt.legend()
    plt.show()

# Base path for files
#base_path = '/home/u28/rahulbangad'
base_path = '/home/u28/rahulbangad'
# Iterate over different iterations
for it_number in range(1, 6):  # Adjust the range as needed for holdouts, etc.
    process_file_set(it_number, base_path)
