import os
import random
import json
import shutil
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sissopp import Inputs, FeatureSpace, SISSORegressor
from sissopp.postprocess.load_models import load_model

def create_holdout_sets(data_path, save_path, num_holdouts, num_iterations, fraction_to_remove, holdout_fraction):
    """
    Creates holdout sets and modified data for iterations.
    """
    # Load the data
    data = pd.read_csv(data_path)
    
    for holdout in range(num_holdouts):
        # Determine the number of rows to reserve for the holdout set
        holdout_size = int(len(data) * holdout_fraction)
        
        # Create a random sample of row indices for the holdout set
        holdout_indices = np.random.choice(data.index, holdout_size, replace=False)
        holdout_set = data.loc[holdout_indices]
        working_set = data.drop(holdout_indices)
        
        # Save the holdout set
        holdout_set.to_csv(f'{save_path}holdout_set_{holdout+1}.csv', index=False)
        
        for i in range(num_iterations):
            # Determine the number of rows to drop from the working set
            num_rows_to_drop = int(len(working_set) * fraction_to_remove)
            
            # Create a random sample of row indices to drop
            drop_indices = np.random.choice(working_set.index, num_rows_to_drop, replace=False)
            
            # Drop the rows
            modified_data = working_set.drop(drop_indices)
            
            # Save the modified data
            modified_data.to_csv(f'{save_path}modified_data_holdout_{holdout+1}_iteration_{i+1}.csv', index=False)
            
            print(f'Holdout {holdout+1}, Iteration {i+1}: Data saved to {save_path}modified_data_holdout_{holdout+1}_iteration_{i+1}.csv')

def update_sisso_json(original_sisso_path, base_save_path, num_holdouts, num_iterations):
    """
    Updates sisso.json files for each holdout and iteration.
    """
    # Load the original sisso.json content
    with open(original_sisso_path, 'r') as file:
        sisso_content = json.load(file)
    
    for holdout in range(1, num_holdouts + 1):
        for iteration in range(1, num_iterations + 1):
            # Define the iteration-specific folder path
            iteration_folder_path = os.path.join(base_save_path, f'Holdout_{holdout}', f'Iteration_{iteration}')
            
            # Ensure the iteration-specific folder exists
            os.makedirs(iteration_folder_path, exist_ok=True)
    
            # Update the data_file path in sisso.json
            sisso_content["data_file"] = f'modified_data_holdout_{holdout}_iteration_{iteration}.csv'
    
            # Define the new sisso.json file path
            new_sisso_file_path = os.path.join(iteration_folder_path, 'sisso.json')
            
            # Save the updated sisso.json file
            with open(new_sisso_file_path, 'w') as new_file:
                json.dump(sisso_content, new_file, indent=4)
    
            print(f'Updated sisso.json saved to {new_sisso_file_path}')
#1
# def create_models(base_sisso_folder_path, models_base_path, additional_files_path, num_columns_to_remove):
#     """
#     Creates models for each holdout and iteration.
#     """
#     # Loop through each holdout directory
#     for holdout_dir in os.listdir(base_sisso_folder_path):
#         holdout_path = os.path.join(base_sisso_folder_path, holdout_dir)
#         if os.path.isdir(holdout_path):
#             print(f"Processing holdout directory: {holdout_dir}")
#             holdout_model_path = os.path.join(models_base_path, holdout_dir)
#             os.makedirs(holdout_model_path, exist_ok=True)
    
#             # Iterate through each iteration folder within the holdout directory
#             for iteration_dir in os.listdir(holdout_path):
#                 iteration_path = os.path.join(holdout_path, iteration_dir)
#                 if os.path.isdir(iteration_path):
#                     print(f"Processing iteration directory: {iteration_dir}")
#                     sisso_file_path = os.path.join(iteration_path, 'sisso.json')
#                     if os.path.exists(sisso_file_path):
#                         print(f"Found sisso.json in: {iteration_dir}")
#                         iteration_model_path = os.path.join(holdout_model_path, iteration_dir)
#                         os.makedirs(iteration_model_path, exist_ok=True)
    
#                         # Change to the unique iteration directory
#                         original_working_directory = os.getcwd()
#                         os.chdir(iteration_model_path)
                        
#                         # Copy the sisso.json file
#                         destination_file_path = os.path.join(iteration_model_path, 'sisso.json')
#                         shutil.copy(sisso_file_path, destination_file_path)
    
#                         # Process the additional CSV file
#                         additional_file_name = f'modified_data_{holdout_dir.lower()}_{iteration_dir.lower()}.csv'
#                         additional_file_source_path = os.path.join(additional_files_path, additional_file_name)
#                         additional_file_destination_path = os.path.join(iteration_model_path, additional_file_name)
#                         if os.path.exists(additional_file_source_path):
#                             df = pd.read_csv(additional_file_source_path)
#                             df.columns = df.columns.str.strip().str.lower()
    
#                             if num_columns_to_remove < 1 or num_columns_to_remove >= len(df.columns):
#                                 raise ValueError("Invalid number of columns to drop. There must be at least one column remaining.")
    
#                             # Randomly drop columns
#                             columns_to_drop = random.sample(list(df.columns), num_columns_to_remove)
#                             df.drop(columns=columns_to_drop, inplace=True)
                            
#                             # Save the modified data
#                             df.to_csv(additional_file_destination_path, index=False)
#                             print(f"Processed and saved {additional_file_destination_path}")
#                         else:
#                             print(f"Additional file {additional_file_source_path} does not exist")
    
#                         # Load inputs and fit the model
#                         inputs = Inputs(destination_file_path)
#                         feature_space = FeatureSpace(inputs)
#                         sisso = SISSORegressor(inputs, feature_space)
    
#                         sisso.fit()
    
#                         # Change back to the original working directory
#                         os.chdir(original_working_directory)
#                     else:
#                         print(f"sisso.json not found in: {iteration_dir}")
#     print(f"All models saved in: {models_base_path}")

#2
# def create_models(base_sisso_folder_path, models_base_path, additional_files_path, num_columns_to_remove):
#     """
#     Creates models for each holdout and iteration.
#     """
#     # Loop through each holdout directory
#     for holdout_dir in os.listdir(base_sisso_folder_path):
#         holdout_path = os.path.join(base_sisso_folder_path, holdout_dir)
#         if os.path.isdir(holdout_path):
#             print(f"Processing holdout directory: {holdout_dir}")
#             holdout_model_path = os.path.join(models_base_path, holdout_dir)
#             os.makedirs(holdout_model_path, exist_ok=True)
    
#             # Iterate through each iteration folder within the holdout directory
#             for iteration_dir in os.listdir(holdout_path):
#                 iteration_path = os.path.join(holdout_path, iteration_dir)
#                 if os.path.isdir(iteration_path):
#                     print(f"Processing iteration directory: {iteration_dir}")
#                     sisso_file_path = os.path.join(iteration_path, 'sisso.json')
#                     if os.path.exists(sisso_file_path):
#                         print(f"Found sisso.json in: {iteration_dir}")
#                         iteration_model_path = os.path.join(holdout_model_path, iteration_dir)
#                         os.makedirs(iteration_model_path, exist_ok=True)
    
#                         # Change to the unique iteration directory
#                         original_working_directory = os.getcwd()  # Save the original working directory
#                         os.chdir(iteration_model_path)  # Change to the iteration model path
                        
#                         # Copy the sisso.json file
#                         destination_file_path = os.path.join(iteration_model_path, 'sisso.json')
#                         shutil.copy(sisso_file_path, destination_file_path)
#                         print(f"Copied {sisso_file_path} to {destination_file_path}")
    
#                         # Process the additional CSV file
#                         additional_file_name = f'modified_data_{holdout_dir.lower()}_{iteration_dir.lower()}.csv'
#                         additional_file_source_path = os.path.join(additional_files_path, additional_file_name)
#                         additional_file_destination_path = os.path.join(iteration_model_path, additional_file_name)
#                         print(f"Processing additional CSV file: {additional_file_source_path}")
                        
#                         if os.path.exists(additional_file_source_path):
#                             df = pd.read_csv(additional_file_source_path)
#                             df.columns = df.columns.str.strip().str.lower()  # Clean the column names
#                             print(f"Columns before dropping: {df.columns}")
    
#                             if num_columns_to_remove < 1 or num_columns_to_remove >= len(df.columns):
#                                 raise ValueError("Invalid number of columns to drop. There must be at least one column remaining.")
    
#                             # Randomly drop columns (ensure at least 1 column remains)
#                             columns_to_drop = random.sample(list(df.columns), num_columns_to_remove)
#                             df.drop(columns=columns_to_drop, inplace=True)
#                             print(f"Columns after dropping: {df.columns}")
                            
#                             # Save the modified data
#                             df.to_csv(additional_file_destination_path, index=False)
#                             print(f"Processed and saved {additional_file_destination_path}")
#                         else:
#                             print(f"Additional file {additional_file_source_path} does not exist")
    
#                         # Load inputs and fit the model
#                         print(f"Loading inputs from: {destination_file_path}")
#                         with open(destination_file_path, 'r') as file:
#                             sisso_content = json.load(file)
#                             print("sisso.json content:", sisso_content)
#                         inputs = Inputs(destination_file_path)
#                         feature_space = FeatureSpace(inputs)
#                         sisso = SISSORegressor(inputs, feature_space)
    
#                         print("Fitting the model...")
#                         sisso.fit()  # Fit the model
    
#                         # Change back to the original working directory
#                         os.chdir(original_working_directory)
#                         print(f"Returned to original working directory: {original_working_directory}")
#                     else:
#                         print(f"sisso.json not found in: {iteration_dir}")
#     print(f"All models saved in: {models_base_path}")

# #final

def create_models(base_sisso_folder_path, models_base_path, additional_files_path, num_columns_to_remove):
    """
    Creates models for each holdout and iteration.
    """
    # The property key that should never be dropped
    property_key_column = "E_RS - E_ZB"
    materials_column = "# Material"

    # Loop through each holdout directory
    for holdout_dir in os.listdir(base_sisso_folder_path):
        holdout_path = os.path.join(base_sisso_folder_path, holdout_dir)
        if os.path.isdir(holdout_path):
            print(f"Processing holdout directory: {holdout_dir}")
            holdout_model_path = os.path.join(models_base_path, holdout_dir)
            os.makedirs(holdout_model_path, exist_ok=True)
    
            # Iterate through each iteration folder within the holdout directory
            for iteration_dir in os.listdir(holdout_path):
                iteration_path = os.path.join(holdout_path, iteration_dir)
                if os.path.isdir(iteration_path):
                    print(f"Processing iteration directory: {iteration_dir}")
                    sisso_file_path = os.path.join(iteration_path, 'sisso.json')
                    if os.path.exists(sisso_file_path):
                        print(f"Found sisso.json in: {iteration_dir}")
                        iteration_model_path = os.path.join(holdout_model_path, iteration_dir)
                        os.makedirs(iteration_model_path, exist_ok=True)
    
                        # Change to the unique iteration directory
                        original_working_directory = os.getcwd()  # Save the original working directory
                        os.chdir(iteration_model_path)  # Change to the iteration model path
                        
                        # Copy the sisso.json file
                        destination_file_path = os.path.join(iteration_model_path, 'sisso.json')
                        shutil.copy(sisso_file_path, destination_file_path)
                        print(f"Copied {sisso_file_path} to {destination_file_path}")
    
                        # Process the additional CSV file
                        additional_file_name = f'modified_data_{holdout_dir.lower()}_{iteration_dir.lower()}.csv'
                        additional_file_source_path = os.path.join(additional_files_path, additional_file_name)
                        additional_file_destination_path = os.path.join(iteration_model_path, additional_file_name)
                        print(f"Processing additional CSV file: {additional_file_source_path}")
                        
                        if os.path.exists(additional_file_source_path):
                            df = pd.read_csv(additional_file_source_path)
                            df.columns = df.columns.str.strip()  # Clean the column names
                            
                            df = df.apply(pd.to_numeric, errors='coerce')  # Convert invalid values to NaN
                            

                            # Ensure the "e_rs - e_zb" column is not dropped
                            property_key_column_1 = property_key_column.strip()
                            materials_column_1 = materials_column.strip()
                            print("this is property key")
                            print(property_key_column_1)
                            non_critical_columns = [col for col in df.columns if col not in [property_key_column_1, materials_column_1]]
                            #non_critical_columns = [col for col in df.columns if col != property_key_column_1]
                            print(non_critical_columns)
                            if num_columns_to_remove >= len(non_critical_columns):
                                raise ValueError("Invalid number of columns to drop. There must be at least one non-critical column remaining.")
    
                            # Randomly drop columns, ensuring the "e_rs - e_zb" column is not dropped
                            columns_to_drop = random.sample(non_critical_columns, num_columns_to_remove)
                            df.drop(columns=columns_to_drop, inplace=True)
                            print(f"Columns after dropping: {df.columns}")
                            
                            # Save the modified data
                            df.to_csv(additional_file_destination_path, index=False)
                            print(f"Processed and saved {additional_file_destination_path}")
                        else:
                            print(f"Additional file {additional_file_source_path} does not exist")
    
                        # Load inputs and fit the model
                        print(f"Loading inputs from: {destination_file_path}")
                        inputs = Inputs(destination_file_path)
                        feature_space = FeatureSpace(inputs)
                        sisso = SISSORegressor(inputs, feature_space)
    
                        print("Fitting the model...")
                        sisso.fit()  # Fit the model
    
                        # Change back to the original working directory
                        os.chdir(original_working_directory)
                        print(f"Returned to original working directory: {original_working_directory}")
                    else:
                        print(f"sisso.json not found in: {iteration_dir}")
    print(f"All models saved in: {models_base_path}")

#try



# def create_models(base_sisso_folder_path, models_base_path, additional_files_path, num_columns_to_remove):
#     """
#     Creates models for each holdout and iteration, ensuring that the "e_rs - e_zb" column is never dropped.
#     """
#     # Property key that should never be dropped
#     property_key_column = "E_RS - E_ZB"
    
#     # Loop through each holdout directory
#     for holdout_dir in os.listdir(base_sisso_folder_path):
#         holdout_path = os.path.join(base_sisso_folder_path, holdout_dir)
#         if os.path.isdir(holdout_path):
#             print(f"Processing holdout directory: {holdout_dir}")
#             holdout_model_path = os.path.join(models_base_path, holdout_dir)
#             os.makedirs(holdout_model_path, exist_ok=True)
    
#             # Iterate through each iteration folder within the holdout directory
#             for iteration_dir in os.listdir(holdout_path):
#                 iteration_path = os.path.join(holdout_path, iteration_dir)
#                 if os.path.isdir(iteration_path):
#                     print(f"Processing iteration directory: {iteration_dir}")
#                     sisso_file_path = os.path.join(iteration_path, 'sisso.json')
#                     if os.path.exists(sisso_file_path):
#                         print(f"Found sisso.json in: {iteration_dir}")
#                         iteration_model_path = os.path.join(holdout_model_path, iteration_dir)
#                         os.makedirs(iteration_model_path, exist_ok=True)
    
#                         # Change to the unique iteration directory
#                         original_working_directory = os.getcwd()
#                         os.chdir(iteration_model_path)
                        
#                         # Copy the sisso.json file
#                         destination_file_path = os.path.join(iteration_model_path, 'sisso.json')
#                         shutil.copy(sisso_file_path, destination_file_path)
    
#                         # Process the additional CSV file
#                         additional_file_name = f'modified_data_{holdout_dir.lower()}_{iteration_dir.lower()}.csv'
#                         additional_file_source_path = os.path.join(additional_files_path, additional_file_name)
#                         additional_file_destination_path = os.path.join(iteration_model_path, additional_file_name)
                        
#                         if os.path.exists(additional_file_source_path):
#                             # Load the data and clean column names
#                             try:
#                                 df = pd.read_csv(additional_file_source_path)
                                
#                                 # Print the columns before converting to lowercase
#                                 print(f"Columns before converting to lowercase for iteration {iteration_dir}: {df.columns.tolist()}")
                                
#                                 # Convert column names to lowercase
#                                 df.columns = df.columns.str.strip().str.lower()
                                
#                                 # Print the columns after converting to lowercase
#                                 print(f"Columns after converting to lowercase for iteration {iteration_dir}: {df.columns.tolist()}")
                                
#                                 # Ensure all numeric columns are valid and handle non-numeric values
#                                 df = df.apply(pd.to_numeric, errors='coerce')  # Convert invalid values to NaN
#                                 df.dropna(inplace=True)  # Drop rows with NaN values

#                                 if property_key_column.lower() not in df.columns:
#                                     raise ValueError(f"Critical column '{property_key_column}' is missing from the data!")

#                                 non_critical_columns = [col for col in df.columns if col != property_key_column.lower()]

#                                 # Check if the number of columns to remove is valid
#                                 if num_columns_to_remove < 1 or num_columns_to_remove >= len(non_critical_columns):
#                                     raise ValueError("Invalid number of columns to drop. There must be at least one non-critical column remaining.")

#                                 # Print the columns that will be dropped
#                                 columns_to_drop = random.sample(non_critical_columns, num_columns_to_remove)
#                                 print(f"Columns to drop for iteration {iteration_dir}: {columns_to_drop}")

#                                 # Drop the selected columns
#                                 df.drop(columns=columns_to_drop, inplace=True)

#                                 # Print the columns after dropping
#                                 print(f"Columns after dropping for iteration {iteration_dir}: {df.columns.tolist()}")
                                
#                                 # Save the modified data with lowercased column names
#                                 df.to_csv(additional_file_destination_path, index=True)
#                                 print(f"Processed and saved {additional_file_destination_path}")
#                             except Exception as e:
#                                 print(f"Error processing CSV file {additional_file_source_path}: {e}")
#                         else:
#                             print(f"Additional file {additional_file_source_path} does not exist")
    
#                         # Load inputs and fit the model
#                         inputs = Inputs(destination_file_path)
#                         feature_space = FeatureSpace(inputs)
#                         sisso = SISSORegressor(inputs, feature_space)
    
#                         # Fit the model
#                         sisso.fit()
    
#                         # Change back to the original working directory
#                         os.chdir(original_working_directory)
#                     else:
#                         print(f"sisso.json not found in: {iteration_dir}")
#     print(f"All models saved in: {models_base_path}")



def copy_holdout_files(source_directory, destination_directory):
    """
    Copies holdout files to their respective model directories.
    """
    # Find all holdout files
    file_pattern = os.path.join(source_directory, 'holdout_set_*.csv')
    holdout_files = glob.glob(file_pattern)
    
    # Copy each holdout file to the corresponding directory
    for file in holdout_files:
        try:
            filename = os.path.basename(file)
            holdout_number = filename.split('_')[2].split('.')[0]
            holdout_folder = os.path.join(destination_directory, f'Holdout_{holdout_number}')
            os.makedirs(holdout_folder, exist_ok=True)
            shutil.copy(file, holdout_folder)
            print(f"Successfully copied {file} to {holdout_folder}")
        except FileNotFoundError as fnf_error:
            print(f"Error: {fnf_error}")
        except Exception as e:
            print(f"An unexpected error occurred while copying {file}: {e}")

def strip_units(column_name):
    """ Remove text within parentheses (assuming these are units) from column names. """
    return re.sub(r'\s*\([^)]*\)', '', column_name).strip()

def get_data_dct(df):
    """Convert a dataframe into a dict with numpy arrays of type float for numeric columns."""
    dct = {}
    for key in df.columns:
        if key == '# material':
            dct[key] = df[key].values
        else:
            try:
                dct[key] = df[key].astype(float).to_numpy()
            except ValueError as e:
                print(f"Error converting {key}: {e}")
                dct[key] = pd.to_numeric(df[key], errors='coerce').to_numpy()
    return dct

def process_file_set(it_number, base_path):
    """
    Processes and evaluates models for a given holdout set.
    """
    # Generate file paths
    model_files = [os.path.join(base_path, f"models/Holdout_{it_number}/Iteration_{i}/models/train_dim_4_model_0.dat") for i in range(1, num_iterations + 1)]
    file_path = os.path.join(base_path, f"models/Holdout_{it_number}/holdout_set_{it_number}.csv")
    print(f"Processing Holdout Set: {file_path}")
    
    # Load models
    models = [load_model(file) for file in model_files]
    
    # Load test data
    data_predict = pd.read_csv(file_path)
    data_predict.columns = [strip_units(col) for col in data_predict.columns]
    
    # Prepare features
    features = data_predict.drop(['E_RS - E_ZB', '# Material'], axis=1)
    features_dict = get_data_dct(features)
    
    # Predict using models
    predictions = [model.eval_many(features_dict) for model in models]
    
    # Calculate average and standard deviation
    average_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)
    
    # Add predictions to DataFrame
    data_predict['Average Predicted EA_A'] = average_predictions
    data_predict['Error'] = data_predict['EA_A'] - data_predict['Average Predicted EA_A']
    data_predict['Std Deviation'] = std_predictions
    
    # Calculate error statistics
    std_deviation = data_predict['Error'].std()
    mean_error = data_predict['Error'].mean()
    large_error_threshold = mean_error + std_deviation
    
    # Identify large errors
    large_errors = data_predict[data_predict['Error'].abs() > large_error_threshold]
    
    # Print errors
    print(f"All Prediction Errors for iteration {it_number}:")
    print(data_predict[['# Material', 'EA_A', 'Average Predicted EA_A', 'Error']])
    
    # Plot results
    plt.figure(figsize=(12, 7))
    plt.scatter(data_predict['EA_A'], data_predict['Average Predicted EA_A'], s=50, color='blue', alpha=0.5, label='Predicted vs Actual')
    plt.title(f'Comparison of Actual and Predicted Values with Std Dev for iteration {it_number}')
    plt.xlabel('Actual EA_A')
    plt.ylabel('Predicted EA_A')
    plt.grid(True)
    
    # Perfect prediction line
    max_value = max(data_predict['EA_A'].max(), data_predict['Average Predicted EA_A'].max())
    min_value = min(data_predict['EA_A'].min(), data_predict['Average Predicted EA_A'].min())
    plt.plot([min_value, max_value], [min_value, max_value], 'r--', label='Perfect Prediction')
    
    # Error bars
    plt.errorbar(data_predict['EA_A'], data_predict['Average Predicted EA_A'], yerr=std_predictions, fmt='o', color='gray', alpha=0.5, label='Std Dev Error Bars')
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Ask user for inputs
    data_path = input("Enter the path to your data CSV file (e.g., '/home/user/data.csv'): ")
    sisso_file_path = input("Enter the path to your original sisso.json file (e.g., '/home/user/sisso.json'): ")
    save_path = input("Enter the path where holdout files will be saved (e.g., '/home/user/holdOuts/'): ")
    base_sisso_folder_path = input("Enter the base directory for SISSO data (e.g., '/home/user/sissoDataNew/'): ")
    models_base_path = input("Enter the base directory for saving models (e.g., '/home/user/models/'): ")
    additional_files_path = save_path  # Assuming holdout files are saved in the same directory
    
    num_columns_to_remove = int(input("Enter the number of columns to remove: "))
    num_holdouts = int(input("Enter the number of holdout sets you want: "))
    num_iterations = int(input("Enter the number of iterations per holdout: "))
    fraction_to_remove = float(input("Enter the fraction of data to remove per iteration (e.g., 0.25 for 25%): "))
    holdout_fraction = float(input("Enter the fraction of data to use as holdout set (e.g., 0.2 for 20%): "))
    
    # Ensure directories exist
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(base_sisso_folder_path, exist_ok=True)
    os.makedirs(models_base_path, exist_ok=True)
    
    # Create holdout sets
    create_holdout_sets(data_path, save_path, num_holdouts, num_iterations, fraction_to_remove, holdout_fraction)
    
    # Update sisso.json files
    update_sisso_json(sisso_file_path, base_sisso_folder_path, num_holdouts, num_iterations)
    
    # Create models
    create_models(base_sisso_folder_path, models_base_path, additional_files_path, num_columns_to_remove)
    
    # Copy holdout files
    copy_holdout_files(save_path, models_base_path)
    
    # Base path for processing files
    base_path = os.path.dirname(models_base_path.rstrip('/'))
    
    # Process files and generate predictions
    for it_number in range(1, num_holdouts + 1):
        process_file_set(it_number, base_path)




#notes
# _propkey is removed from the data file check that 
# dont just directly drop the column check for "property_key" value in sisso.json and besides that drop the column to fit the model  (additional_file_source_path se randomly drop mat kro)

#run and test 

#try not to ask to user to type path for every thing make it easier only ask user to put the number of iteration , holdouts ,0.25.0.2




