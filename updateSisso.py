import json
import os

# Path to the original sisso.json file
sisso_file_path = "/Users/rahulbangad/Downloads/sisso.json"

# Load the original sisso.json content
with open(sisso_file_path, 'r') as file:
    sisso_content = json.load(file)

# Base path where modified data files are stored
save_path = '/Users/rahulbangad/Downloads/sissoFolder/'

# Ensure the save path directory exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Parameters
num_holdouts = 5
num_iterations = 10

# Loop through each holdout and iteration to create new sisso.json files
for holdout in range(1, num_holdouts + 1):
    for iteration in range(1, num_iterations + 1):
        # Update the data_file path
        sisso_content["data_file"] = f'{save_path}modified_data_holdout_{holdout}_iteration_{iteration}.csv'
        
        # Define the new sisso.json file path
        new_sisso_file_path = f'{save_path}sisso_holdout_{holdout}_iteration_{iteration}.json'
        
        # Save the updated sisso.json file
        with open(new_sisso_file_path, 'w') as new_file:
            json.dump(sisso_content, new_file, indent=4)

        print(f'Updated sisso.json saved to {new_sisso_file_path}')
