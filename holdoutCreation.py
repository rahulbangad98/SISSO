import pandas as pd
import numpy as np
import os

# Define the path where files will be saved
save_path = '/Users/rahulbangad/Desktop/taskTho'

# Check if the directory exists, if not, create it
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load the data from CSV file
data = pd.read_csv('/Users/rahulbangad/Downloads/data copy.csv')

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
