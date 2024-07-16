

import os
from sissopp import Inputs, FeatureSpace, SISSORegressor
import json

# Directory containing the generated sisso.json files
sisso_folder_path = '/Users/rahulbangad/Downloads/sissoFolder/'

# List all sisso.json files in the directory
sisso_files = [f for f in os.listdir(sisso_folder_path) if f.startswith('sisso_holdout') and f.endswith('.json')]

for sisso_file in sisso_files:
    # Print just the filename
    print(sisso_file)
    
    # Construct the full path to the sisso.json file
    sisso_file_path = os.path.join(sisso_folder_path, sisso_file)
    #print(sisso_file_path)
    # Load the current sisso.json content
    inputs = Inputs(sisso_file)
    
    # # # Initialize FeatureSpace and SISSORegressor with the current inputs
    feature_space = FeatureSpace(inputs)
    sisso = SISSORegressor(inputs, feature_space)
    
    # # # Fit the model
    sisso.fit()
    
    print(f'Processed {sisso_file_path}')
