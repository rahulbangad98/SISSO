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
    model_files = [os.path.join(base_path, f"models/it{it_number}/{i}/train_dim_4_model_0.dat") for i in range(1, 11)]
    file_path = os.path.join(base_path, f"taskThoholdout_set_{it_number}.csv")

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
base_path = '/home/u28/rahulbangad'

# Iterate over different iterations
for it_number in range(1, 3):  # Adjust the range as needed for it1, it2, etc.
    process_file_set(it_number, base_path)
