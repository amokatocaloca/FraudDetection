import pandas as pd
import numpy as np
import sys
import os

def simulate_data_changes(data_path, output_dir):
    np.random.seed(42)  # For reproducibility

    # Check if the data file exists
    if not os.path.isfile(data_path):
        print(f"Data file {data_path} does not exist.")
        sys.exit(1)

    # Read the original dataset with explicit encoding to avoid issues
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except Exception as e:
        print(f"Failed to read {data_path} due to: {e}")
        sys.exit(1)
    
    print("DataFrame columns: ", df.columns)  # Debug: Print DataFrame columns
    if 'amount' not in df.columns:
        print("Column 'amount' not found. Available columns: ", df.columns)
        sys.exit(1)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    numeric_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    string_columns = ['nameOrig', 'nameDest']

    # Check and modify numeric columns, except for isFraud and isFlaggedFraud
    for column in numeric_columns:
        if column in df.columns:
            random_factor = np.random.rand() * 0.5 + 0.75  # Random factor between 0.75 and 1.25
            df[column] *= random_factor
        else:
            print(f"Column {column} not found in DataFrame.")

    # Shuffle string columns
    for column in string_columns:
        if column in df.columns:
            df[column] = np.random.permutation(df[column])
        else:
            print(f"Column {column} not found in DataFrame.")

    # Save the modified dataset as "modified_fraud_data.csv"
    output_file_path = os.path.join(output_dir, "modified_fraud_data.csv")
    df.to_csv(output_file_path, index=False)
    print(f"Modified dataset saved to {output_file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python simulate_data_changes.py <input_data_path> <output_data_directory>")
        sys.exit(1)
    
    input_data_path = sys.argv[1]
    output_data_directory = sys.argv[2]
    simulate_data_changes(input_data_path, output_data_directory)
