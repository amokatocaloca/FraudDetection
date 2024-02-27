import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_and_filter(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return

    df = pd.read_csv(filename)
    
    if df.empty:
        print("The provided dataset is empty.")
        return

    transaction_types = ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'CASH_IN', 'DEBIT']
    
    for transaction_type in transaction_types:
        filtered_df = df[df['type'] == transaction_type].copy()
        sampled_df = filtered_df.sample(frac=0.7, random_state=42)

        le = LabelEncoder()
        sampled_df['type'] = le.fit_transform(sampled_df['type'])
        sampled_df['nameOrig'] = le.fit_transform(sampled_df['nameOrig'])
        sampled_df['nameDest'] = le.fit_transform(sampled_df['nameDest'])

        processed_data_path = os.path.join('DLBDSMTP01', 'data', 'processed', f'processed_{transaction_type.lower()}_data.csv')
        sampled_df.to_csv(processed_data_path, index=False)

        print(f"Processed file saved: {processed_data_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess_and_filter.py <path_to_data_file>")
        sys.exit(1)
    
    data_file_path = sys.argv[1]
    preprocess_and_filter(data_file_path)
