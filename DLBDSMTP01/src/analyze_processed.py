import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for file generation without requiring a window server
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_transaction_type(filename):
    """
    Extracts the transaction type from the filename.
    Assumes the filename format is 'processed_{transaction_type}_data.csv'.
    """
    base_name = os.path.basename(filename)
    transaction_type = base_name.split('_')[1]  # Assuming the second element after split is the transaction type.
    return transaction_type

def analyze(filename):
    if not os.path.exists(filename):
        logging.error(f"The file {filename} does not exist.")
        sys.exit(1)

    df = pd.read_csv(filename, usecols=['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud'])
    logging.info(f'Number of fraudulent transactions: {df["isFraud"].sum()}')

    features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    X = df[features].values
    y = df['isFraud'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=X_pca, columns=['PC 1', 'PC 2'])
    pca_df['isFraud'] = y 

    plt.figure(figsize=(10, 8))
    colors = {0: 'blue', 1: 'red'}
    labels = {0: 'Non-fraud', 1: 'Fraud'}
    for category, color in colors.items():
        subset = pca_df[pca_df['isFraud'] == category]
        plt.scatter(subset['PC 1'], subset['PC 2'], c=color, label=labels[category], alpha=0.5)
    plt.title('PCA of Transactions')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)

    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    transaction_type = get_transaction_type(filename)
    plot_filename = f"{transaction_type}_pca.png"
    plot_path = os.path.join(plot_dir, plot_filename)
    
    plt.savefig(plot_path)
    logging.info(f"Plot saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        logging.error("Please provide a filename as an argument.")
