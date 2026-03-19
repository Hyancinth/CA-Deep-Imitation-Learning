import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

X_COLUMNS = [
    'theta1',
    'theta2',
    'target_x',
    'target_y',
    'obstacle_x',
    'obstacle_y',
    'ee_dist_to_target',
    'ee_dist_to_obstacle',
    'min_dist_obstacle_link_1',
    'min_dist_obstacle_link_2',
    'u1_prev',
    'u2_prev'
]

Y_COLUMNS = [
    'u1', 
    'u2'
]

def split_test_train(data, test_size=0.2):
    """
    Split data dict into training and testing sets

    load_data_from_file should be called before this to load the data from the h5 file into a dict
    This should be called before generating input-output pairs to prevent data leakage between train and test sets
    """
    keys = list(data.keys()) # list of all runs
    train_keys, test_keys = train_test_split(keys, test_size=test_size, random_state=42) # split runs into train and test sets

    train_data = {k: data[k] for k in train_keys}
    test_data = {k: data[k] for k in test_keys}

    return train_data, test_data

def generate_input_output_data(data):
    """
    Convert data dict to array of input features (X) and outputs (Y)

    Returns:
    - x (numpy array): array of input features for all runs, shape (num_samples, num_features)
    - y (numpy array): array of outputs for all runs, shape (num_samples, num_outputs)
    """
    x = []
    y = []
    for run in data.keys():
        run_data = data[run]
        x_run = np.column_stack([run_data[col] for col in X_COLUMNS])
        y_run = np.column_stack([run_data[col] for col in Y_COLUMNS])

        x.append(x_run)
        y.append(y_run)
    
    return np.vstack(x), np.vstack(y)

def scale_features(x_train, x_test):
    """
    Scale input features

    Returns:
    - x_train_scaled (numpy array): scaled training input features
    - x_test_scaled (numpy array): scaled testing input features
    """

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    save_dir = "model/scalers/"

    scaler_filename = save_dir + "input_scaler.pkl"
    joblib.dump(scaler, scaler_filename)

    return x_train_scaled, x_test_scaled


if __name__ == "__main__":
    from data.load_data import load_data_from_file
    file_path = "model/data/test_data_1.h5"
    data = load_data_from_file(file_path) # only two runs 
    x, y = generate_input_output_data(data)
    print("Input features (X) shape: ", x.shape)
    print("Output labels (Y) shape: ", y.shape)
    print("First 5 rows of X: \n", x[:5])
    

