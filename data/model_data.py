import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

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
    'u2_prev',
    'ee_dx_target',
    'ee_dy_target',
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

def generate_input_output_data(data, exclude_columns = None):
    """
    Convert data dict to array of input features (X) and outputs (Y)

    Returns:
    - x (numpy array): array of input features for all runs, shape (num_samples, num_features)
    - y (numpy array): array of outputs for all runs, shape (num_samples, num_outputs)
    """
    if not exclude_columns:
        exclude_columns = []
    
    filter_x_columns = [col for col in X_COLUMNS if col not in exclude_columns]

    x = []
    y = []
    for run in data.keys():
        run_data = data[run]
        x_run = np.column_stack([run_data[col] for col in filter_x_columns])
        y_run = np.column_stack([run_data[col] for col in Y_COLUMNS])

        x.append(x_run)
        y.append(y_run)
    
    return np.vstack(x), np.vstack(y)

def scale_features(x_train, x_test, prev_scaler = None):
    """
    Scale input features

    Returns:
    - x_train_scaled (numpy array): scaled training input features
    - x_test_scaled (numpy array): scaled testing input features
    """

    if not prev_scaler:
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = "model/scalers/"
        scaler_filename = save_dir + f"input_scaler_{timestamp}.pkl"
        joblib.dump(scaler, scaler_filename)
    else:
        x_train_scaled = prev_scaler.fit_transform(x_train)
        x_test_scaled = prev_scaler.transform(x_test)
        scaler_filename = prev_scaler
        timestamp = 0


    return x_train_scaled, x_test_scaled, scaler_filename, timestamp



if __name__ == "__main__":
    from data.load_data import load_data_from_file
    file_path = "model/data/test_data_1.h5"
    data = load_data_from_file(file_path) # only two runs 
    x, y = generate_input_output_data(data)
    print("Input features (X) shape: ", x.shape)
    print("Output labels (Y) shape: ", y.shape)
    print("First 5 rows of X: \n", x[:5])
    x_train_scaled, x_test_scaled, _ = scale_features(x, x)

