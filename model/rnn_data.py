import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_X_COLUMNS = [
    "theta1",
    "theta2",
    "target_x",
    "target_y",
    "obstacle_x",
    "obstacle_y",
    "ee_dist_to_target",
    "ee_dist_to_obstacle",
    "min_dist_obstacle_link_1",
    "min_dist_obstacle_link_2",
    "u1_prev",
    "u2_prev",
    "ee_dx_target",
    "ee_dy_target",
]

DEFAULT_Y_COLUMNS = ["u1", "u2"]


def load_h5_runs(file_path: str):
    data = {}
    with h5py.File(file_path, "r") as f:
        for run_key in f.keys():
            data[run_key] = {}
            for ds_key in f[run_key].keys():
                data[run_key][ds_key] = f[run_key][ds_key][:]
    return data


def build_sequences_from_runs(
    data_dict,
    x_columns=None,
    y_columns=None,
    exclude_columns=None,
):
    if x_columns is None:
        x_columns = DEFAULT_X_COLUMNS
    if y_columns is None:
        y_columns = DEFAULT_Y_COLUMNS
    if exclude_columns is None:
        exclude_columns = []

    effective_x_columns = [c for c in x_columns if c not in exclude_columns]

    X_list = []
    Y_list = []
    run_keys = sorted(data_dict.keys(), key=lambda x: int(x.split("_")[1]))

    for run_key in run_keys:
        run = data_dict[run_key]

        x_seq = np.stack([run[col] for col in effective_x_columns], axis=1)  # (T, F)
        y_seq = np.stack([run[col] for col in y_columns], axis=1)            # (T, 2)

        X_list.append(x_seq)
        Y_list.append(y_seq)

    X = np.stack(X_list, axis=0)   # (N, T, F)
    Y = np.stack(Y_list, axis=0)   # (N, T, 2)

    return X, Y, effective_x_columns, y_columns


def split_runs(X, Y, test_size=0.2, random_state=42):
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)


def scale_sequence_features(X_train, X_test):
    n_train, t_train, f_train = X_train.shape
    n_test, t_test, f_test = X_test.shape

    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, f_train)
    X_test_2d = X_test.reshape(-1, f_test)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, t_train, f_train)
    X_test_scaled = scaler.transform(X_test_2d).reshape(n_test, t_test, f_test)

    return X_train_scaled, X_test_scaled, scaler