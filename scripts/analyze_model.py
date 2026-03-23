import torch
import h5py as h5
import numpy as np
import joblib

from model.basicAnn import basicAnn
from analysis.analysis_loop import run_model, get_hidden_data
from data.model_data import X_COLUMNS, Y_COLUMNS
from analysis.utils import load_model, load_scaler
from utils.utils import fk
from visualization.visualize_model import plot_train_test_losses, plot_ee_trajectories
from data.write_data import write_data_to_file

if __name__ == "__main__":
    training_file_name = "data_322_01_100"
    hidden_file_name = "hidden_test_data_5"
    hidden_file_path = "model/hidden_test_data/hidden_test_data_5.h5"
    save_dir = "model/trained_models/"
    num_epochs = 500
    learning_rate = 0.001
    exclude_columns = ['u1_prev', 'u2_prev']

    model_path = f"{save_dir}/model_2026-03-23_15-04-10.pt"
    scaler_path = 'model/scalers/input_scaler_2026-03-23_15-04-10.pkl'
    timestamp = "2026-03-23_15-04-10"

    model = load_model(model_path, input_size=len(X_COLUMNS) - len(exclude_columns), output_size=len(Y_COLUMNS))
    scaler = load_scaler(scaler_path)

    data = get_hidden_data(hidden_file_path)
    theta0 = np.array([data['theta1_init'], data['theta2_init']])
    target = data['goal']
    obstacle = data['obstacle']
    a = [1.0, 1.0]

    ee_trajectory_pred, joint1_trajectory_pred, theta_trajectory_pred, u_trajectory_pred = run_model(model, scaler, theta0, target, obstacle, a)

    ee_trajectory_gt = np.array([fk([data['theta1'][i], data['theta2'][i]], a)[2:4] for i in range(len(data['theta1']))])
    print(f"Length of ground truth EE trajectory: {len(ee_trajectory_gt)}")

    initial_robot_link = fk(theta0, a) # for drawing initial robot config in the plot

    # plot the predicted and ground truth end-effector trajectories, along with the target and obstacle positions and the initial robot configuration
    plot_ee_trajectories(ee_trajectory_gt, ee_trajectory_pred, target, obstacle, initial_robot_link, a)

    # save predicted trajectory and other data to h5 file
    data_to_save = {
        "run_number": 0,  
        "theta1": theta_trajectory_pred[:, 0],
        "theta2": theta_trajectory_pred[:, 1],
        "u1": u_trajectory_pred[:, 0],
        "u2": u_trajectory_pred[:, 1],
        "ee_x": ee_trajectory_pred[:, 0],
        "ee_y": ee_trajectory_pred[:, 1],
        "joint1_x": joint1_trajectory_pred[:, 0],
        "joint1_y": joint1_trajectory_pred[:, 1],
        "target_x": np.array([target[0]]),
        "target_y": np.array([target[1]]),
        "obstacle_x": np.array([obstacle[0]]),
        "obstacle_y": np.array([obstacle[1]]),
        "hidden_data_file_path": hidden_file_path,
    }

    write_data_to_file(data_to_save, filename=f"basicann2_model_predictions_{training_file_name}_{timestamp}_{hidden_file_name}_exclude_{'_'.join(exclude_columns)}.h5")
    print(f"Model predictions saved to h5 file: basicann2_model_predictions_{training_file_name}_{timestamp}_{hidden_file_name}_exclude_{'_'.join(exclude_columns)}.h5")

