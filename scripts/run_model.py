import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

from data.load_data import load_data_from_file
from data.model_data import split_test_train, generate_input_output_data, scale_features, X_COLUMNS, Y_COLUMNS
from data.write_data import write_data_to_file
from model.basicAnn import basicAnn
from model.train_test_nn import create_data_loaders, train_model
from utils.utils import fk, dist_to_links
from visualization.visualize_model import plot_train_test_losses, plot_ee_trajectories


def train_and_evaluate_model(file_path, model, exclude_columns = None, num_epochs=200, learning_rate=0.001):
    """
    Train model on the dataset
    
    Args:
        file_path (str): path to the dataset file
        model (nn.Module): PyTorch model to be trained
        exclude_columns (list of str): list of column names to exclude from the input features

    Returns:
    - model (nn.Module): trained PyTorch model
    - train_losses (torch.Tensor): tensor containing the training loss at each epoch
    - test_losses (torch.Tensor): tensor containing the test loss at each epoch
    - scaler_filename (str): filename of the saved scaler used for feature scaling
    - timestamp (str): timestamp corresponding to the saved model and scaler
    """
    data = load_data_from_file(file_path)
    train_data, test_data = split_test_train(data)
    if exclude_columns is None:
        exclude_columns = []
    x_train, y_train = generate_input_output_data(train_data, exclude_columns)
    x_test, y_test = generate_input_output_data(test_data, exclude_columns)
    x_train_scaled, x_test_scaled, scaler_filename, timestamp = scale_features(x_train, x_test)

    # create data loaders
    train_loader, test_loader = create_data_loaders(x_train_scaled, y_train, x_test_scaled, y_test)

    # initialize model and train
    nn = model
    model, train_losses, test_losses = train_model(nn, train_loader, test_loader, num_epochs=num_epochs, learning_rate=learning_rate)

    return model, train_losses, test_losses, scaler_filename, timestamp

def get_hidden_data(file_path, exclude_columns = None):
    """
    Get data from the hidden dataset for evaluating the trained model

    Args:
        file_path (str): path to the hidden dataset file
        exclude_columns (list of str): list of column names to exclude from the input features

    Returns:
    - data (dict): dictionary containing data from the hidden dataset
    """
    data = load_data_from_file(file_path)
    theta1_init = data['run_0']['theta1'][0]
    theta2_init = data['run_0']['theta2'][0]
    goal = np.array([data['run_0']['target_x'][0], data['run_0']['target_y'][0]])
    obstacle = np.array([data['run_0']['obstacle_x'][0], data['run_0']['obstacle_y'][0]])
    theta1 = data['run_0']['theta1'][:] # for later when we want to compare the predicted and ground truth trajectories
    theta2 = data['run_0']['theta2'][:]
    if exclude_columns is None:
        exclude_columns = []
    x, y = generate_input_output_data(data, exclude_columns)
    # x_scaled, _, _ = scale_features(x, x) # scale the data using the same scaler that was used for training

    data = {
        'theta1_init': theta1_init,
        'theta2_init': theta2_init,
        'theta1': theta1,
        'theta2': theta2,
        'x': x, 
        # 'x_scaled': x_scaled,
        'y': y,
        'goal': goal,
        'obstacle': obstacle
    }

    return data

def compute_distances(theta, target, obstacle, a):
    """
    Compute various distances and relative positions needed for feature construction

    Args:
        theta (numpy array): array containing the joint angles [theta1, theta2]
        target (numpy array): array containing the target position [target_x, target_y]
        obstacle (numpy array): array containing the obstacle position [obstacle_x, obstacle_y]
        a (numpy array): array containing the link lengths [a1, a2]

    Returns:
    - ee_dist_to_target (float): distance from end-effector to target
    - ee_dist_to_obstacle (float): distance from end-effector to obstacle
    - dist_obstacles_links (list of float): list containing the minimum distance from the obstacle to each of the robot links
    - ee_dx_target (float): x distance from end-effector to target
    - ee_dy_target (float): y distance from end-effector to target
    """
    x1, y1, x2, y2 = fk(theta, a)
    ee_pos = np.array([x2, y2])
    ee_dist_to_target = np.linalg.norm(ee_pos - target)
    ee_dist_to_obstacle = np.linalg.norm(ee_pos - obstacle)
    dist_obstacles_links = dist_to_links(obstacle, theta, a)
    ee_dx_target = target[0] - ee_pos[0]
    ee_dy_target = target[1] - ee_pos[1]
    
    return ee_dist_to_target, ee_dist_to_obstacle, dist_obstacles_links, ee_dx_target, ee_dy_target

def build_feature_vector(theta, target, obstacle, u_prev, a, exclude_columns = None):
    """
    Build input feature array for the model

    Args:
        theta (numpy array): array containing the joint angles [theta1, theta2]
        target (numpy array): array containing the target position [target_x, target_y]
        obstacle (numpy array): array containing the obstacle position [obstacle_x, obstacle_y]
        u_prev (numpy array): array containing the previous control inputs [u1_prev, u2_prev]
            u_prev is no longer being used as a feature in the current model, but it is included here in case we want to use it in future iterations of the model. If not using it, simply pass in an array of zeros for u_prev and exclude 'u1_prev' and 'u2_prev' from the input features using the exclude_columns parameter
        a (numpy array): array containing the link lengths [a1, a2]
        exclude_columns (list of str): list of column names to exclude from the input features 
    """
    theta1 = theta[0]
    theta2 = theta[1]

    if exclude_columns is None:
        exclude_columns = []

    ee_dist_to_target, ee_dist_to_obstacle, dist_obstacles_links, ee_dx_target, ee_dy_target = compute_distances(theta, target, obstacle, a)
    
    feature_dict = {
        'theta1': theta1,
        'theta2': theta2,
        'target_x': target[0],
        'target_y': target[1],
        'obstacle_x': obstacle[0],
        'obstacle_y': obstacle[1],
        'ee_dist_to_target': ee_dist_to_target,
        'ee_dist_to_obstacle': ee_dist_to_obstacle,
        'min_dist_obstacle_link_1': dist_obstacles_links[0],
        'min_dist_obstacle_link_2': dist_obstacles_links[1],
        'u1_prev': u_prev[0],
        'u2_prev': u_prev[1],
        # 'ee_dx_target': ee_dx_target,
        # 'ee_dy_target': ee_dy_target
    }
    

    x = np.array([feature_dict[col] for col in feature_dict if col not in exclude_columns])

    return x

def predict_control(model, x, scaler):
    """
    Get the model's predicted control input

    Args:
        model (nn.Module): trained PyTorch model
        x (numpy array): input feature array for the current time step
        scaler (sklearn scaler object): scaler used for feature scaling during training, used to scale the input features before feeding into the model
    
    Returns:
    - u (numpy array): array containing the predicted control inputs [u1, u2]
    """
    x_scaled = scaler.transform([x])

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        u = model(x_tensor).numpy()

    u = np.clip(u, -3.0, 3.0)
    return u[0]

def run_model(model, scaler, theta0, target, obstacle, a, exclude_columns = None):
    """
    Run the trained model on hidden dataset

    Args:
        model (nn.Module): trained PyTorch model
        scaler (sklearn scaler object): scaler used for feature scaling during training, used to scale the input features before feeding into the model
        theta0 (numpy array): array containing the initial joint angles [theta1_init, theta2_init]
        target (numpy array): array containing the target position [target_x, target_y]
        obstacle (numpy array): array containing the obstacle position [obstacle_x, obstacle_y]
        a (numpy array): array containing the link lengths [a1, a2]
        exclude_columns (list of str): list of column names to exclude from the input features
    
    Returns:
    - ee_trajectory (numpy array): array containing the predicted end-effector trajectory, shape (num_time_steps, 2)
    """
    if exclude_columns is None:
        exclude_columns = []

    dt = 0.1
    num_steps = 100 # matches the number of time steps the MPC took

    theta = np.array(theta0)
    u_prev = np.array([0.0, 0.0])

    # initial forward kinematics
    fk_result = fk(theta, a)
    joint1_pos = fk_result[0:2]
    ee_pos = fk_result[2:4]

    # return 
    theta_trajectory = [theta.copy()]
    ee_trajectory = [ee_pos]
    joint1_trajectory = [joint1_pos]
    u_trajectory = [u_prev]

    for step in range(num_steps):
        x = build_feature_vector(theta, target, obstacle, u_prev, a, exclude_columns)
        u = predict_control(model, x, scaler)

        # update theta using euler integration
        theta = theta + u * dt
        u_prev = u

        j1_pos = fk(theta, a)[:2]
        joint1_trajectory.append(j1_pos)
        ee_pos = fk(theta, a)[2:4]
        ee_trajectory.append(ee_pos)

        theta_trajectory.append(theta.copy())
        u_trajectory.append(u.copy())

        print(f"Step {step+1}/{num_steps}, Theta: {theta}, Control: {u}, EE Position: {ee_pos}")

    return np.array(ee_trajectory), np.array(joint1_trajectory), np.array(theta_trajectory), np.array(u_trajectory)


if __name__ == "__main__":
    """
    Steps for evaluating the trained model on the hidden dataset
    1. Load the hidden dataset using load_data_from_file and generate input-output pairs using generate_input_output_data
    2. Only use the first time step of the input feature
    3. Scale the input features using the same scaler that was used for training (this should be saved to a file during training)
    4. In a loop over the same number of time steps as the dataset, use the model to predict the next state given the current state, and update the current state using the predicted next state
    5. The model will output u1, u2
    6. Convert u1, u2 to theta1, theta2 using euler integration: theta1_next = theta1_current + u1 * dt, theta2_next = theta2_current + u2 * dt
    7. Using the theta calculate the position of j1 and j2 (ee) using fk and calculate all the required features to feed into the model
    7. Store store both the joint velocities (u1, u2) and the joint angles (theta1, theta2) at each time step for both the model predictions and the ground truth from the dataset
    """
    # change these to appropriate file 
    training_file_name = "data_317_01_100"
    training_file_path = "model/data/data_317_01_100.h5"
    hidden_file_path = "model/hidden_test_data/hidden_test_data.h5"

    exclude_columns = ['u1_prev', 'u2_prev']
    # exclude_columns = []
    # exclude_columns = ['u1_prev', 'u2_prev', 'ee_dx_target', 'ee_dy_target']

    # load the training data and train the model
    nn = basicAnn(input_size=len(X_COLUMNS) - len(exclude_columns), output_size=len(Y_COLUMNS))
    model, train_losses, test_losses, scaler_filename, timestamp = train_and_evaluate_model(training_file_path, nn, exclude_columns, num_epochs = 500, learning_rate=0.001)
    plot_train_test_losses(train_losses, test_losses)

    # save state dictionary of the trained model
    # save the model under the same timestamp as the scaler
    save_dir = "model/trained_models/"
    model_filename = save_dir + f"model_{timestamp}.pt"    
    torch.save(model.state_dict(), model_filename)

    # load hidden dataset
    data = get_hidden_data(hidden_file_path)    

    a = [1.0, 1.0]
    theta0 = np.array([data['theta1_init'], data['theta2_init']])
    target = data['goal']
    obstacle = data['obstacle']
    scaler = joblib.load(scaler_filename)
    # run model on hidden dataset and get predicted end effector trajectory
    ee_trajectory_pred, joint1_trajectory_pred, theta_trajectory_pred, u_trajectory_pred = run_model(model, scaler, theta0, target, obstacle, a, exclude_columns)
    print(f"Length of predicted EE trajectory: {len(ee_trajectory_pred)}")

    # ground truth end effector trajectory
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

    write_data_to_file(data_to_save, filename=f"basicann2_model_predictions_{training_file_name}_{timestamp}.h5")
    print(f"Model predictions saved to h5 file: basicann2_model_predictions_{training_file_name}_{timestamp}.h5")
