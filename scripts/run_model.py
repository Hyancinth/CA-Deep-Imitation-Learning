import torch
import numpy as np
import matplotlib.pyplot as plt

from data.load_data import load_data_from_file
from data.model_data import split_test_train, generate_input_output_data, scale_features
from model.basicAnn import basicAnn
from model.train_test_nn import create_data_loaders, train_model
from utils.utils import fk, dist_to_links


def train_and_evaluate_model(file_path):
    x_train_scaled, y_train, x_test_scaled, y_test = load_data_from_file(file_path)

    # create data loaders
    train_loader, test_loader = create_data_loaders(x_train_scaled, y_train, x_test_scaled, y_test)

    # initialize model and train
    model = basicAnn()
    model, train_losses, test_losses = train_model(model, train_loader, test_loader)

    return model, train_losses, test_losses

def load_data(file_path):
    data = load_data_from_file(file_path)
    theta1_init = data['run_0']['theta1'][0]
    theta2_init = data['run_0']['theta2'][0]
    train_data, test_data = split_test_train(data)
    x_train, y_train = generate_input_output_data(train_data)
    x_test, y_test = generate_input_output_data(test_data)
    x_train_scaled, x_test_scaled = scale_features(x_train, x_test)

    return x_train_scaled, y_train, x_test_scaled, y_test, theta1_init, theta2_init

def test_data(file_path):
    # the dataset only contains a single run, so we can just load the data and generate input-output pairs without splitting into train and test sets
    data = load_data_from_file(file_path)
    x, y = generate_input_output_data(data)
    x_scaled, _ = scale_features(x, x) # scale the data using the same scaler for both train and test sets
    return x_scaled, y

def compute_distances(theta, target, obstacle, a):
    x1, y1, x2, y2 = fk(theta, a)
    ee_pos = np.array([x2, y2])
    ee_dist_to_target = np.linalg.norm(ee_pos - target)
    ee_dist_to_obstacle = np.linalg.norm(ee_pos - obstacle)
    dist_obstacles_links = dist_to_links(obstacle, theta, a)

    return ee_dist_to_target, ee_dist_to_obstacle, dist_obstacles_links


def build_feature_vector(theta, target, obstacle, u_prev, a):
    theta1 = theta[0]
    theta2 = theta[1]

    ee_dist_to_target, ee_dist_to_obstacle, dist_obstacles_links = compute_distances(theta, target, obstacle, a)
    x = np.array([
        theta1,
        theta2,
        target[0],
        target[1],
        obstacle[0],
        obstacle[1],
        ee_dist_to_target,
        ee_dist_to_obstacle,
        dist_obstacles_links[0],
        dist_obstacles_links[1],
        u_prev[0],
        u_prev[1]
    ])

    return x

def predict_control(model, x, scaler):
    x_scaled = scaler.transform([x])

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        u = model(x_tensor).numpy()

    return u[0]

def run_model(model, scaler, theta0, target, obstacle, a):
    dt = 0.1
    num_steps = 100

    theta = np.array(theta0)
    u_prev = np.array([0.0, 0.0])

    ee_trajectory = []

    for step in range(num_steps):
        x = build_feature_vector(theta, target, obstacle, u_prev, a)
        u = predict_control(model, x, scaler)

        # update theta using euler integration
        theta = theta + u * dt
        u_prev = u

        ee_pos = fk(theta, a)[2:4]
        ee_trajectory.append(ee_pos)
    
    return np.array(ee_trajectory)


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
    pass
