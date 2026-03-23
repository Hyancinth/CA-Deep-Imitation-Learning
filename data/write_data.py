import numpy as np
import h5py as h5
from pathlib import Path

from utils.utils import dist_to_links, fk, point_in_workspace

def generate_goal_point(x0, a):
    """
    Generate a random goal point within the workspace of the robot
    """
    a1 = a[0]
    a2 = a[1]

    # add margin to ensure target is not too close to edge of the workspace
    margin = 0.1
    r_min = abs(a1 - a2) + margin
    r_max = a1 + a2 - margin
    
    while True:
        theta = np.random.uniform(0, np.pi)  # y >= 0 
        r = np.sqrt(np.random.uniform(r_min**2, r_max**2))  # uniform in area

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # prevent goal from being too close to edge of the workspace
        min_dist = min(dist_to_links(np.array([x, y]), x0, a))
        if min_dist > 0.2:
            return np.array([x, y])


def generate_obstacle_point(x0, a, target):
    """
    Generate a random obstacle point within the workspace of the robot, ensuring that the obstacle is not too close to the joint links
    """
    a1 = a[0]
    a2 = a[1]
    r_max = a1 + a2
    # obstacle is not constrained by r_min since it can be close to the robot base as long as it's not too close to the links

    while True:
        theta = np.random.uniform(0, np.pi)
        r = np.sqrt(np.random.uniform(0, r_max**2))

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        obstacle = np.array([x, y])

        # it should be fine if the obstacle is close to the target since the MPC should learn to navigate around it, 
        # as long as it's not too close to the links which would make the task infeasible
    
        # distance from target
        # if np.linalg.norm(obstacle - target) < 0.3:
        #     continue

        # distance from robot links
        min_dist = min(dist_to_links(obstacle, x0, a))
        if min_dist > 0.2:
            return obstacle

def generate_data(mpc, target, obstacle, a):
    """
    Generate training data for imitation learning by extracting features from MPC data
    """

    ee_dist_to_target = []
    ee_dist_to_obstacle = []
    min_dist_obstacle_link_1 = []
    min_dist_obstacle_link_2 = []
    ee_dx_target = []
    ee_dy_target = []

    # extract joint angles
    theta1 = mpc.data['_x', 'theta1'].flatten()
    theta2 = mpc.data['_x', 'theta2'].flatten()

    # extract control inputs (joint velocities)
    u1 = mpc.data['_u', 'u1'].flatten()
    u2 = mpc.data['_u', 'u2'].flatten()

    u1_prev = np.concatenate(([0.0], u1[:-1])) # assume 0 velocity at the first time step and shift the control inputs
    u2_prev = np.concatenate(([0.0], u2[:-1]))

    assert len(theta1) == len(theta2) == len(u1) == len(u2), "Data length mismatch" 

    for i in range(len(theta1)): # iterate through each time step to compute features that depend on the joint angles and positions
        ee_pos = fk([theta1[i], theta2[i]], a)
        ee_x = ee_pos[2]
        ee_y = ee_pos[3]

        ee_dist_to_target.append(np.sqrt((ee_x - target[0])**2 + (ee_y - target[1])**2))
        ee_dist_to_obstacle.append(np.sqrt((ee_x - obstacle[0])**2 + (ee_y - obstacle[1])**2))
        dist_obstacles_links = dist_to_links(obstacle, [theta1[i], theta2[i]], a)
        min_dist_obstacle_link_1.append(dist_obstacles_links[0])
        min_dist_obstacle_link_2.append(dist_obstacles_links[1])
        ee_dx_target.append(target[0] - ee_x)
        ee_dy_target.append(target[1] - ee_y)


    return {
        'theta1': theta1,
        'theta2': theta2,
        'target_x': np.full_like(theta1, target[0]), # same value across all time steps, make it the same length as other features for easier dataframe creation later
        'target_y': np.full_like(theta1, target[1]),
        'obstacle_x': np.full_like(theta1, obstacle[0]),
        'obstacle_y': np.full_like(theta1, obstacle[1]),
        'min_dist_obstacle_link_1': min_dist_obstacle_link_1,
        'min_dist_obstacle_link_2': min_dist_obstacle_link_2,
        'ee_dist_to_target': ee_dist_to_target,
        'ee_dist_to_obstacle': ee_dist_to_obstacle,
        'u1_prev': u1_prev,
        'u2_prev': u2_prev,
        'u1': u1,
        'u2': u2,
        'ee_dx_target': ee_dx_target,
        'ee_dy_target': ee_dy_target
    }
    
def write_data_to_file(data, filename, type='model_data'):
    """
    Write data to H5 file

    Format of data:
    '/'
        'run_{i}'
            data
    """
    base_dir = Path(__file__).resolve().parent
    if type == 'model_data':
        filepath = base_dir.parent/"model"/"data"/filename
    elif type == 'model_prediction':
        filepath = base_dir.parent/"analysis"/"model_predictions"/filename
    elif type =="hidden_test_data":
        filepath = base_dir.parent/"model"/"hidden_test_data"/filename
        
    filepath.parent.mkdir(parents=True, exist_ok=True) # create data directory if it doesn't exist

    f = h5.File(filepath, 'a')
    # the data should have an entry specifying the run number for group creation
    grp = f.require_group(f"run_{data['run_number']}")
    grp.attrs['run_number'] = data['run_number'] # store run number as group attribute for easy access
    for key, value in data.items():
        if key != 'run_number':
            dset = grp.create_dataset(key, data=value)
    
    # print file structure for verification
    print(f"Data written to {filename}. Current file structure:")

    # print_file_structure(f)

    f.close()
    
def print_file_structure(file, indent=0):
    """
    Recursively print the structure of h5 file
    """
    for key in file.keys(): 
        item = file[key] # 
        if isinstance(item, h5.Group):
            print('  ' * indent + f"Group: {key}")
            print_file_structure(item, indent + 1) # recursive call to print subgroup structure
        else: 
            print('  ' * indent + f"Dataset: {key}, shape: {item.shape}, dtype: {item.dtype}")