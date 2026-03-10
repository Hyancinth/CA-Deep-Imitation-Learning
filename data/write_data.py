import numpy as np
import h5py as h5
from pathlib import Path

from utils.utils import dist_obstacle_to_links, fk, point_in_workspace

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

def generate_goal_point():
    """
    Generate a random goal point within the workspace of the robot
    """
    while True:
        x = np.random.uniform(-2.0, 2.0)
        y = np.random.uniform(0.0, 2.0)

        if point_in_workspace(x, y):
            return (x, y)

def generate_obstacle_point(theta, a):
    """
    Generate a random obstacle point within the workspace of the robot
    """
    while True:
        x = np.random.uniform(-2.0, 2.0)
        y = np.random.uniform(0.0, 2.0)

        if point_in_workspace(x, y) and min(dist_obstacle_to_links((x, y), theta, a)) > 0.1: # ensure obstacle is not too close to the robot links
            return (x, y)

def generate_data(mpc, target, obstacle, a):
    """
    Generate training data for imitation learning by extracting features from MPC data
    """

    ee_dist_to_target = []
    ee_dist_to_obstacle = []
    min_dist_obstacle_link_1 = []
    min_dist_obstacle_link_2 = []

    # extract joint angles
    theta1 = mpc.data['_x', 'theta1'].flatten()
    theta2 = mpc.data['_x', 'theta2'].flatten()

    # extract control inputs (joint velocities)
    u1 = mpc.data['_u', 'u1'].flatten()
    u2 = mpc.data['_u', 'u2'].flatten()

    u1_prev = np.concatenate(([0.0], u1[:-1])) # assume 0 velocity at the first time step and shift the control inputs
    u2_prev = np.concatenate(([0.0], u2[:-1]))

    assert len(theta1) == len(theta2) == len(u1) == len(u2), "Data length mismatch" 

    for i in range(len(theta1)):
        ee_pos = fk([theta1[i], theta2[i]], a)
        ee_x = ee_pos[2]
        ee_y = ee_pos[3]

        ee_dist_to_target.append(np.sqrt((ee_x - target[0])**2 + (ee_y - target[1])**2))
        ee_dist_to_obstacle.append(np.sqrt((ee_x - obstacle[0])**2 + (ee_y - obstacle[1])**2))
        dist_obstacles_links = dist_obstacle_to_links(obstacle, [theta1[i], theta2[i]], a)
        min_dist_obstacle_link_1.append(dist_obstacles_links[0])
        min_dist_obstacle_link_2.append(dist_obstacles_links[1])

    return {
        'theta1': theta1,
        'theta2': theta2,
        'target_x': target[0],
        'target_y': target[1],
        'obstacle_x': obstacle[0],
        'obstacle_y': obstacle[1],
        'min_dist_obstacle_link_1': min_dist_obstacle_link_1,
        'min_dist_obstacle_link_2': min_dist_obstacle_link_2,
        'u1_prev': u1_prev,
        'u2_prev': u2_prev,
        'u1': u1,
        'u2': u2
    }
    
def write_data_to_file(data, filename):
    """
    Write data to H5 file

    Format of data:
    '/'
        'run_{i}'
            data
    """
    base_dir = Path(__file__).resolve().parent
    filepath = base_dir.parent/"model"/"data"/filename
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

    print_file_structure(f)

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