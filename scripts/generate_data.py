import numpy as np
import casadi as ca

from utils.utils import fk

from data.write_data import generate_goal_point, generate_obstacle_point, generate_data, write_data_to_file
from mpc.simpleMPC2 import mpc_controller

def sample_target_and_obstacle(x0, a):
    """
    Generate a random target and obstacle point within the workspace of the robot, ensuring that the target is not too close to the initial position

    Does not guarantee that the robot can reach the target without colliding with the obstacle
    """
    target = generate_goal_point(x0, a)
    obstacle = generate_obstacle_point(x0, a, target)

    return target, obstacle

def run_mpc_controller(mpc, simulator, p_template, p_template_sim, target, obstacle, max_steps=100, x0=np.array([ca.pi/6, -ca.pi/6]), a=np.array([1.0, 1.0])):
    """
    Run the MPC controller for a given target and obstacle

    Returns:
    - success (boolean): whether the MPC successfully reached the target within an acceptable distance
    - data (dict): a dictionary containing the features extracted from the MPC run, or None if the run was unsuccessful
    """
    # update parameters
    p_template['_p', 0, 'target'] = target.reshape(2,1)
    p_template['_p', 0, 'obstacle'] = obstacle.reshape(2,1)

    p_template_sim['target'] = target.reshape(2,1)
    p_template_sim['obstacle'] = obstacle.reshape(2,1)

    simulator.reset_history()
    mpc.reset_history()

    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()

    for i in range(max_steps):
        u0 = mpc.make_step(x0)
        x0 = simulator.make_step(u0)

        if not mpc.solver_stats["success"]:
            print(f"MPC solver failed at step {i+1}. Aborting this run.")
            return False, None

    final_state = np.array([
        mpc.data['_x', 'theta1'][-1][0],
        mpc.data['_x', 'theta2'][-1][0]
    ])

    final_state_pos = fk(final_state, a)
    ee_final_pos = final_state_pos[2:4]
    distance = np.linalg.norm(ee_final_pos - target)

    if distance > 0.1:
        print("MPC failed to reach the target within the acceptable distance. Distance: ", distance)
        return False, None

    return True, generate_data(mpc, target, obstacle, a)

if __name__ == "__main__":
    mpc, simulator, p_template, p_template_sim = mpc_controller()

    num_runs = 100 # number of runs to generate data for - change this to what you want
    run = 0

    while run < num_runs:
        x0 = np.array([ca.pi/6, -ca.pi/6])
        a = np.array([1.0, 1.0])

        target, obstacle = sample_target_and_obstacle(x0, a)
        print(f"Attempting run {run+1}/{num_runs} with target: {target}, obstacle: {obstacle}")
        success, data = run_mpc_controller(mpc, simulator, p_template, p_template_sim, target, obstacle, x0=x0, a=a)

        if success:
            print(f"Run {run+1}/{num_runs} successful. Saving data...")
            data['run_number'] = run
            write_data_to_file(data, 'data_311_01_30.h5')
            # "date_month_day_#dataset_run.h5"
            run += 1
        else:
            print(f"Run {run+1} failed. Retrying with a new target and obstacle...")
