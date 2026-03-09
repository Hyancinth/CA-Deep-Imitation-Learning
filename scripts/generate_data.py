import numpy as np

"""
This script generates data from the MPC controller and saves it to a file for later use in training the imitation learning model.

mpc.data.data_fields: data
{'_time': 1, '_x': 2, '_y': 2, '_u': 2, '_z': 0, '_tvp': 0, '_p': 0, '_aux': 1, '_eps': 0, 'opt_p_num': 4, '_opt_x_num': 102, '_opt_aux_num': 25, '_lam_g_num': 352, 'success': 1, 't_wall_total': 1}

Use: mpc.solver_stats["success"] to check if the MPC solver succeeds at finding a solution
Boolean: True if it succeeds, False if it fails 

t_wall_total: total time taken by the MPC solver to find a solution at each step (in seconds)

how to run: python -m scripts.generate_data
"""

# test data generation
from data.write_data import generate_goal_point, generate_obstacle_point, generate_data, write_data_to_file
from mpc.simpleMPC import mpc_controller


import casadi as ca

if __name__ == "__main__":
    
    x0 = np.array([ca.pi/6, -ca.pi/6])
    a = np.array([1.0, 1.0])

    num_runs = 2
    for run in range(num_runs):
        target = generate_goal_point()
        obstacle = generate_obstacle_point(x0, a)
        print(f"Run {run}: Target: {target}, Obstacle: {obstacle}")

        mpc, simulator = mpc_controller(target=target, obstacle=obstacle)

        simulator.reset_history()
        mpc.reset_history()

        mpc.x0 = x0
        simulator.x0 = x0

        mpc.set_initial_guess()

        for i in range(20): 
            u0 = mpc.make_step(x0)
            x0 = simulator.make_step(u0)
        
        data = generate_data(mpc, target, obstacle, a)
        data['run_number'] = run # add run number to data for tracking

        write_data_to_file(data, 'training_data_test.h5')