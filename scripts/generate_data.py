import numpy as np
import casadi as ca

"""
This script generates data from the MPC controller and saves it to a file for later use in training the imitation learning model.

mpc.data.data_fields: data
{'_time': 1, '_x': 2, '_y': 2, '_u': 2, '_z': 0, '_tvp': 0, '_p': 0, '_aux': 1, '_eps': 0, 'opt_p_num': 4, '_opt_x_num': 102, '_opt_aux_num': 25, '_lam_g_num': 352, 'success': 1, 't_wall_total': 1}

Use: mpc.solver_stats["success"] to check if the MPC solver succeeds at finding a solution
Boolean: True if it succeeds, False if it fails 

t_wall_total: total time taken by the MPC solver to find a solution at each step (in seconds)

how to run: python -m scripts.generate_data
"""

from data.write_data import generate_goal_point, generate_obstacle_point, generate_data, write_data_to_file
from mpc.simpleMPC2 import mpc_controller

if __name__ == "__main__":
    mpc, simulator, p_template, p_template_sim = mpc_controller()

    num_runs = 2
    run = 0

    max_attempts_per_run = 5
    attempts = 0

    while run < num_runs:
        x0 = np.array([ca.pi/6, -ca.pi/6])
        a = np.array([1.0, 1.0])
        target = np.array(generate_goal_point())
        obstacle = np.array(generate_obstacle_point(x0, a))
        print(f"Run {run}: Target: {target}, Obstacle: {obstacle}")

        p_template['_p', 0, 'target'] = target.reshape(2,1)
        p_template['_p', 0, 'obstacle'] = obstacle.reshape(2,1)

        p_template_sim['target'] = target.reshape(2,1)
        p_template_sim['obstacle'] = obstacle.reshape(2,1)

        simulator.reset_history()
        mpc.reset_history()

        mpc.x0 = x0
        simulator.x0 = x0

        mpc.set_initial_guess()

        for i in range(100): 
            u0 = mpc.make_step(x0)
            x0 = simulator.make_step(u0)
            if not mpc.solver_stats["success"]:
                print(f"Run {run} failed at step {i}. Retrying with a new target and obstacle.")
                break # exit the loop and retry with a new target and obstacle if MPC solver fails
        
        if not mpc.solver_stats["success"]:
            continue

        # if mpc.solver_stats["success"]:
            # also need to check if the final state is close enough to the target to consider it a successful run for data generation
        final_state = mpc.data['_x', 'theta1'][-1][0], mpc.data['_x', 'theta2'][-1][0]
        distance_to_target = np.linalg.norm(np.array(final_state) - np.array(target))
        
        if distance_to_target > 0.1:  # Adjust the threshold as needed
            print(f"Run {run} did not reach the target. Retrying with a new target and obstacle.")
            continue
    
        print(f"Run {run} succeeded. Generating data...")
        data = generate_data(mpc, target, obstacle, a)
        data['run_number'] = run # add run number to data for tracking

        write_data_to_file(data, 'training_data_test_2.h5')
        run += 1 # only increment run if MPC solver succeeds
        # else:
        #     print(f"Run {run} failed to find a solution. Retrying with a new target and obstacle.")