import numpy as np

"""
This script generates data from the MPC controller and saves it to a file for later use in training the imitation learning model.

mpc.data.data_fields: data
{'_time': 1, '_x': 2, '_y': 2, '_u': 2, '_z': 0, '_tvp': 0, '_p': 0, '_aux': 1, '_eps': 0, 'opt_p_num': 4, '_opt_x_num': 102, '_opt_aux_num': 25, '_lam_g_num': 352, 'success': 1, 't_wall_total': 1}

Use: mpc.solver_stats["success"] to check if the MPC solver succeeds at finding a solution
Boolean: True if it succeeds, False if it fails 
"""