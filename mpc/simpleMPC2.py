import numpy as np
import do_mpc
from do_mpc.data import save_results, load_results
import casadi as ca

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter

from utils.modelling import link_points
from utils.utils import fk, jacobian, dist_obstacle_to_links
from visualization.visualize import visualize, save_animation

"""
Update simpleMPC.py to have the target and obstacle positions as parameters for easier data generation
use p template
"""

def mpc_controller(a1 = 1.0, a2 = 1.0, Ts=0.1):

    model = do_mpc.model.Model('discrete')

    # States (x)
    # joint angles
    theta1 = model.set_variable(var_type='_x', var_name='theta1')
    theta2 = model.set_variable(var_type='_x', var_name='theta2')

    # Control inputs (u)
    # joint velocities
    u1 = model.set_variable(var_type='_u', var_name='u1')
    u2 = model.set_variable(var_type='_u', var_name='u2')   

    # Parameters
    # target and goal position that will be set outside the MPC loop for data generation
    target = model.set_variable('_p', 'target', (2,1))
    obstacle = model.set_variable('_p', 'obstacle', (2,1))

    # Dynamics
    model.set_rhs('theta1', theta1 + u1*Ts)
    model.set_rhs('theta2', theta2 + u2*Ts)

    model.setup()
