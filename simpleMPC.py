import numpy as np
import do_mpc
import casadi as ca

def fk(theta, a):
    """
    Calculate forward kinematics for a 2-link RR manipulator
    """
    a1 = a[0]
    a2 = a[1]

    theta1 = theta[0]
    theta2 = theta[1]

    x1 = a1 * ca.cos(theta1)
    y1 = a1 * ca.sin(theta1)

    x2 = x1 + a2*ca.cos(theta1 + theta2)
    y2 = y1 + a2*ca.sin(theta1 + theta2)

    return [x1, y1, x2, y2]

def mpc_controller(a1 = 1.0, a2 = 1.0, Ts=0.05, target=np.array([1.5, 1.5]), obstacle=np.array([1.0, 1.0])):
    model = do_mpc.model.Model('discrete')

    # States (x)
    # joint angles
    theta1 = model.set_variable(var_type='_x', var_name='theta1')
    theta2 = model.set_variable(var_type='_x', var_name='theta2')

    # Control inputs (u)
    # joint velocities
    u1 = model.set_variable(var_type='_u', var_name='u1')
    u2 = model.set_variable(var_type='_u', var_name='u2')   

    # Dynamics
    model.set_rhs('theta1', theta1 + u1*Ts)
    model.set_rhs('theta2', theta2 + u2*Ts)

    # Parameters
    alpha = 0.89 # SSM 
    d_ = 0.3 # distance leeway for SSM

    gamma = 500 # repulsion weight in cost function
    beta = 3 # weight for repulsion function

    model.setup() 

    # MPC controller
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_horizon': 10, 
        't_step': Ts, 
        'store_full_solution': True,
    }

    mpc.set_param(**setup_mpc)
    
    # Distances
    pos = fk([model.aux['theta1'], model.aux['theta2']], [a1, a2]) # joint positions
    x1 = pos[0]
    y1 = pos[1]
    x2 = pos[2]
    y2 = pos[3]

    ee_dist_to_obs = ca.sqrt((x2 - obstacle[0])**2 + (y2 - obstacle[1])**2)
    ee_dist_to_target = ca.sqrt((x2 - target[0])**2 + (y2 - target[1])**2)

    j1_dist_to_obs = ca.sqrt((x1 - obstacle[0])**2 + (y1 - obstacle[1])**2)

    # Cost function 
    epsilon = 1e-6 # prevent division by zero
    mu = ca.exp(-beta*(ee_dist_to_obs**2/(ee_dist_to_target**2 + epsilon)))
    cost = ee_dist_to_target**2 + (model.aux['u1']**2 + model.aux['u2']**2) + gamma*mu**2
    mpc.set_objective(cost)

    # Constraints
    obs_radius = 0.2
    joint_radius = 0.1

    ssm_1 = alpha**2 *(j1_dist_to_obs**2 - (obs_radius + joint_radius + d_)**2)
    ssm_2 = alpha**2 *(ee_dist_to_obs**2 - (obs_radius + joint_radius + d_)**2)
    
mpc_controller()