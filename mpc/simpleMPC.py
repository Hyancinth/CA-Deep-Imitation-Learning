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

def mpc_controller(a1 = 1.0, a2 = 1.0, Ts=0.1, target=np.array([1.5, 1]), obstacle=np.array([1.0, 0.8])):
    """
    Create and setup MPC controller for 2-link RR manipulator
    """
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

    model.setup() 

    # MPC controller
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_horizon': 25, 
        't_step': Ts, 
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.max_cpu_time': 0.05} # set maximum solve time for each MPC step (in seconds)
    }

    mpc.set_param(**setup_mpc)


    # Distances
    pos = fk([model.x['theta1'], model.x['theta2']], [a1, a2]) # joint positions
    x1 = pos[0]
    y1 = pos[1]
    x2 = pos[2]
    y2 = pos[3]

    # generate points along the links for SSM constraints
    n_points = 4
    mid_link_points = link_points([model.x['theta1'], model.x['theta2']], [a1, a2], n_points = n_points)

    ee_dist_to_obs = ca.sqrt((x2 - obstacle[0])**2 + (y2 - obstacle[1])**2)
    ee_dist_to_target = ca.sqrt((x2 - target[0])**2 + (y2 - target[1])**2)
    j1_dist_to_obs = ca.sqrt((x1 - obstacle[0])**2 + (y1 - obstacle[1])**2)

    # Cost function 

    # Tuning parameters
    alpha = 0.89 # SSM 
    d_ = 0.05 # distance leeway for SSM

    gamma = 5 # repulsion weight in cost function
    beta = 2 # weight for repulsion function

    epsilon = 1e-6 # prevent division by zero
    mu = ca.exp(-beta*(ee_dist_to_obs**2/(ee_dist_to_target**2 + epsilon)))
    
    # cost = (model.u['u1']**2 + model.u['u2']**2) + gamma*mu**2
    # mterm = 200.0 * ee_dist_to_target**2   
    cost = ee_dist_to_target**2 + (model.u['u1']**2 + model.u['u2']**2) + gamma*mu**2
    mterm = ca.DM.zeros(1,1)
    mpc.set_objective(lterm=cost, mterm=mterm)
    mpc.set_rterm(u1=1e-6, u2=1e-6) # the cost function should penalize control inputs already

    # Modeling
    obs_radius = 0.1
    joint_radius = 0.1

    # Constraints

    # SSM constraints
    j1, j2 = jacobian([model.x['theta1'], model.x['theta2']], [a1, a2])
    v_j1 = j1 @ ca.vertcat(model.u['u1'], model.u['u2']) # cartesian velocity vector of joint 1
    v_j2 = j2 @ ca.vertcat(model.u['u1'], model.u['u2']) # cartesian velocity vector of end effector

    # zeta_1 = alpha**2 *(j1_dist_to_obs**2 - (obs_radius + joint_radius + d_)**2) # SSM constraint for joint 1
    # zeta_2 = alpha**2 *(ee_dist_to_obs**2 - (obs_radius + joint_radius + d_)**2) # SSM constraint for end effector

    min_zeta = 1e-4 
    zeta_1 = ca.fmax(alpha**2 * (j1_dist_to_obs**2 - (obs_radius + joint_radius + d_)**2), min_zeta)
    zeta_2 = ca.fmax(alpha**2 * (ee_dist_to_obs**2 - (obs_radius + joint_radius + d_)**2), min_zeta)

    v1 = v_j1[0]**2 + v_j1[1]**2 # velocity magnitude squared of joint 1
    v_ee = v_j2[0]**2 + v_j2[1]**2 # velocity magnitude squared of end effector

    mpc.set_nl_cons('ssm_joint1', v1 - zeta_1, ub = 0) # v^2 - zeta <= 0 -> v^2 <= zeta, velocity is reduced as the robot gets closer to the obstacle
    mpc.set_nl_cons('ssm_ee', v_ee - zeta_2, ub = 0)

    for id, (px, py) in enumerate(mid_link_points): # SSM constraints for points along the links
        point_dist_to_obs = ca.sqrt((px - obstacle[0])**2 + (py - obstacle[1])**2)

        if id < n_points: 
            v_point = v1 # approximate velocity magnitude of points along link 1 using velocity magnitude of joint 1
        else:
            v_point = v_ee # approximate velocity magnitude of points along link 2 using velocity magnitude of end effector

        zeta_point = ca.fmax(alpha**2 * (point_dist_to_obs**2 - (obs_radius + joint_radius + d_)**2), min_zeta)
        mpc.set_nl_cons(f'ssm_point_{id}', v_point - zeta_point, ub = 0)

    # Ground constraint (prevent robot from hitting the ground)
    mpc.set_nl_cons('ground1', -y1, ub = -0.1) 
    mpc.set_nl_cons('ground2', -y2, ub = -0.1)

    # joint angle constraints
    mpc.bounds['lower','_x','theta1'] = -ca.pi # rad
    mpc.bounds['upper','_x','theta1'] = ca.pi

    mpc.bounds['lower','_x','theta2'] = -ca.pi # rad
    mpc.bounds['upper','_x','theta2'] = ca.pi

    # joint velocity constraints
    mpc.bounds['lower','_u','u1'] = -3.0 # rad/s
    mpc.bounds['upper','_u','u1'] = 3.0 
    
    mpc.bounds['lower','_u','u2'] = -3.0 # rad/s
    mpc.bounds['upper','_u','u2'] = 3.0 

    mpc.setup()

    # Simulator
    simulator = do_mpc.simulator.Simulator(model)
    simulator.settings.t_step = Ts
    simulator.setup()

    print("MPC and Simulator setup complete.")
    return mpc, simulator   

def simulate(target, obstacle):
    """
    Simulate the robot motion using the MPC controller and simulator, given a target position and obstacle position
    """
    mpc, simulator = mpc_controller(target=target, obstacle=obstacle)

    simulator.reset_history()
    mpc.reset_history()
    
    x0 = np.array([ca.pi/6, -ca.pi/6]) # initial joint angles

    # set initial state for both MPC and simulator
    mpc.x0 = x0
    simulator.x0 = x0

    mpc.set_initial_guess() # initial guess for optimization variables using x0, u0

    # open loop simulation
    # u0 = mpc.make_step(x0)

    # simulate for 100 steps
    for i in range(100): 
        u0 = mpc.make_step(x0)
        x0 = simulator.make_step(u0)
    
    # save_results([mpc, simulator])

    return mpc, simulator

if __name__ == "__main__":
    target = np.array([1.0, 1.0])
    obstacle = np.array([1.4, 1.0])

    init_theta = np.array([ca.pi/6, -ca.pi/6])
    a = np.array([1.0, 1.0])

    dist = dist_obstacle_to_links(obstacle, init_theta, a)
    sphere_dist = dist - (0.1 + 0.1) # distance from obstacle to robot links minus radius of obstacle and robot joints
    if sphere_dist < 0.1:
        print(f"Warning: Obstacle is too close to the robot's initial configuration: {sphere_dist:.3f}m from the links. Consider moving the obstacle or adjusting the initial joint angles.")
    else:
        mpc, simulator = simulate(target, obstacle)
        anim = visualize(mpc, simulator, target, obstacle)
    
    # save_animation(anim, 'mpc_animation.gif')
