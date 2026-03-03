import numpy as np
import do_mpc
from do_mpc.data import save_results, load_results
import casadi as ca

import matplotlib.pyplot as plt
import matplotlib as mpl

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

def jacobian(theta, a):
    """
    Calculate jacobian matrix for each joint of a 2-link RR manipulator
    """
    a1 = a[0]
    a2 = a[1]
    theta1 = theta[0]
    theta2 = theta[1]

    j1 = ca.vertcat(
        ca.horzcat(-a1*ca.sin(theta1), 0),
        ca.horzcat(a1*ca.cos(theta1), 0)
    )

    j2 = ca.vertcat(
        ca.horzcat(-a1*ca.sin(theta1) - a2*ca.sin(theta1 + theta2), -a2*ca.sin(theta1 + theta2)),
        ca.horzcat(a1*ca.cos(theta1) + a2*ca.cos(theta1 + theta2), a2*ca.cos(theta1 + theta2))
    )

    return j1, j2

save_pos = []

def mpc_controller(a1 = 1.0, a2 = 1.0, Ts=0.1, target=np.array([1.5, 0.5]), obstacle=np.array([1.0, 0.3])):
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
        'n_horizon': 100, 
        't_step': Ts, 
        'store_full_solution': True,
    }

    mpc.set_param(**setup_mpc)
    
    # Tuning parameters
    alpha = 0.89 # SSM 
    d_ = 0.05 # distance leeway for SSM

    gamma = 10 # repulsion weight in cost function
    beta = 3 # weight for repulsion function


    # Distances
    pos = fk([model.x['theta1'], model.x['theta2']], [a1, a2]) # joint positions
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
    
    cost = ee_dist_to_target**2 + (model.u['u1']**2 + model.u['u2']**2) + gamma*mu**2
    mterm = ca.DM.zeros(1,1)
    mpc.set_objective(lterm=cost, mterm=mterm)

    mpc.set_rterm(u1=0.1, u2=0.1)

    # Modeling
    obs_radius = 0.1
    joint_radius = 0.2

    # Constraints

    # SSM constraints
    j1, j2 = jacobian([model.x['theta1'], model.x['theta2']], [a1, a2])
    v_j1 = j1 @ ca.vertcat(model.u['u1'], model.u['u2']) # cartesian velocity vector of joint 1
    v_j2 = j2 @ ca.vertcat(model.u['u1'], model.u['u2']) # cartesian velocity vector of end effector

    zeta_1 = alpha**2 *(j1_dist_to_obs**2 - (obs_radius + joint_radius + d_)**2) # SSM constraint for joint 1
    zeta_2 = alpha**2 *(ee_dist_to_obs**2 - (obs_radius + joint_radius + d_)**2) # SSM constraint for end effector

    v1 = v_j1[0]**2 + v_j1[1]**2 # velocity magnitude squared of joint 1
    v_ee = v_j2[0]**2 + v_j2[1]**2 # velocity magnitude squared of end effector

    mpc.set_nl_cons('ssm1', v1 - zeta_1, ub = 0)
    mpc.set_nl_cons('ssm2', v_ee - zeta_2, ub = 0)

    # joint angle constraints
    mpc.bounds['lower','_x','theta1'] = -ca.pi
    mpc.bounds['upper','_x','theta1'] = ca.pi

    mpc.bounds['lower','_x','theta2'] = -ca.pi
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

# def robot_link_lines(x):
#     x = x.flatten()
#     a = [1.0, 1.0]


def simulate_and_visualize():
    mpc, simulator = mpc_controller()

    simulator.reset_history()
    mpc.reset_history()
    
    x0 = np.array([ca.pi/6, ca.pi/6]) # initial joint angles

    mpc.x0 = x0
    simulator.x0 = x0

    mpc.set_initial_guess()

    # open loop simulation
    # u0 = mpc.make_step(x0)

    # Simulate
    for i in range(100):
        u0 = mpc.make_step(x0)
        x0 = simulator.make_step(u0)
    
    # save_results([mpc, simulator])
    
    mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
    sim_graphics = do_mpc.graphics.Graphics(simulator.data)

    fig, ax = plt.subplots(2, sharex = True, figsize=(16,9))
    fig.align_ylabels()

    for g in [sim_graphics, mpc_graphics]:
        g.add_line(var_type='_x', var_name='theta1', axis=ax[0], color = '#1f77b4')
        g.add_line(var_type='_x', var_name='theta2', axis=ax[0], color = '#ff7f0e')
        g.add_line(var_type='_u', var_name='u1', axis=ax[1], color = '#1f77b4')
        g.add_line(var_type='_u', var_name='u2', axis=ax[1], color = '#ff7f0e')
    
    ax[0].set_ylabel('Joint Angles (rad)')
    ax[1].set_ylabel('Joint Velocities (rad/s)')
    ax[1].set_xlabel('Time (s)')

    for line_i in mpc_graphics.pred_lines['_x', 'theta1']: line_i.set_color('#1f77b4')
    for line_i in mpc_graphics.pred_lines['_x', 'theta2']: line_i.set_color('#ff7f0e')
    for line_i in mpc_graphics.pred_lines['_u', 'u1']: line_i.set_color('#1f77b4')
    for line_i in mpc_graphics.pred_lines['_u', 'u2']: line_i.set_color('#ff7f0e')

    for line_i in mpc_graphics.pred_lines.full: line_i.set_alpha(0.2)

    lines = sim_graphics.result_lines['_x', 'theta1'] + sim_graphics.result_lines['_x', 'theta2']
    ax[0].legend(lines, ['theta1', 'theta2'], title = 'Joint Angles (rad)')

    lines = sim_graphics.result_lines['_u', 'u1'] + sim_graphics.result_lines['_u', 'u2']
    ax[1].legend(lines, ['u1', 'u2'], title = 'Joint Velocities (rad/s)')

    mpc_graphics.plot_predictions(t_ind=0)
    sim_graphics.plot_results()
    sim_graphics.reset_axes()
    plt.show()

if __name__ == "__main__":
    simulate_and_visualize()