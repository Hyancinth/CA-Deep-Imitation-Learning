import numpy as np
import do_mpc
from do_mpc.data import save_results, load_results
import casadi as ca

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter

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

def jacobian(theta, a = np.array([1.0, 1.0])):
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

def mpc_controller(a1 = 1.0, a2 = 1.0, Ts=0.1, target=np.array([0.5, 1]), obstacle=np.array([1.0, 0.8])):
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
    }

    mpc.set_param(**setup_mpc)
    
    # Tuning parameters
    alpha = 0.89 # SSM 
    d_ = 0.05 # distance leeway for SSM

    gamma = 5 # repulsion weight in cost function
    beta = 2 # weight for repulsion function


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

    mpc.set_nl_cons('ssm1', v1 - zeta_1, ub = 0)
    mpc.set_nl_cons('ssm2', v_ee - zeta_2, ub = 0)

    # Ground constraint (prevent robot from going below the ground)
    mpc.set_nl_cons('ground1', -y1, ub = -0.1)
    mpc.set_nl_cons('ground2', -y2, ub = -0.1)

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


def simulate():
    mpc, simulator = mpc_controller()

    simulator.reset_history()
    mpc.reset_history()
    
    x0 = np.array([ca.pi/6, -ca.pi/6]) # initial joint angles

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

    return mpc, simulator

def robot_motion(mpc, simulator):
    # extract data
    theta1_sim = simulator.data['_x', 'theta1']
    print(len(theta1_sim))
    theta2_sim = simulator.data['_x', 'theta2']

    theta1_mpc = mpc.data['_x', 'theta1']
    print(len(theta1_mpc))
    theta2_mpc = mpc.data['_x', 'theta2']

    # store joint positions
    x1_sim = []
    y1_sim = []
    x2_sim = []
    y2_sim = []

    x1_mpc = []
    y1_mpc = []
    x2_mpc = []
    y2_mpc = []

    # loop through joint angles and calculate joint positions using forward kinematics
    for t1, t2 in zip(theta1_sim, theta2_sim):
        t1 = t1[0]
        t2 = t2[0]

        pos = fk([t1, t2], [1.0, 1.0])
        x1_sim.append(pos[0])
        y1_sim.append(pos[1])
        x2_sim.append(pos[2])
        y2_sim.append(pos[3])
    
    for t1, t2 in zip(theta1_mpc, theta2_mpc):
        t1 = t1[0]
        t2 = t2[0]

        pos = fk([t1, t2], [1.0, 1.0])
        x1_mpc.append(pos[0])
        y1_mpc.append(pos[1])
        x2_mpc.append(pos[2])
        y2_mpc.append(pos[3])

    return {'x1_sim': x1_sim, 'y1_sim': y1_sim, 'x2_sim': x2_sim, 'y2_sim': y2_sim,
            'x1_mpc': x1_mpc, 'y1_mpc': y1_mpc, 'x2_mpc': x2_mpc, 'y2_mpc': y2_mpc}

def visualize(mpc, simulator):
    
    mpc_graphics = do_mpc.graphics.Graphics(mpc.data) # results from MPC 
    # sim_graphics = do_mpc.graphics.Graphics(simulator.data)

    fig = plt.figure(figsize=(16,9))
    num_plot_rows = 4
    num_plot_cols = 2
    ax1 = plt.subplot2grid((num_plot_rows, num_plot_cols), (0, 0), rowspan=4) # mpc predicted
    # ax2 = plt.subplot2grid((num_plot_rows, num_plot_cols), (0, 1), rowspan = 4) # simulated
    ax3 = plt.subplot2grid((num_plot_rows, num_plot_cols), (0, num_plot_cols-1)) 
    ax4 = plt.subplot2grid((num_plot_rows, num_plot_cols), (1, num_plot_cols-1)) 
    ax5 = plt.subplot2grid((num_plot_rows, num_plot_cols), (2, num_plot_cols-1)) 
    ax6 = plt.subplot2grid((num_plot_rows, num_plot_cols), (3, num_plot_cols-1))

    ax3.set_ylabel('theta1 (rad)')
    ax4.set_ylabel('theta2 (rad)')
    ax5.set_ylabel('u1 (rad/s)')
    ax6.set_ylabel('u2 (rad/s)')

    # axis on the right
    for ax in [ax3, ax4, ax5, ax6]:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        if ax!= ax6:
            ax.xaxis.set_ticklabels([])

    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    # ax2.set_xlim(-0.5, 2.5)
    # ax2.set_ylim(-0.5, 2.5)
    ax6.set_xlabel('time (s)')

    for g in [mpc_graphics]: 
        g.add_line(var_type='_x', var_name='theta1', axis=ax3   , color = '#1f77b4')
        g.add_line(var_type='_x', var_name='theta2', axis=ax4, color = '#ff7f0e')
        g.add_line(var_type='_u', var_name='u1', axis=ax5, color = '#1f77b4')
        g.add_line(var_type='_u', var_name='u2', axis=ax6, color = '#ff7f0e')
    
    for line_i in mpc_graphics.pred_lines['_x', 'theta1']: line_i.set_color('#1f77b4')
    for line_i in mpc_graphics.pred_lines['_x', 'theta2']: line_i.set_color('#ff7f0e')
    for line_i in mpc_graphics.pred_lines['_u', 'u1']: line_i.set_color('#1f77b4')
    for line_i in mpc_graphics.pred_lines['_u', 'u2']: line_i.set_color('#ff7f0e')

    for line_i in mpc_graphics.pred_lines.full: line_i.set_alpha(0.2)

    link1_mpc = ax1.plot([], [], 'o-', lw = 4, color='#1f77b4')
    link2_mpc = ax1.plot([], [], 'o-', lw = 4, color='#ff7f0e')

    # link1_sim = ax2.plot([], [], 'o-', lw = 4, color='#1f77b4')
    # link2_sim = ax2.plot([], [], 'o-', lw = 4, color='#ff7f0e')

    fig.align_ylabels()
    fig.tight_layout()

    obs_radius = 0.1
    obs_center = (1.0, 0.8)
    obs_circle_mpc = plt.Circle(obs_center, obs_radius, color='r', alpha=0.5)
    # obs_circle_sim = plt.Circle(obs_center, obs_radius, color='r', alpha=0.5)
    ax1.add_patch(obs_circle_mpc)
    # ax2.add_patch(obs_circle_sim)

    goal_center = (0.5, 1.0)
    goal_circle = plt.Circle(goal_center, 0.01, color='b', alpha=0.5)
    ax1.add_patch(goal_circle)

    robot_motion_data = robot_motion(mpc, simulator)
    # x1_sim = robot_motion_data['x1_sim']
    # y1_sim = robot_motion_data['y1_sim']
    # x2_sim = robot_motion_data['x2_sim']
    # y2_sim = robot_motion_data['y2_sim']
    x1_mpc = robot_motion_data['x1_mpc']
    y1_mpc = robot_motion_data['y1_mpc']
    x2_mpc = robot_motion_data['x2_mpc']
    y2_mpc = robot_motion_data['y2_mpc']

    circles = []
    def update(i):
        for circle in circles:
            circle.remove()
        circles.clear()

        joint_radius = 0.1
        mpc_circle_ee = plt.Circle((x2_mpc[i], y2_mpc[i]), joint_radius, color='g', alpha=0.5)
        mpc_circle_j1 = plt.Circle((x1_mpc[i], y1_mpc[i]), joint_radius, color='g', alpha=0.5)
        ax1.add_patch(mpc_circle_ee)
        ax1.add_patch(mpc_circle_j1)

        # sim_circle_ee = plt.Circle((x2_sim[i], y2_sim[i]), joint_radius, color='g', alpha=0.5)
        # sim_circle_j1 = plt.Circle((x1_sim[i], y1_sim[i]), joint_radius, color='g', alpha=0.5)
        # ax2.add_patch(sim_circle_ee)
        # ax2.add_patch(sim_circle_j1)

        circles.extend([mpc_circle_ee, mpc_circle_j1])

        link1_mpc[0].set_data([0, x1_mpc[i]], [0, y1_mpc[i]])
        link2_mpc[0].set_data([x1_mpc[i], x2_mpc[i]], [y1_mpc[i], y2_mpc[i]])
        # link1_sim[0].set_data([0, x1_sim[i]], [0, y1_sim[i]])
        # link2_sim[0].set_data([x1_sim[i], x2_sim[i]], [y1_sim[i], y2_sim[i]])

        mpc_graphics.plot_predictions(i)
        # sim_graphics.plot_results(i)
        mpc_graphics.plot_results(i)
        mpc_graphics.reset_axes()
        # sim_graphics.reset_axes()

    anim = FuncAnimation(fig, update, frames = len(x1_mpc),  repeat = True)

    plt.show()

    return anim

def save_animation(anim, filename):
    gif_writer = ImageMagickWriter(fps=20)
    try:
        anim.save(filename, writer=gif_writer)
    except Exception as e:
        print(f"Error saving animation: {e}")
    

if __name__ == "__main__":
    mpc, simulator = simulate()
    anim = visualize(mpc, simulator)
    # save_animation(anim, 'mpc_animation.gif')
