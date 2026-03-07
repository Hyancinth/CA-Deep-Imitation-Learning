import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter
import do_mpc
from utils import robot_motion
from modelling import link_points

def visualize(mpc, simulator, target=np.array([1.5, 0.5]), obstacle=np.array([1.0, 0.8]), obs_radius=0.1, joint_radius=0.1):
    """
    Visualize predicted robot motion, joint angles and joint velocities evolution over time from the MPC
    """
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

    for g in [mpc_graphics]: # add MPC graphics to the right axes (if you want to also do simulation graphics, add it to the list)
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

    ax1.hlines(0, -2.5, 2.5, colors='k')

    obs_center = (obstacle[0], obstacle[1])
    obs_circle_mpc = plt.Circle(obs_center, obs_radius, color='r', alpha=0.5)
    # obs_circle_sim = plt.Circle(obs_center, obs_radius, color='r', alpha=0.5)
    ax1.add_patch(obs_circle_mpc)
    # ax2.add_patch(obs_circle_sim)

    target_center = (target[0], target[1])
    target_circle = plt.Circle(target_center, 0.01, color='b', alpha=0.5)
    ax1.add_patch(target_circle)

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

        mpc_circle_ee = plt.Circle((x2_mpc[i], y2_mpc[i]), joint_radius, color='g', alpha=0.5)
        mpc_circle_j1 = plt.Circle((x1_mpc[i], y1_mpc[i]), joint_radius, color='g', alpha=0.5)
        ax1.add_patch(mpc_circle_ee)
        ax1.add_patch(mpc_circle_j1)

        circles.extend([mpc_circle_ee, mpc_circle_j1])

        mid_link_points = link_points([mpc.data['_x', 'theta1'][i][0], mpc.data['_x', 'theta2'][i][0]], [1.0, 1.0], n_points = 4)
        for px, py in mid_link_points:
            px = float(px)
            py = float(py)
            link_circle = plt.Circle((px, py), joint_radius, color='g', alpha=0.5)
            ax1.add_patch(link_circle)
            circles.append(link_circle)

        # sim_circle_ee = plt.Circle((x2_sim[i], y2_sim[i]), joint_radius, color='g', alpha=0.5)
        # sim_circle_j1 = plt.Circle((x1_sim[i], y1_sim[i]), joint_radius, color='g', alpha=0.5)
        # ax2.add_patch(sim_circle_ee)
        # ax2.add_patch(sim_circle_j1)

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
    """
    Saves animation to a gif
    """
    gif_writer = ImageMagickWriter(fps=20)
    try:
        anim.save(filename, writer=gif_writer)
    except Exception as e:
        print(f"Error saving animation: {e}")
