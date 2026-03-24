import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils.modelling import link_points
from utils.utils import robot_motion_from_data

def get_data_from_h5(h5_path, run_i):
    """
    Extract data for a specific run from the h5 file and return it as numpy arrays
    """
    with h5.File(h5_path, 'r') as f:
        run_key = "run_" + str(run_i)
        run_data = f[run_key]

        # extract data
        theta1 = run_data['theta1'][:]
        theta2 = run_data['theta2'][:]
        if run_data['target_x'].shape == (): # handle scalar case (old data format)
            target = np.array([float(run_data['target_x'][()]), float(run_data['target_y'][()])])
            obstacle = np.array([float(run_data['obstacle_x'][()]), float(run_data['obstacle_y'][()])])
        else:
            obstacle = np.array([float(run_data['obstacle_x'][0]), float(run_data['obstacle_y'][0])])
            target   = np.array([float(run_data['target_x'][0]), float(run_data['target_y'][0])])
        u1 = run_data['u1'][:]
        u2 = run_data['u2'][:]

    f.close()
    return theta1, theta2, obstacle, target, u1, u2

def visualize_h5(h5_path, obs_radius=0.1, joint_radius=0.1):
    """
    Visualize robot motion, joint angles, and joint velocities from h5 file
    """
    with h5.File(h5_path, 'r') as f:
        num_runs = len(f.keys())
    f.close()
    
    for run_i in range(num_runs):
        theta1, theta2, obstacle, target, u1, u2 = get_data_from_h5(h5_path, run_i)
        robot_motion = robot_motion_from_data(theta1, theta2)

        print(obstacle, target)
        x1 = robot_motion["x1"]
        y1 = robot_motion["y1"]
        x2 = robot_motion["x2"]
        y2 = robot_motion["y2"]

        n_steps = len(theta1)
        time = np.arange(n_steps)*0.1

        fig = plt.figure(figsize=(16, 9))
        num_plot_rows, num_plot_cols = 4, 2

        # plt.title(f'Run {run_i}')
        fig.suptitle(f'Run {run_i}')

        ax1 = plt.subplot2grid((num_plot_rows, num_plot_cols), (0, 0), rowspan=4)
        ax3 = plt.subplot2grid((num_plot_rows, num_plot_cols), (0, 1))
        ax4 = plt.subplot2grid((num_plot_rows, num_plot_cols), (1, 1), sharex=ax3)
        ax5 = plt.subplot2grid((num_plot_rows, num_plot_cols), (2, 1), sharex=ax3)
        ax6 = plt.subplot2grid((num_plot_rows, num_plot_cols), (3, 1), sharex=ax3)

        ax3.set_ylabel('theta1 (rad)')
        ax4.set_ylabel('theta2 (rad)')
        ax5.set_ylabel('u1 (rad/s)')
        ax6.set_ylabel('u2 (rad/s)')
        ax6.set_xlabel('time step')

        for ax in [ax3, ax4, ax5, ax6]:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            if ax != ax6:  
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        ax6.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        ax6.set_xlim(0, time[-1]+0.2)
        ax6.set_xticks(np.arange(0, time[-1] + 0.2, 2.0))

        # plot full time-series in the right column (static)
        ax3.plot(time, theta1, color='#1f77b4')
        ax4.plot(time, theta2, color='#ff7f0e')
        ax5.plot(time, u1,     color='#1f77b4')
        ax6.plot(time, u2,     color='#ff7f0e')

        ax1.set_xlim(-2.5, 2.5)
        ax1.set_ylim(-2.5, 2.5)
        ax6.set_xlabel('time (s)')      

        fig.align_ylabels()
        # fig.tight_layout()
        fig.tight_layout(pad=1.5, h_pad=0.5)

        ax1.hlines(0, -2.5, 2.5, colors='k')

        obs_circle = plt.Circle((obstacle[0], obstacle[1]), obs_radius, color='r', alpha=0.5)
        ax1.add_patch(obs_circle)

        target_circle = plt.Circle((target[0], target[1]), 0.05, color='b', alpha=0.5)
        ax1.add_patch(target_circle)

            
        # animated vertical cursor lines
        vlines = [ax.axvline(0, color='k', lw=1, ls='--') for ax in [ax3, ax4, ax5, ax6]]
        
        link1_line, = ax1.plot([], [], 'o-', lw=4, color='#1f77b4')
        link2_line, = ax1.plot([], [], 'o-', lw=4, color='#ff7f0e')

        circles = []
        animations = []

        def update(i):
            # clear previous dynamic circles
            for c in circles:
                c.remove()
            circles.clear()

            # joint / link circles
            circle_ee = plt.Circle((x2[i], y2[i]), joint_radius, color='m', alpha=0.5)
            circle_j1 = plt.Circle((x1[i], y1[i]), joint_radius, color='g', alpha=0.5)
            ax1.add_patch(circle_ee)
            ax1.add_patch(circle_j1)
            circles.extend([circle_ee, circle_j1])

            mid_link_points = link_points([theta1[i], theta2[i]], [1.0, 1.0], n_points=4)
            for px, py in mid_link_points:
                link_circle = plt.Circle((float(px), float(py)), joint_radius, color='g', alpha=0.5)
                ax1.add_patch(link_circle)
                circles.append(link_circle)

            # robot links
            link1_line.set_data([0, x1[i]], [0, y1[i]])
            link2_line.set_data([x1[i], x2[i]], [y1[i], y2[i]])

            # cursor lines
            for vl in vlines:
                vl.set_xdata([time[i], time[i]])

            # plt.title.set_text(f'step {i}')

        anim = FuncAnimation(fig, update, frames=n_steps, repeat=True, interval=80)

        fig.align_ylabels()
        fig.tight_layout()
        plt.show()
        
        animations.append(anim)
        del anim

    return animations

if __name__ == "__main__":
<<<<<<< HEAD
    h5_path = "model/data/hidden_test_data.h5"
=======
    h5_path = "model/hidden_test_data/hidden_test_data_2.h5"
>>>>>>> testing/basic-dnn
    visualize_h5(h5_path)