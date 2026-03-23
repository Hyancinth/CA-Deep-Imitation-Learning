import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils.modelling import link_points
from utils.utils import robot_motion_from_data, fk

from matplotlib.animation import FuncAnimation, ImageMagickWriter
from matplotlib.animation import PillowWriter

from visualization.utils import get_data_from_h5, save_animation

def visualize_h5_comparison(mpc_h5_path, pred_h5_path, run_i=0, obs_radius=0.1, joint_radius=0.1, save_gif=False, gif_filename="robot_comparison.gif"):
    """
    Animate side-by-side comparison of ground truth (hidden H5) vs predicted (NN H5)

    Args:
        mpc_h5_path (str): path to ground truth hidden dataset
        pred_h5_path (str): path to predicted rollout H5 file
        run_i (int): run index to visualize
        obs_radius (float): obstacle radius
        joint_radius (float): radius for joint circles
        save_gif (bool): whether to save the animation as GIF
        gif_filename (str): filename for saved GIF
    """
    theta1_gt, theta2_gt, obstacle, target, u1_gt, u2_gt = get_data_from_h5(mpc_h5_path, run_i)
    
    robot_motion_gt = robot_motion_from_data(theta1_gt, theta2_gt)
    x1_gt, y1_gt = robot_motion_gt["x1"], robot_motion_gt["y1"]
    x2_gt, y2_gt = robot_motion_gt["x2"], robot_motion_gt["y2"]

    n_steps = len(theta1_gt)

    with h5.File(pred_h5_path, 'r') as f:
        run_key = f"run_{run_i}"
        run_data = f[run_key]
        x1_pred = run_data['joint1_x'][:]
        y1_pred = run_data['joint1_y'][:]
        x2_pred = run_data['ee_x'][:]
        y2_pred = run_data['ee_y'][:]
        u1_pred = run_data['u1'][:]
        u2_pred = run_data['u2'][:]
        target_pred = [float(run_data['target_x'][0]), float(run_data['target_y'][0])]
        obstacle_pred = [float(run_data['obstacle_x'][0]), float(run_data['obstacle_y'][0])]

    n_steps_pred = len(x1_pred)
    n_steps = min(n_steps, n_steps_pred)  

    fig = plt.figure(figsize=(18, 8))
    ax1 = plt.subplot(1, 2, 1) 
    ax2 = plt.subplot(1, 2, 2) 

    for ax, obs, targ in zip([ax1, ax2], [obstacle, obstacle_pred], [target, target_pred]):
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.hlines(0, -2.5, 2.5, colors='k')
        ax.add_patch(plt.Circle((obs[0], obs[1]), obs_radius, color='r', alpha=0.5))
        ax.add_patch(plt.Circle((targ[0], targ[1]), 0.05, color='b', alpha=0.5))

    ax1.set_title(f"MPC")
    ax2.set_title(f"Neural Network Prediction")

    link1_gt, = ax1.plot([], [], 'o-', lw=4, color='#1f77b4')
    link2_gt, = ax1.plot([], [], 'o-', lw=4, color='#ff7f0e')
    link1_pred, = ax2.plot([], [], 'o-', lw=4, color='#1f77b4')
    link2_pred, = ax2.plot([], [], 'o-', lw=4, color='#ff7f0e')

    circles_gt = []
    circles_pred = []

    def update(i):
        for c in circles_gt: c.remove()
        circles_gt.clear()
        circles_gt.extend([
            ax1.add_patch(plt.Circle((x1_gt[i], y1_gt[i]), joint_radius, color='g', alpha=0.5)),
            ax1.add_patch(plt.Circle((x2_gt[i], y2_gt[i]), joint_radius, color='m', alpha=0.5))
        ])
        link1_gt.set_data([0, x1_gt[i]], [0, y1_gt[i]])
        link2_gt.set_data([x1_gt[i], x2_gt[i]], [y1_gt[i], y2_gt[i]])

        for c in circles_pred: c.remove()
        circles_pred.clear()
        circles_pred.extend([
            ax2.add_patch(plt.Circle((x1_pred[i], y1_pred[i]), joint_radius, color='g', alpha=0.5)),
            ax2.add_patch(plt.Circle((x2_pred[i], y2_pred[i]), joint_radius, color='m', alpha=0.5))
        ])
        link1_pred.set_data([0, x1_pred[i]], [0, y1_pred[i]])
        link2_pred.set_data([x1_pred[i], x2_pred[i]], [y1_pred[i], y2_pred[i]])

        return link1_gt, link2_gt, link1_pred, link2_pred

    anim = FuncAnimation(fig, update, frames=n_steps, repeat=True, interval=80)

    plt.tight_layout()
    plt.show()

    if save_gif:
        save_animation(anim, gif_filename)

def visualize_joint_thetas(mpc_h5_path, pred_h5_path, run_i=0):
    """
    Plot joint angles over time for mpc vs Prediction
    """
    theta1_gt, theta2_gt, _, _, _, _ = get_data_from_h5(mpc_h5_path, run_i)
    
    with h5.File(pred_h5_path, 'r') as f:
        run_key = f"run_{run_i}"
        run_data = f[run_key]
        theta1_pred = run_data['theta1'][:]
        theta2_pred = run_data['theta2'][:]

    time = np.arange(len(theta1_gt))*0.1  
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, theta1_gt, label='MPC Theta1')
    plt.plot(time, theta1_pred, label='ANN Prediction Theta1')
    plt.legend()
    plt.title('Joint 1 Angle over Time')

    plt.subplot(2, 1, 2)
    plt.plot(time, theta2_gt, label='MPC Theta2')
    plt.plot(time, theta2_pred, label='ANN Prediction Theta2')
    plt.legend()
    plt.title('Joint 2 Angle over Time')

    plt.tight_layout()
    plt.show()

def visualize_control_inputs(mpc_h5_path, pred_h5_path, run_i=0):
    """
    Plot control inputs over time for mpc vs nn prediction
    """
    _, _, _, _, u1_gt, u2_gt = get_data_from_h5(mpc_h5_path, run_i)
    
    with h5.File(pred_h5_path, 'r') as f:
        run_key = f"run_{run_i}"
        run_data = f[run_key]
        u1_pred = run_data['u1'][:]
        u2_pred = run_data['u2'][:]

    time = np.arange(len(u1_gt))*0.1 
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, u1_gt, label='MPC Control Input 1')
    plt.plot(time, u1_pred, label='ANN Prediction Control Input 1')
    plt.legend()
    plt.title('Control Input 1 over Time')

    plt.subplot(2, 1, 2)
    plt.plot(time, u2_gt, label='MPC Control Input 2')
    plt.plot(time, u2_pred, label='ANN Prediction Control Input 2')
    plt.legend()
    plt.title('Control Input 2 over Time')

    plt.tight_layout()
    plt.show()  

if __name__ == "__main__":
    mpc_h5_path = "model/hidden_test_data/hidden_test_data.h5"
    pred_h5_path = "analysis/model_predictions/basicann2_model_predictions_data_317_01_100_2026-03-22_14-30-52.h5"
    visualize_h5_comparison(mpc_h5_path, pred_h5_path, run_i=0, save_gif=False, gif_filename="robot_comparison.gif")
    visualize_joint_thetas(mpc_h5_path, pred_h5_path, run_i=0)
    visualize_control_inputs(mpc_h5_path, pred_h5_path, run_i=0)