import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils.modelling import link_points
from utils.utils import robot_motion_from_data, fk

from matplotlib.animation import FuncAnimation, ImageMagickWriter
from matplotlib.animation import PillowWriter

from visualization.utils import get_data_from_h5, save_animation


def visualize_h5_comparison(
    mpc_h5_path,
    pred_h5_path,
    run_i=0,
    obs_radius=0.1,
    joint_radius=0.1,
    save_gif=False,
    gif_filename="robot_comparison.gif",
    save_snapshot_path=None,
    model_label="RNN",
):
    """
    Animate side-by-side comparison of ground truth (hidden H5) vs predicted (NN H5).

    Args:
        mpc_h5_path (str): path to ground truth hidden dataset
        pred_h5_path (str): path to predicted rollout H5 file
        run_i (int): run index to visualize
        obs_radius (float): obstacle radius
        joint_radius (float): radius for joint circles
        save_gif (bool): whether to save the animation as GIF using PillowWriter
        gif_filename (str): filename for saved GIF
        save_snapshot_path (str | None): if provided, save a static snapshot of the
            final frame as a PNG at this path (useful for reports)
        model_label (str): label shown in the subplot title, e.g. "RNN" or "ANN"
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
        target_pred   = [float(run_data['target_x'][0]),   float(run_data['target_y'][0])]
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
        ax.add_patch(plt.Circle((obs[0],  obs[1]),  obs_radius, color='r', alpha=0.5, label='Obstacle'))
        ax.add_patch(plt.Circle((targ[0], targ[1]), 0.05,       color='b', alpha=0.5, label='Target'))

    ax1.set_title("MPC (Ground Truth)")
    ax2.set_title(f"{model_label} Prediction")

    link1_gt,   = ax1.plot([], [], 'o-', lw=4, color='#1f77b4')
    link2_gt,   = ax1.plot([], [], 'o-', lw=4, color='#ff7f0e')
    link1_pred, = ax2.plot([], [], 'o-', lw=4, color='#1f77b4')
    link2_pred, = ax2.plot([], [], 'o-', lw=4, color='#ff7f0e')

    circles_gt   = []
    circles_pred = []

    def update(i):
        for c in circles_gt:
            c.remove()
        circles_gt.clear()
        circles_gt.extend([
            ax1.add_patch(plt.Circle((x1_gt[i], y1_gt[i]), joint_radius, color='g', alpha=0.5)),
            ax1.add_patch(plt.Circle((x2_gt[i], y2_gt[i]), joint_radius, color='m', alpha=0.5)),
        ])
        link1_gt.set_data([0, x1_gt[i]], [0, y1_gt[i]])
        link2_gt.set_data([x1_gt[i], x2_gt[i]], [y1_gt[i], y2_gt[i]])

        for c in circles_pred:
            c.remove()
        circles_pred.clear()
        circles_pred.extend([
            ax2.add_patch(plt.Circle((x1_pred[i], y1_pred[i]), joint_radius, color='g', alpha=0.5)),
            ax2.add_patch(plt.Circle((x2_pred[i], y2_pred[i]), joint_radius, color='m', alpha=0.5)),
        ])
        link1_pred.set_data([0, x1_pred[i]], [0, y1_pred[i]])
        link2_pred.set_data([x1_pred[i], x2_pred[i]], [y1_pred[i], y2_pred[i]])

        return link1_gt, link2_gt, link1_pred, link2_pred

    anim = FuncAnimation(fig, update, frames=n_steps, repeat=True, interval=80)

    plt.tight_layout()

    if save_snapshot_path:
        # Draw the final frame and save as a static PNG
        update(n_steps - 1)
        plt.savefig(save_snapshot_path, bbox_inches="tight", dpi=150)
        print(f"Snapshot saved → {save_snapshot_path}")

    plt.show()

    if save_gif:
        try:
            writer = PillowWriter(fps=15)
            anim.save(gif_filename, writer=writer)
            print(f"GIF saved → {gif_filename}")
        except Exception as e:
            print(f"GIF save failed ({e}). Falling back to ImageMagickWriter.")
            save_animation(anim, gif_filename)


def visualize_joint_thetas(mpc_h5_path, pred_h5_path, run_i=0, save_path=None, model_label="RNN"):
    """
    Plot joint angles (theta1, theta2) over time for MPC vs model prediction.

    Args:
        mpc_h5_path (str): path to ground truth hidden dataset
        pred_h5_path (str): path to predicted rollout H5 file
        run_i (int): run index to visualize
        save_path (str | None): if provided, save the figure to this path
        model_label (str): label used for the prediction curves
    """
    theta1_gt, theta2_gt, _, _, _, _ = get_data_from_h5(mpc_h5_path, run_i)

    with h5.File(pred_h5_path, 'r') as f:
        run_key = f"run_{run_i}"
        run_data = f[run_key]
        theta1_pred = run_data['theta1'][:]
        theta2_pred = run_data['theta2'][:]

    time = np.arange(len(theta1_gt)) * 0.1

    fig, (ax_t1, ax_t2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax_t1.plot(time, theta1_gt,   label='MPC Theta1')
    ax_t1.plot(time, theta1_pred, label=f'{model_label} Theta1', linestyle='--')
    ax_t1.set_ylabel('Theta1 (rad)')
    ax_t1.set_title('Joint 1 Angle over Time')
    ax_t1.legend()
    ax_t1.grid(True)

    ax_t2.plot(time, theta2_gt,   label='MPC Theta2')
    ax_t2.plot(time, theta2_pred, label=f'{model_label} Theta2', linestyle='--')
    ax_t2.set_xlabel('Time (s)')
    ax_t2.set_ylabel('Theta2 (rad)')
    ax_t2.set_title('Joint 2 Angle over Time')
    ax_t2.legend()
    ax_t2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Joint angles plot saved → {save_path}")

    plt.show()


def visualize_control_inputs(mpc_h5_path, pred_h5_path, run_i=0, save_path=None, model_label="RNN"):
    """
    Plot control inputs (u1, u2) over time for MPC vs model prediction.

    Args:
        mpc_h5_path (str): path to ground truth hidden dataset
        pred_h5_path (str): path to predicted rollout H5 file
        run_i (int): run index to visualize
        save_path (str | None): if provided, save the figure to this path
        model_label (str): label used for the prediction curves
    """
    _, _, _, _, u1_gt, u2_gt = get_data_from_h5(mpc_h5_path, run_i)

    with h5.File(pred_h5_path, 'r') as f:
        run_key = f"run_{run_i}"
        run_data = f[run_key]
        u1_pred = run_data['u1'][:]
        u2_pred = run_data['u2'][:]

    time = np.arange(len(u1_gt)) * 0.1

    fig, (ax_u1, ax_u2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax_u1.plot(time, u1_gt,   label='MPC u1 (Joint 1 Velocity)')
    ax_u1.plot(time, u1_pred, label=f'{model_label} u1', linestyle='--')
    ax_u1.set_ylabel('u1 (rad/s)')
    ax_u1.set_title('Control Input 1 (Joint 1 Velocity) over Time')
    ax_u1.legend()
    ax_u1.grid(True)

    ax_u2.plot(time, u2_gt,   label='MPC u2 (Joint 2 Velocity)')
    ax_u2.plot(time, u2_pred, label=f'{model_label} u2', linestyle='--')
    ax_u2.set_xlabel('Time (s)')
    ax_u2.set_ylabel('u2 (rad/s)')
    ax_u2.set_title('Control Input 2 (Joint 2 Velocity) over Time')
    ax_u2.legend()
    ax_u2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Control inputs plot saved → {save_path}")

    plt.show()


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Example: visualise RNN predictions vs MPC ground truth for hidden test
    # data 2, 5, and 6.  Update the pred_h5_path entries to point to the
    # H5 files produced by scripts/run_rnn_hidden_test.py.
    # -----------------------------------------------------------------------
    from pathlib import Path
    import os

    repo_root  = Path(__file__).resolve().parents[1]
    imgs_dir   = repo_root / "imgs" / "rnn_comparison"
    os.makedirs(imgs_dir, exist_ok=True)

    SCENARIOS = [
        {
            "label":        "hidden_test_2",
            "mpc_h5":       str(repo_root / "model" / "hidden_test_data" / "hidden_test_data_2.h5"),
            "pred_h5":      str(repo_root / "analysis" / "model_predictions" / "rnn_predictions_data_322_01_100_hidden_test_data_2_run0_20260325_190952_excl_all_features.h5"),
        },
        {
            "label":        "hidden_test_5",
            "mpc_h5":       str(repo_root / "model" / "hidden_test_data" / "hidden_test_data_5.h5"),
            "pred_h5":      str(repo_root / "analysis" / "model_predictions" / "rnn_predictions_data_322_01_100_hidden_test_data_5_run0_20260325_190953_excl_all_features.h5"),
        },
        {
            "label":        "hidden_test_6",
            "mpc_h5":       str(repo_root / "model" / "hidden_test_data" / "hidden_test_data_6.h5"),
            "pred_h5":      str(repo_root / "analysis" / "model_predictions" / "rnn_predictions_data_322_01_100_hidden_test_data_6_run0_20260325_190953_excl_all_features.h5"),
        },
    ]

    for sc in SCENARIOS:
        label    = sc["label"]
        mpc_h5   = sc["mpc_h5"]
        pred_h5  = sc["pred_h5"]

        # ---- robot animation snapshot ----
        visualize_h5_comparison(
            mpc_h5_path=mpc_h5,
            pred_h5_path=pred_h5,
            run_i=0,
            save_gif=True,
            gif_filename=str(imgs_dir / f"rnn_data322_{label}_run0_robot_animation.gif"),
            save_snapshot_path=str(imgs_dir / f"rnn_data322_{label}_run0_robot_snapshot.png"),
            model_label="RNN",
        )

        # ---- joint angles plot ----
        visualize_joint_thetas(
            mpc_h5_path=mpc_h5,
            pred_h5_path=pred_h5,
            run_i=0,
            save_path=str(imgs_dir / f"rnn_data322_{label}_run0_joint_angles.png"),
            model_label="RNN",
        )

        # ---- control inputs plot ----
        visualize_control_inputs(
            mpc_h5_path=mpc_h5,
            pred_h5_path=pred_h5,
            run_i=0,
            save_path=str(imgs_dir / f"rnn_data322_{label}_run0_control_inputs.png"),
            model_label="RNN",
        )
