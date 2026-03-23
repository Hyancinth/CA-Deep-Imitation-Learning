import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter


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


def save_animation(anim, filename):
    """
    Saves animation to a GIF
    """
    gif_writer = ImageMagickWriter(fps=20)
    try:
        anim.save(filename, writer=gif_writer)
        print(f"Animation saved as {filename}")
    except Exception as e:
        print(f"Error saving animation: {e}")