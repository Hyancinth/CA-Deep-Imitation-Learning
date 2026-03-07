import numpy as np
import casadi as ca

from simpleMPC import simulate
from visualization import visualize, save_animation

def main():
    target = np.array([1.0, 1.3])
    obstacle = np.array([0.8, 0.8])

    init_theta = np.array([ca.pi/6, -ca.pi/6])
    a = np.array([1.0, 1.0])

    mpc, simulator = simulate(target, obstacle)
    anim = visualize(mpc, simulator, target, obstacle)

    # save_animation(anim, 'mpc_animation.gif')

if __name__ == "__main__":
    main()