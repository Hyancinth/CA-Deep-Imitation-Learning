# CA-Deep-Imitation-Learning
DNN that learns to imitate an MPC for collision avoidance 

## Todo: 
- Stop the optimizer if it takes too long to find a solution (obstacle + target position isn't feasible)
    - This is complete by adding nlpsol_opts': {'ipopt.max_cpu_time': 0.05}
    - Need to be able to handly this case in the data generation stage and try a different combination
    - Combinations that causes an exit: target = np.array([1.0, 1.0]), obstacle = np.array([1.4, 1.0])
- Joint velocities get super jagged (bouncy) when it needs to move near the obstacle to reach the goal (probably due to repulsion + SSM)
    - Some examples:
        - target = np.array([1.5, 1.0]), obstacle = np.array([1.0, 0.8])
- This configuration: target = np.array([1.0, 1.3]), obstacle = np.array([0.8, 0.8]) is infeasible and is also very jagged in joint velocities
- Rewrite and clean up code
    - Code was moved out of simpleMPC to their own files. main.py now centrally runs everything and params.py was deleted
    - Can still be cleaned up more (especially with comments etc) 
- Create data generation script
    - Think about how the data should be saved

## Complete:
- Add more circles along the robot link and update cost/constraint function to prevent robot link collision with obstacle
- Implement ground constraint (prevent robot arm from hitting the ground at y = 0)
- Visualize and animate robot joint motion, joint angles, and joint velocities
- Calculate cartesian velocities for SSM constraint
- Calculate Jacobian for each joint
- Implement zeta parameter


