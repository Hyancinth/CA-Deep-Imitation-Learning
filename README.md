# CA-Deep-Imitation-Learning
DNN that learns to imitate an MPC for collision avoidance 

## Todo: 
- Joint velocities get super jagged (bouncy) when it needs to move near the obstacle to reach the goal (probably due to repulsion + SSM)
- Rewrite and clean up code
    - Code was moved out of simpleMPC to their own files. main.py now centrally runs everything and params.py was deleted
    - Can still be cleaned up more (especially with comments etc) 
- Create a new MPC version where the target and obstacle positions are now parameters within the MPC
    - This way during the data generation process, the MPC doesn't need to be rebuilt each time, just need to change the parameters

## Notes:
- Combinations that causes an exit: target = np.array([1.0, 1.0]), obstacle = np.array([1.4, 1.0])
- Combinations that cause jagged velocity: np.array([1.5, 1.0]), obstacle = np.array([1.0, 0.8]) 
- This configuration: target = np.array([1.0, 1.3]), obstacle = np.array([0.8, 0.8]) is infeasible and is also very jagged in joint velocities

## Complete:
- Data generation script
- Stop the optimizer if it takes too long to find a solution (obstacle + target position isn't feasible)
- Add more circles along the robot link and update cost/constraint function to prevent robot link collision with obstacle
- Implement ground constraint (prevent robot arm from hitting the ground at y = 0)
- Visualize and animate robot joint motion, joint angles, and joint velocities
- Calculate cartesian velocities for SSM constraint
- Calculate Jacobian for each joint
- Implement zeta parameter


