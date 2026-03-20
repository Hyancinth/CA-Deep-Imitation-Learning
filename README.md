# CA-Deep-Imitation-Learning
DNN that learns to imitate an MPC for collision avoidance 

How to run: python -m folder_name.module_name

## Todo: 
- Create a new 100 run dataset that includes x and y distances from the target to ee
- Continue testing neural network
    - The 100 run dataset and exclusion of u1_prev and u2_prev helped improve the performance of the model 
    - Test changing the layer architecture to be: 128 -> 62 -> 32 -> 2
- Rewrite and clean up code
    - Can still be cleaned up more (especially with comments etc) 

## Notes:
- data_317_01_100.5 is the latest working 
- Joint velocities get super jagged (bouncy) when it needs to move near the obstacle to reach the goal (probably due to repulsion + SSM)
- Combinations that causes an exit: target = np.array([1.0, 1.0]), obstacle = np.array([1.4, 1.0])
- Combinations that cause jagged velocity: np.array([1.5, 1.0]), obstacle = np.array([1.0, 0.8]) 
- This configuration: target = np.array([1.0, 1.3]), obstacle = np.array([0.8, 0.8]) is infeasible and is also very jagged in joint velocities

## Complete:
- Training and testing NN pipeline + visualization
- Fixed data generation script
- Create another visualization method for animating robot motion from generated h5 file
- Create a new MPC version where the target and obstacle positions are now parameters within the MPC
- Data generation script
- Stop the optimizer if it takes too long to find a solution (obstacle + target position isn't feasible)
- Add more circles along the robot link and update cost/constraint function to prevent robot link collision with obstacle
- Implement ground constraint (prevent robot arm from hitting the ground at y = 0)
- Visualize and animate robot joint motion, joint angles, and joint velocities
- Calculate cartesian velocities for SSM constraint
- Calculate Jacobian for each joint
- Implement zeta parameter


