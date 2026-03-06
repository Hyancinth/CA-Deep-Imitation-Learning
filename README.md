# CA-Deep-Imitation-Learning
DNN that learns to imitate an MPC for collision avoidance 

## Todo: 
- Stop the optimizer if it takes too long to find a solution (obstacle + target position isn't feasible)
    - The current code in utils.py might not be sufficient
- Rewrite and clean up code

## Complete:
- Add more circles along the robot link and update cost/constraint function to prevent robot link collision with obstacle
- Implement ground constraint (prevent robot arm from hitting the ground at y = 0)
- Visualize and animate robot joint motion, joint angles, and joint velocities
- Calculate cartesian velocities for SSM constraint
- Calculate Jacobian for each joint
- Implement zeta parameter


