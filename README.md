# CA-Deep-Imitation-Learning
DNN that learns to imitate an MPC for collision avoidance 

How to run: python -m folder_name.module_name

## Todo: 
- Create functions that loads saved model + scaler and then loads hidden datasets for testing (use run_model)
- Continue testing neural network
    - Increase the number of training epochs to 600
    - Using data_322_01_100 can do tests to see if excluding certain columns has better impact (for example we found that using ee_dx/dy_target causes the model to not 
    follow the ground constraint)
- Rewrite and clean up code
    - Can still be cleaned up more (especially with comments etc) 

## Notes:
- Basic Ann 2
    - Architecture was changed to be: input -> 265 -> 128 -> 64 -> output (2) 
    - At 500 epochs, trained on data_317_01_100, the performance is decent though it does still stop short of the goal. Perhaps increase the number of epochs to 600.  
- List of data + hidden_test_data that contains u1_prev, u2_prev, ee_dx_target, ee_dy_target
    - data_322_01_100.h5
    - data_320_01_100.h5
    - hidden_test_data_6.h5
    - hidden_test_data_5.h5
    - hidden_test_data_4.h5
    - hidden_test_data_3.h5
    - hidden_test_data_2.h5
- _5 and _6 are both more interesting test scenarios
- List of data + hidden_test_data that contains u1_prev, u2_prev
    - data_317_01_100.h5
    - data_313_01_30.h5
    - hidden_test_data_1.h5
    - hidden_test_data.h5
- data_317_01_100.h5 and hidden_test_data(_1).h5 are files that do not have ee_dx_target/ee_dy_target
    - The best results is with these files but removing u1_prev and u2_prev. The model generally follows the mpc trajectory, but doesn't reach the goal (200 epochs)
    - Even at 500 epochs, sometimes the model also violates the ground constraint
- data_320_01_100.h5 and hidden_test_data_2.h5 are files that do have ee_dx_target/ee_dy_target
    - Has the lowest loss but the problem is the model violates the constraint where the robot cannot touch the ground (with or without removing u1_prev, u2_prev)
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


