import numpy as np

def dist_obstacle_to_links(obstacle, theta, a):
    """
    Compute the distance from the obstacle to the line defining each link of the robot
    """
    theta1 = theta[0]
    theta2 = theta[1]
    a1 = a[0]
    a2 = a[1]

    # points along link 1
    p1_start = np.array([0, 0])
    p1_end = np.array([a1 * np.cos(theta1), a1 * np.sin(theta1)])

    # points along link 2
    p2_start = p1_end
    p2_end = np.array([a1 * np.cos(theta1) + a2 * np.cos(theta1 + theta2), 
                       a1 * np.sin(theta1) + a2 * np.sin(theta1 + theta2)])

    d1 = np.abs((p1_end[0] - p1_start[0])*obstacle[0] - (p1_end[1] - p1_start[1])*obstacle[1] + p1_end[0]*p1_start[1] - p1_end[1]*p1_start[0]) / np.linalg.norm(p1_end - p1_start)

    d2 = np.abs((p2_end[0] - p2_start[0])*obstacle[0] - (p2_end[1] - p2_start[1])*obstacle[1] + p2_end[0]*p2_start[1] - p2_end[1]*p2_start[0]) / np.linalg.norm(p2_end - p2_start)

    return min(d1, d2)