import numpy as np
import casadi as ca

def fk(theta, a):
    """
    Calculate forward kinematics for a 2-link RR manipulator
    """
    a1 = a[0]
    a2 = a[1]

    theta1 = theta[0]
    theta2 = theta[1]

    x1 = a1 * ca.cos(theta1)
    y1 = a1 * ca.sin(theta1)

    x2 = x1 + a2*ca.cos(theta1 + theta2)
    y2 = y1 + a2*ca.sin(theta1 + theta2)

    return [x1, y1, x2, y2]

def jacobian(theta, a = np.array([1.0, 1.0])):
    """
    Calculate jacobian matrix for each joint of a 2-link RR manipulator
    """
    a1 = a[0]
    a2 = a[1]
    theta1 = theta[0]
    theta2 = theta[1]

    # joint 1 jacobian
    j1 = ca.vertcat(
        ca.horzcat(-a1*ca.sin(theta1), 0),
        ca.horzcat(a1*ca.cos(theta1), 0)
    )

    # joint 2 jacobian
    j2 = ca.vertcat(
        ca.horzcat(-a1*ca.sin(theta1) - a2*ca.sin(theta1 + theta2), -a2*ca.sin(theta1 + theta2)),
        ca.horzcat(a1*ca.cos(theta1) + a2*ca.cos(theta1 + theta2), a2*ca.cos(theta1 + theta2))
    )

    return j1, j2

def robot_motion(mpc, simulator):
    """
    Extract joint positions from MPC and simulator data, and calculate the (x,y) positions of each joint using forward kinematics
    """
    # extract data
    theta1_sim = simulator.data['_x', 'theta1']
    theta2_sim = simulator.data['_x', 'theta2']

    theta1_mpc = mpc.data['_x', 'theta1']
    theta2_mpc = mpc.data['_x', 'theta2']

    # store joint positions
    x1_sim = []
    y1_sim = []
    x2_sim = []
    y2_sim = []

    x1_mpc = []
    y1_mpc = []
    x2_mpc = []
    y2_mpc = []

    # loop through joint angles and calculate joint positions using forward kinematics
    for t1, t2 in zip(theta1_sim, theta2_sim): # simulation
        t1 = t1[0]
        t2 = t2[0]

        pos = fk([t1, t2], [1.0, 1.0])
        x1_sim.append(pos[0])
        y1_sim.append(pos[1])
        x2_sim.append(pos[2])
        y2_sim.append(pos[3])
    
    for t1, t2 in zip(theta1_mpc, theta2_mpc): # MPC
        t1 = t1[0]
        t2 = t2[0]

        pos = fk([t1, t2], [1.0, 1.0])
        x1_mpc.append(pos[0])
        y1_mpc.append(pos[1])
        x2_mpc.append(pos[2])
        y2_mpc.append(pos[3])

    return {'x1_sim': x1_sim, 'y1_sim': y1_sim, 'x2_sim': x2_sim, 'y2_sim': y2_sim,
            'x1_mpc': x1_mpc, 'y1_mpc': y1_mpc, 'x2_mpc': x2_mpc, 'y2_mpc': y2_mpc}

def point_in_workspace(x, y, a = [1.0, 1.0]):
    """
    Check if a point (x, y) is within the reachable workspace of the robot

    The workspace of a 2 Dof RR manipulator is an annulus defined by:
    - Outer radius: a1 + a2
    - Inner radius: |a1 - a2|

    A point is in the workspace if its distance from the origin (assuming robot base is at the origin) is between the inner and outer radius 
    """
    a1 = a[0]
    a2 = a[1]

    dist = np.sqrt(x**2 + y**2)

    return abs(a1 - a2) <= dist <= (a1 + a2)

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

    # distance from obstacle to line (vector) defining each link
    d1 = np.abs((p1_end[0] - p1_start[0])*obstacle[0] - (p1_end[1] - p1_start[1])*obstacle[1] + p1_end[0]*p1_start[1] - p1_end[1]*p1_start[0]) / np.linalg.norm(p1_end - p1_start)

    d2 = np.abs((p2_end[0] - p2_start[0])*obstacle[0] - (p2_end[1] - p2_start[1])*obstacle[1] + p2_end[0]*p2_start[1] - p2_end[1]*p2_start[0]) / np.linalg.norm(p2_end - p2_start)

    return [d1, d2]
