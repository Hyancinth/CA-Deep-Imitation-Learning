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

def robot_motion_from_data(theta1_arr, theta2_arr, a = [1.0, 1.0]):
    """
    Calculate the (x,y) positions of each joint using forward kinematics for given arrays of joint angles
    """
    x1_arr = []
    y1_arr = []
    x2_arr = []
    y2_arr = []

    for t1, t2 in zip(theta1_arr, theta2_arr):
        pos = fk([t1, t2], a)
        x1_arr.append(pos[0])
        y1_arr.append(pos[1])
        x2_arr.append(pos[2])
        y2_arr.append(pos[3])

    return {'x1': x1_arr, 'y1': y1_arr, 'x2': x2_arr, 'y2': y2_arr}

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

def dist_point_to_segment(point, seg_start, seg_end):
    """Distance from a point to a line segment, compatible with both numpy and CasADi"""
    point = np.array(point).flatten()
    seg_start = np.array(seg_start).flatten()
    seg_end = np.array(seg_end).flatten()
    
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    
    t = np.dot(point - seg_start, seg_vec) / seg_len_sq
    t = np.clip(t, 0, 1)
    
    closest = seg_start + t * seg_vec
    return np.linalg.norm(point - closest)


def dist_obstacle_to_links(obstacle, theta, a):
    """
    Compute the distance from the obstacle to each link segment of the robot.
    """
    theta1 = theta[0]
    theta2 = theta[1]
    a1 = a[0]
    a2 = a[1]

    # Link 1: base → joint
    p1_start = np.array([0.0, 0.0])
    p1_end   = np.array([a1 * np.cos(theta1),
                          a1 * np.sin(theta1)])

    # Link 2: joint → end-effector
    p2_start = p1_end
    p2_end   = np.array([a1 * np.cos(theta1) + a2 * np.cos(theta1 + theta2),
                          a1 * np.sin(theta1) + a2 * np.sin(theta1 + theta2)])

    d1 = dist_point_to_segment(obstacle, p1_start, p1_end)
    d2 = dist_point_to_segment(obstacle, p2_start, p2_end)

    return [d1, d2]


if __name__ == "__main__":
    target = np.array([0.36387244, 1.63541161])
    obstacle = np.array([1.68484258, 0.38647358])

    print(dist_obstacle_to_links(obstacle, [ca.pi/6, -ca.pi/6], [1.0, 1.0]))