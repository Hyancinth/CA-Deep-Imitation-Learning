import numpy as np

def link_points(theta, a = [1.0, 1.0], n_points = 4):
    """
    Generate intermediate points along robot links for better modelling and collision avoidance
    """
    theta1 = theta[0]
    theta2 = theta[1]
    a1 = a[0]
    a2 = a[1]

    points = []

    # points along link 1
    for i in range(1, n_points + 1):
        t = i/(n_points + 1) # parameter from 1/(n_points + 1) to n_points/(n_points + 1)
        x = t * a1 * np.cos(theta1)
        y = t * a1 * np.sin(theta1)
        points.append((x, y))
    
    # points along link 2
    for i in range(1, n_points + 1):
        t = i/(n_points + 1)
        x = a1 * np.cos(theta1) + t * a2 * np.cos(theta1 + theta2)
        y = a1 * np.sin(theta1) + t * a2 * np.sin(theta1 + theta2)
        points.append((x, y))

    return points

