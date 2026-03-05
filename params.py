"""
File that holds global parameters for the MPC
"""
import numpy as np

## Robot parameters
# link lengths
a1 = 1.0
a2 = 1.0

## MPC parameters
# time step
Ts = 0.1

# target position
target = np.array([1.5, 0.5])