import numpy as np

"""
A handful of waypoint sets to be tested with the pointing simulator.
"""


def L_trajectory(side_length=0.5):
    return np.array([[0, 0, 0, 0],
                     [0, side_length, 0, 0],
                     [side_length, side_length, 0, 0]])


def box_trajectory(side_length=0.5):
    return np.array([[0, 0, 0, 0],
                     [0, side_length, 0, 0],
                     [side_length, side_length, 0, 0],
                     [side_length, 0, 0, 0],
                     [0, 0, 0, 0]])


def semicircle_trajectory(R=0.5, n_waypoints=10):
    waypoints = [np.array([R*np.cos(np.pi/n_waypoints*i), R*np.sin(np.pi/n_waypoints*i), 0, 0]) for i in range((n_waypoints+1))]
    return np.array(waypoints)


def spiral_trajectory(R = 0.2, n_per_circ=10, n_circles=1, decay=0.5):
    waypoints = [np.array([R/(decay*i+1)*np.cos(np.pi/n_per_circ*i), R/(decay*i+1)*np.sin(np.pi/n_per_circ*i), 0, 0]) for i in range((n_per_circ+1)*n_circles)]
    return np.array(waypoints)