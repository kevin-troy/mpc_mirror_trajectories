import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utilities.dynamics_tools import linearize, euler_integrator
from utilities.plot_tools import plot_trajectory, plot_hist, plot_summary
import utilities.generate_waypoints as gen_wp
from linear_mpc import mpc_step
from mirror_dynamics import mirror_dynamics

# Simulation fidelity. Increasing this will increase compute time as well as require Q,P,R matrix tuning.
FIDELITY_FACTOR = 1

# Define system parameters
R_actuators = 0.5                   # Position of actuators relative to mirror center. Assumes circularly symmetric layout
dt = .1/FIDELITY_FACTOR             # Sim timestep
Ixx = 0.1                           # Moment of inertia. Assume same for both axes.
Iyy = 0.1

n = 4                               # Number of states
m = 4                               # Number of control inputs
N = 10*FIDELITY_FACTOR              # Number of lookahead steps. Can trim down trajectory later.
P = 1e6*np.eye(n)                   # Terminal state cost.

# State and control constraints
R = np.eye(4)
Q = 1000*np.eye(4)
rx = np.array([np.pi/4, np.pi/4, np.pi/10, np.pi/10])
ru = np.array([.1, .1, .1, .1])

# Tolerances
convergence_tol = 1e-2      # Tolerance determining if state reaches goal state
voilation_tol = 1e-2        # Tolerance of computing state constraint violations
constraint_margin_perc = 5  # N% tolerance for constraints to account for nonlinearity in true system.

# Maximum number of MPC calls per waypoint pair
max_iters_per_subtraj = 150*FIDELITY_FACTOR


"""
Trajectory Selection
"""
waypoints = gen_wp.L_trajectory(side_length=0.5)                        # L Trajectory
waypoints = gen_wp.box_trajectory(side_length=0.5)                      # Box trajectory
waypoints = gen_wp.semicircle_trajectory(R=0.5, n_waypoints=10)         # Semicircle Trajectory
waypoints = gen_wp.spiral_trajectory(R=.5, decay=0.1, n_circles=2)      # Spiral Trajectory

"""
Formulate problem dynamics
"""
discrete_dynamics = euler_integrator(mirror_dynamics, dt),
A, B = linearize(euler_integrator(mirror_dynamics,dt), waypoints[0], np.zeros((4,)))
A = np.squeeze(np.array(A))
B = np.squeeze(np.array(B))

N_subtrajs = len(waypoints)-1
s_hist = np.zeros((N_subtrajs*max_iters_per_subtraj+1, 4))
u_hist = np.zeros((N_subtrajs*max_iters_per_subtraj+1, 4))
s_subtrajs = np.zeros((N_subtrajs*max_iters_per_subtraj, N+1, 4))
u_subtrajs = np.zeros((N_subtrajs*max_iters_per_subtraj, N, 4))
s_hist[0] = waypoints[0]
u_hist[0] = np.zeros((4,))

# Solve the planning problem iteratively until the system converges to the waypoint.
k = 0
for wp in waypoints[1:]:
    for subtraj_iter in tqdm(range(max_iters_per_subtraj)):

        # Compute next control input (include 5% margin for constraints)
        u_hist[k+1], s_subtrajs[k], u_subtrajs[k] = mpc_step(s_hist[k], wp, A, B, P, Q, R, N, (100-constraint_margin_perc)/100*rx, (100-constraint_margin_perc)/100*ru)

        # Sim with nonlinear dynamics
        s_hist[k+1] = s_hist[k] + dt*mirror_dynamics(s_hist[k], u_hist[k+1])

        if np.any(np.abs(s_hist[k]-voilation_tol)>rx):
            raise RuntimeError('State constraint violation! Breaking')

        k += 1
        if np.linalg.norm(s_hist[k]-wp)<convergence_tol:
            break


"""
# Plot simulated trajectory
fig, ax = plot_trajectory(s_hist[:k], waypoints)

# Plot tip, tilt time histories
fig, ax = plot_hist(s_hist[:k, 0:2], rx[0:2])

# Plot tip, tilt rates time histories
fig, ax = plot_hist(s_hist[:k, 2:], rx[2:],
                    title="Angular Velocity Histories",
                    ylabel="Angular Velocity [rad/s]")

# Plot tip control history
fig, ax = plot_hist(u_hist[:k, 0:2], ru[0:2],
                    title="Tilt Control Histories",
                    series_labels=["F1", "F2"],
                    ylabel="Force [N]")
# Plot tilt control history
fig, ax = plot_hist(u_hist[:k, 0:2], ru[0:2],
                    title="Tilt Control Histories",
                    series_labels=["F1", "F2"],
                    ylabel="Force [N]")

"""

# Plot summary plots for single MPC step results
fig, axs = plot_summary(s_subtrajs[0], waypoints, u=u_subtrajs[0], rx=(100-constraint_margin_perc)/100*rx, ru=(100-constraint_margin_perc)/100*ru)


# Plot scenario summary plots for simulated rollout of control
fig, axs = plot_summary(s_hist[:k], waypoints, u=u_hist, rx=(100-constraint_margin_perc)/100*rx, ru=(100-constraint_margin_perc)/100*ru)
plt.show()
