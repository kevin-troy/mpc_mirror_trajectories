import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cvxpy as cvx
from utilities.dynamics_tools import rk4_integrator, linearize, euler_integrator
from tqdm import tqdm


FIDELITY_FACTOR = 1

# Define system parameters
R_actuators = 0.5    # Position of actuators relative to mirror center. Assumes circularly symmetric layout
dt = .1/FIDELITY_FACTOR             # Sim timestep
Ixx = 0.1            # Moment of inertia. Assume same for both axes.
Iyy = 0.1

n = 4
m = 4
N = 10*FIDELITY_FACTOR               # Number of lookahead steps. Can trim down trajectory later.
P = 1e6*np.eye(n)     # Terminal state cost.

# State and control constraints
R = np.eye(4)
#Q = np.array([[1000,0,0,0],[0,1000,0,0], [0,0,1,0], [0,0,0,1]])
Q = np.eye(4)
RX = np.array([np.pi/4, np.pi/4, np.pi/10, np.pi/10])
ru = np.array([.1, .1, .1, .1])


def mirror_dynamics(s,u):
    """
    computes the state derivative from input state and control for a single axis of a pointing mirror.

    state = [tip, tilt, tip_dot, tilt_dot, ...etc]
    control = [F1, F2, F3, F4] where F1, F2 correspond to actuators about the tip axis and F3, F4 correspond to
        actuators about the tilt axis.
    """

    tip, tilt, tip_d, tilt_d = s
    F1, F2, F3, F4 = u

    return jnp.array([tip_d,
                      tilt_d,
                      R_actuators*(F1-F2)*jnp.cos(tip)/Ixx,
                      R_actuators*(F3-F4)*jnp.cos(tilt)/Iyy])


"""
Box trajectory
"""
waypoints = np.array([[0,0,0,0], [0, .5, 0, 0], [.5, .5, 0, 0]])#, [.2, 0, 0, 0]])

"""
Spiral trajectory [stepped]
"""
n_per_circ = 10
n_circles = 3
#waypoints = [np.array([0.5/(i+1)*np.cos(np.pi/n_per_circ*i), 0.5/(i+1)*np.sin(np.pi/n_per_circ*i), 0, 0]) for i in range((n_per_circ+1)*n_circles)]

"""
Circle Trajectory (continuous)
"""
n_per_circ = 10
#waypoints = [np.array([0.2*np.cos(np.pi/n_per_circ*i), 0.2*np.sin(np.pi/n_per_circ*i), 0, 0]) for i in range((n_per_circ+1)*n_circles)]


def solve_subtrajectory(s0, u0, sg):

    s_cvx = cvx.Variable((N+1, n))
    u_cvx = cvx.Variable((N, m))

    constraints = [s_cvx[0] == s0]                  # Initial state constraint
    #constraints += [u_cvx[0] == u0]                 # Initial and final control constraints
    objective = cvx.quad_form(s_cvx[-1]-sg, P)

    for k in range(N):
        objective += cvx.quad_form(s_cvx[k]-sg, Q)  # State deviation cost
        objective += cvx.quad_form(u_cvx[k], R)                 # Control cost
        #objective += cvx.norm(s_cvx[k]-waypoints[1])*k

        # Dynamics constraint
        constraints += [s_cvx[k+1,:] == A@s_cvx[k,:] + B@u_cvx[k,:]]

        # State constraints
        constraints += [cvx.norm(s_cvx[k, idx],2) <= val for idx, val in enumerate(RX)]

        # Control constraints
        constraints += [cvx.norm(u_cvx[k, idx],2) <= val for idx, val in enumerate(ru)]

    problem = cvx.Problem(cvx.Minimize(objective), constraints)
    problem.solve()

    # round off any small control inputs <0.001
    ustar = u_cvx.value[0]
    if np.linalg.norm(ustar)<1e-3:
        ustar = 0.0


    return ustar, problem.status == 'optimal', s_cvx.value, u_cvx.value


# Formulate problem.
discrete_dynamics = euler_integrator(mirror_dynamics, dt),
A, B = linearize(euler_integrator(mirror_dynamics,dt), waypoints[0], np.zeros((4,)))
A = np.squeeze(np.array(A))
B = np.squeeze(np.array(B))
convergence_tol = 1e-2

max_iters_per_subtraj = 150
# Solve the planning problem iteratively until the system converges to the waypoint.

N_subtrajs = len(waypoints)-1
s_hist = np.zeros((N_subtrajs*max_iters_per_subtraj+1, 4))
u_hist = np.zeros((N_subtrajs*max_iters_per_subtraj+1, 4))
s_subtrajs = np.zeros((N_subtrajs*max_iters_per_subtraj, N+1, 4))
u_subtrajs = np.zeros((N_subtrajs*max_iters_per_subtraj, N, 4))
s_hist[0] = waypoints[0]
u_hist[0] = np.zeros((4,))

k = 0
for wp in waypoints[1:]:
    for subtraj_iter in tqdm(range(max_iters_per_subtraj)):
        # Compute next control input
        u_hist[k+1], is_optimal, s_subtrajs[k], u_subtrajs[k] = solve_subtrajectory(s_hist[k], u_hist[k], wp)

        #s_hist[k+1] = A@s_hist[k] + B@u_hist[k+1]                              # Sim with linear dynamics
        s_hist[k+1] = s_hist[k] + dt*mirror_dynamics(s_hist[k], u_hist[k+1])    # Sim with nonlinear dynamics

        #if not is_optimal:
            #print('Non-optimal trajectory! breaking...')

        #if np.any(np.abs(s_hist[k])>rx):
            #print('State constraint violation! Breaking')

        k += 1
        if np.linalg.norm(s_hist[k]-wp)<convergence_tol:
            #print('Waypoint Reached, nice! norm= ',np.linalg.norm(s_hist[k]))
            break


fig, ax = plt.subplots()
ax.plot(s_hist[:k,0], s_hist[:k,1], ls='dashdot', marker='o')
ax.set_xlim([-.75,.75])
ax.set_ylim([-.75,.65])
ax.set_title("Trajectory")
ax.set_aspect('equal')


fig2, ax2 = plt.subplots()
ax2.plot(s_hist[:k,2])
ax2.plot(s_hist[:k,3])
ax2.set_title("Angular Velocities")
ax2.hlines(RX[2], 0, k, ls='--', color='black')
ax2.hlines(-RX[2], 0, k, ls='--', color='black')
ax2.set_xlim([0,k])
ax2.set_ylim([-1.5*RX[2], 1.5*RX[2]])


fig3, ax3 = plt.subplots()
ax3.plot(u_hist[:k,0])
ax3.plot(u_hist[:k,1])
ax3.set_title("Tip Controls")
ax3.set_ylim([-1.5*ru[0], 1.5*ru[0]])
ax3.hlines(ru[0], 0, k, ls='--', color='black')
ax3.hlines(-ru[0], 0, k, ls='--', color='black')
ax3.set_xlim([0,k])

fig3, ax3 = plt.subplots()
ax3.plot(u_hist[:k,2])
ax3.plot(u_hist[:k,3])
ax3.set_title("Tilt Controls")
ax3.set_ylim([-1.5*ru[2], 1.5*ru[2]])
ax3.hlines(ru[0], 0, k, ls='--', color='black')
ax3.hlines(-ru[0], 0, k, ls='--', color='black')


fig3, ax3 = plt.subplots()
ax3.plot(s_hist[:k,0])
ax3.plot(s_hist[:k,1])
ax3.set_title("tip/tilt vs. time")
plt.show()