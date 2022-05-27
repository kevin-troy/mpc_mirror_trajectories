import numpy as np
import cvxpy as cvx

"""
Basic implementation of Linear MPC
"""


def mpc_step(s0, sg, A, B, P, Q, R, N, rx, ru, minimum_control=1e-3):
    """
    Computes a single iteration of MPC

    :param s0: Initial State
    :param sg: Goal State
    :param A: Dynamics A matrix
    :param B: Dynamics B matrix
    :param P: Terminal state cost matrix
    :param Q: State deviation cost matrix
    :param R: Control cost matrix
    :param N: Lookahead horizon (steps)
    :param rx: Control constraints (evaluated as norms)
    :param ru: Control constraints (evaluated as norms)
    :param minimum_control: Minimum permissible control. All values <= this will be rounded to zero.
    :return u*: Optimal control for next timestep
    :return status: Optimization problem status
    :return s_cvx: State solution to optimization
    :return u_cvx: Control solution to optimization
    """

    n = Q.shape[0]
    m = R.shape[0]

    s_cvx = cvx.Variable((N+1, n))
    u_cvx = cvx.Variable((N, m))

    constraints = [s_cvx[0] == s0]                      # Initial state constraint
    objective = cvx.quad_form(s_cvx[-1]-sg, P)          # Terminal state cost

    for k in range(N):
        objective += cvx.quad_form(s_cvx[k]-sg, Q)      # State deviation cost
        objective += cvx.quad_form(u_cvx[k], R)         # Control cost

        # Dynamics constraint
        constraints += [s_cvx[k+1, :] == A@s_cvx[k, :] + B@u_cvx[k, :]]

        # State constraints
        constraints += [cvx.norm(s_cvx[k, idx], 2) <= val for idx, val in enumerate(rx)]

        # Control constraints
        constraints += [cvx.norm(u_cvx[k, idx], 2) <= val for idx, val in enumerate(ru)]

    problem = cvx.Problem(cvx.Minimize(objective), constraints)
    problem.solve()

    if problem.status != 'optimal':
        raise RuntimeError("MPC Solver Failed. Status = ", problem.status)

    # round off any small control inputs < minimum_control
    ustar = u_cvx.value[0]
    if np.linalg.norm(ustar) <= minimum_control:
        ustar = 0.0

    return ustar, s_cvx.value, u_cvx.value