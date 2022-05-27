import jax.numpy as jnp

"""
Simple continuous-time nonlinear dynamics for a pointing mirror
"""

def mirror_dynamics(s, u, R=0.5, Ixx=0.1, Iyy=0.1):
    """
    Computes the state derivative from input state and control for a single axis of a pointing mirror.

    state = [tip, tilt, tip_dot, tilt_dot, ...etc]
    control = [F1, F2, F3, F4] where F1, F2 correspond to actuators about the tip axis and F3, F4 correspond to
        actuators about the tilt axis.
    """

    tip, tilt, tip_d, tilt_d = s
    F1, F2, F3, F4 = u

    return jnp.array([tip_d,
                      tilt_d,
                      R*(F1-F2)*jnp.cos(tip)/Ixx,
                      R*(F3-F4)*jnp.cos(tilt)/Iyy])
