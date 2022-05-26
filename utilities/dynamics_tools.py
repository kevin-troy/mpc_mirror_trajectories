"""

A general set of tools for discretization and linearization of dynamical systems.

"""

import jax
import jax.numpy as jnp
import numpy as np


# Discretizes dynamics 'f' with a 4th order Runge Kutta integration schema.
def rk4_integrator(f: callable, dt: float):
    def integrator(s, u, dt=dt):
        k1 = dt * f(s, u)
        k2 = dt * f(s + k1 / 2, u)
        k3 = dt * f(s + k2 / 2, u)
        k4 = dt * f(s + k3, u)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return integrator


# Discretizes dynamics 'f' with a basic Euler integrator
def euler_integrator(f:callable, dt:float):
    def integrator(s,u):
        return s+dt*f(s,u)
    return integrator


# Linearizes dynamical system f at the provided states and controls
def linearize(f: callable,
              s: jnp.ndarray,
              u: jnp.ndarray):
    A = jax.jacfwd(f, argnums=(0,))(s,u)
    B = jax.jacfwd(f, argnums=(1,))(s,u)

    return A, B