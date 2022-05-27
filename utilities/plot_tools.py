import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(s, sg=None, border_size=.1, title="Angle-Space Trajectory"):
    # Plots the trajectory in the tip-tilt plane.

    fig, ax = plt.subplots()
    ax.plot(s[:, 0], s[:, 1], ls='dashdot', linewidth=2)
    ax.set_xlim([np.min(s[:, 0])-border_size, np.max(s[:, 0])+border_size])
    ax.set_ylim([np.min(s[:, 1])-border_size, np.max(s[:, 1])+border_size])

    if sg is not None:
        ax.scatter(sg[:, 0], sg[:, 1], marker='*', s=100, color="red", alpha=1)

    ax.set_title(title)
    ax.grid()
    ax.set_xlabel('Tip (theta) [rad]')
    ax.set_ylabel('Tilt (phi) [rad]')
    ax.set_aspect('equal')
    return fig, ax


def plot_hist(s, rx, sg=None,
              border_size=0.1,
              title="Tip/Tilt Histories",
              series_labels=["Tip", "Tilt"],
              ylabel="Angle [rad]"):
    # Plots the time histories of various states or controls.
    # Assumes identical constraints for tip and tilt axes states.

    k = len(s)

    fig, ax = plt.subplots()
    ax.plot(s[:, 0], label=series_labels[0])
    ax.plot(s[:, 1], label=series_labels[1])
    ax.set_title(title)
    ax.hlines(rx[0], 0, k, ls='--', color='black')
    ax.hlines(-rx[0], 0, k, ls='--', color='black')

    if sg is not None:
        ax.scatter(sg[:,0], sg[:,1], marker='*', s=100, color="red", alpha=1)

    ax.set_xlim([0, k])
    ax.set_ylim([-rx[0]-border_size, rx[0]+border_size])
    ax.grid()
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Simulation Timestep')
    ax.legend()

    return fig, ax


def plot_summary(s, sg, u=None, rx=None, ru=None):
    fig, axs = plt.subplots(1, 4)

    """
    Plot tip/tilt trajectory
    """
    border_size = 0.1
    axs[0].plot(s[:, 0], s[:, 1], ls='dashdot', linewidth=2)
    axs[0].set_xlim([np.min(s[:, 0])-border_size, np.max(s[:, 0])+border_size])
    axs[0].set_ylim([np.min(s[:, 1])-border_size, np.max(s[:, 1])+border_size])

    if sg is not None:
        axs[0].scatter(sg[:,0], sg[:,1], marker='*', s=100, color="red", alpha=1)

    axs[0].set_title("Angle-Space Trajectory")
    axs[0].grid()
    axs[0].set_xlabel('Tip (theta) [rad]')
    axs[0].set_ylabel('Tilt (phi) [rad]')
    axs[0].set_aspect('equal')

    """
    Plot tip/tilt time history
    """
    k = len(s)

    axs[1].plot(s[:, 0], label="Tip")
    axs[1].plot(s[:, 1], label="Tilt")
    axs[1].hlines(rx[0], 0, k, ls='--', color='black')
    axs[1].hlines(-rx[0], 0, k, ls='--', color='black')

    axs[1].set_xlim([0, k])
    axs[1].set_ylim([-rx[0]-border_size, rx[0]+border_size])
    axs[1].grid()
    axs[1].set_ylabel("Angle [rad]")
    axs[1].set_xlabel('Simulation Timestep')
    axs[1].set_title("Tip/Tilt Time Histories")
    axs[1].legend()

    """
    Plot tip/tilt rate time history
    """
    axs[2].plot(s[:, 2], label="Tip")
    axs[2].plot(s[:, 3], label="Tilt")
    axs[2].hlines(rx[2], 0, k, ls='--', color='black')
    axs[2].hlines(-rx[2], 0, k, ls='--', color='black')
    axs[2].set_xlim([0, k])
    axs[2].set_ylim([-rx[2]-border_size, rx[2]+border_size])
    axs[2].grid()
    axs[2].set_ylabel("Angular Velocity [rad/s]")
    axs[2].set_xlabel('Simulation Timestep')
    axs[2].set_title("Tip/Tilt Rate Time Histories")
    axs[2].legend()


    """
    Plot control time history
    """
    border_size = 0.05
    axs[3].plot(u[:, 0], label="F1")
    axs[3].plot(u[:, 1], label="F2")
    axs[3].plot(u[:, 2], label="F3")
    axs[3].plot(u[:, 3], label="F4")
    axs[3].hlines(ru[0], 0, k, ls='--', color='black')
    axs[3].hlines(-ru[0], 0, k, ls='--', color='black')
    axs[3].set_xlim([0, k])
    axs[3].set_ylim([-ru[0]-border_size, ru[0]+border_size])
    axs[3].grid()
    axs[3].set_ylabel("Force [N]")
    axs[3].set_xlabel('Simulation Timestep')
    axs[3].set_title("Control Time Histories")
    axs[3].legend()

    return fig, axs

