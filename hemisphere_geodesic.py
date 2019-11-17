#!/usr/bin/env python

"""
This file implements an agent that walks on the geodesic between two points.

This is achieved by initializing the agent such that it is tangent to the
geodesic, and then simply forward integrating the path until the agent is at (or
close) to the target.
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mathutils import *
from plotutils import *
from agent3d import Agent3DSphere



def main():

    # agent setup
    Start = np.array(uv2xyz(-0.8 * np.pi, 0.4 * np.pi)).flatten()
    Target = np.array(uv2xyz(+np.pi/20, 0.4 * np.pi)).flatten()
    agent = Agent3DSphere(speed=0.01, A=Start, B=Target)

    # setup
    Tmax = 5000
    do_plot = True
    live_plot = False

    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        try:
            # does not work with all mplot3d versions
            ax.set_aspect('equal', 'box')
        except:
            pass
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        hide_axes(ax)

        vis_sphere = None
        vis_agent = None
        vis_coord = None
        vis_trace = None
        vis_A = None
        vis_B = None

        draw_hidden_aspect_cube(ax)

    for timestep in range(Tmax):

        # move around
        agent.move(random_walk=False)
        dist = np.linalg.norm(agent.X - Target)
        if dist <= agent.speed/2:
            # reached goal, numerically...
            break

        if do_plot and live_plot:
            # draw calls
            clean_all(ax, vis_sphere, vis_agent, vis_coord, vis_trace, vis_A, vis_B)
            vis_sphere, vis_agent, vis_coord, vis_trace, vis_A, vis_B = draw_all(ax, agent, Start, Target)
            ax.set_title(f"{timestep} / {Tmax}")
            plt.pause(0.001)
        else:
            print(f"{timestep} / {Tmax}")

    # draw calls
    if do_plot:
        clean_all(ax, vis_sphere, vis_agent, vis_coord, vis_trace, vis_A, vis_B)
        vis_sphere, vis_agent, vis_coord, vis_trace, vis_A, vis_B = draw_all(ax, agent, Start, Target)

        ax.set_title(f"{Tmax} / {Tmax}")
        plt.show()

if __name__ == "__main__":
    main()


