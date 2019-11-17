#!/usr/bin/env python

"""
This file implements an agent that randomly walks on a hemisphere. The random
walk uses constant speed and randomly selects a new heading direction based on a
draw from a laplacian distribution.

While walking, the agent places push-pull particles that simulate the expected
behavior of on-center and off-surround grid fields of grid cells.
"""

import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mathutils import *
from plotutils import *
from agent3d import Agent3DSphere
from particlesystem import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-particles-json', type=str            , help='Load particles from json')
    parser.add_argument('--dont-save'          , action='store_true' , help='Do not save data at end')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # agent
    speed = 0.02
    agent = Agent3DSphere(speed=speed)

    # particle setup
    particles_json_path = 'data/particles.json'
    particles_args = {
            'dist_type'             : 'euclidean',
            #'mindist'               : 0.1,
            'mindist'               : 0.05,
            'maxdist'               : 2.0 * 0.1,
            'mindist_rnd_threshold' : 0.9,
            'mindist_rnd_chance'    : 0.1,
            'alpha'                 : 0.1,
            'alpha_decay'           : 0.9999,
            'mem'                   : 0.9,
        }
    particles = PushPullParticleSystem(**particles_args)
    # load particles from file?
    if args.load_particles_json is not None and args.load_particles_json != '':
        particles.load_json(args.load_particles_json)
    # update. might not be necessary after loading a file...
    particles.update(agent.X)

    #Tmax = 500
    #Tmax = 250000
    Tmax = 1000000
    live_plot = True


    # plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal', 'box')
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
    vis_particles = None
    draw_hidden_aspect_cube(ax)

    # simulation
    for timestep in range(Tmax):

        # move around
        agent.move()

        # update the symbols
        particles.update(agent.X)

        # plotting
        if live_plot and ((timestep % 1000) == 0):
            clean_all(ax, vis_sphere, vis_agent, vis_coord, vis_trace)
            if vis_particles is not None:
                ax.collections.remove(vis_particles)

            vis_sphere, vis_agent, vis_coord, vis_trace, _, _= draw_all(ax, agent, plot_trace=False)
            vis_particles = draw_particles(ax, particles, color='blue')
            ax.set_title(f"{timestep} / {Tmax}")

        if live_plot:
            plt.pause(0.001)

        if (timestep % 100) == 0:
            print(f"Timestep {timestep} / {Tmax}; Particles: {len(particles)}")


    # draw calls
    clean_all(ax, vis_sphere, vis_agent, vis_coord, vis_trace)
    if vis_particles is not None:
        ax.collections.remove(vis_particles)
    vis_sphere, vis_agent, vis_coord, vis_trace, _, _= draw_all(ax, agent, plot_trace=False)
    vis_particles = draw_particles(ax, particles, color='blue')

    ax.set_title(f"{timestep} / {Tmax}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # save all data
    if not args.dont_save:
        particles.save_json(particles_json_path)

    # final plot
    ax.set_title(f"{Tmax} / {Tmax}")
    plt.show()


if __name__ == "__main__":
    main()
