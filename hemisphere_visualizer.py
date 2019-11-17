#!/usr/bin/env python

"""
This file visualies trajectories computed using dense packing of grid fields and
sampling on a hemisphere. It also computes the ground-truth geodesic between the
two points.

See hemisphere_walker.py for learning the distribution, and hemisphere_tss.py to
create samples.
"""

import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mathutils import *
from plotutils import *
from pointgen import *
from agent3d import Agent3DSphere
from particlesystem import *

import modules
import utils
import pointgen
import algorithms
import style




def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # start and target symbols
    Start = np.array(uv2xyz(-.75 * np.pi, 0.4 * np.pi)).flatten()
    Target = np.array(uv2xyz(+np.pi/20, 0.4 * np.pi)).flatten()

    # 'agent' for ground truth. simply used as integrator
    agent = Agent3DSphere(speed=0.01, A=Start, B=Target)

    # files
    particles_json_path = 'data/particles.1M.010.json'
    trajectorysamples_json_path = 'data/trajectorysamples.1M.010.json'

    particles = PushPullParticleSystem()
    particles.load_json(particles_json_path)

    # plot setup
    stride = 1
    fig = plt.figure(figsize=(13,13))

    ax = fig.add_subplot(1, 2, 1, projection='3d', azim=-60, elev=52)
    try:
        # does not work with all mplot3d versions
        ax.set_aspect('equal')
    except:
        pass
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    hide_axes(ax)
    draw_hidden_aspect_cube(ax)

    vis_sphere = None
    vis_agent = None
    vis_coord = None
    vis_trace = None
    vis_particles = None

    print("Computing ground truth geodesic")
    for timestep in range(1000):
        # move around
        agent.move(random_walk=False)
        dist = np.linalg.norm(agent.X - Target)
        if dist <= agent.speed/2:
            # reached goal, or would overshoot
            break

    # loading trajectory samples and symbols
    with open(trajectorysamples_json_path, 'r') as f:
        state_dict = json.load(f)

    M = state_dict['M_samples']
    ps = state_dict['trajectory_data']
    symbols = state_dict['symbol_data']

    # compute sample average
    avg_trace = []
    Ns = len(ps[0])
    for n in range(Ns):
        avg = np.asarray(ps[0][n])
        for m in range(1,M):
            avg += np.asarray(ps[m][n])
        avg /= np.linalg.norm(avg)
        avg_trace.append(avg)

    # draw calls
    clean_all(ax, vis_sphere, vis_agent, vis_coord, vis_trace)
    if vis_particles is not None:
        ax.collections.remove(vis_particles)

    # draw grid field centers
    vis_particles = draw_particles(ax, particles, color='gray', s=1)
    vis_sphere, _, _, _, _, _ = draw_all(ax, agent=agent, A=Start, B=Target, plot_trace=True, plot_agent=False, stride=stride)

    # draw monte carlo samples
    for k in range(M):
        for i in range(1, len(ps[k])):
            s = np.asarray(ps[k][i-1])
            t = np.asarray(ps[k][i])

            v = t - s
            draw_vector(ax, s, v,
                    color=style.color.sample,
                    lw=0.4,
                    scale=1.0,
                    alpha=0.5) # style.alpha.sample)

    # draw sample average
    xs = []
    ys = []
    zs = []
    for X in avg_trace:
        xs.append(X[0]*1.0001)
        ys.append(X[1]*1.0001)
        zs.append(X[2]*1.0001)
    ax.plot3D(xs, ys, zs=zs, color='black', linewidth=2.0, linestyle='--')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax._axis3don = False
    ax.set_title('3D View')


    space = np.linspace(0, np.pi, 100)
    nxs = np.cos(space)
    nys = np.sin(space)

    agent_xs = []
    agent_ys = []
    agent_zs = []
    for i in range(len(agent.X_history)):
        agent_xs.append(agent.X_history[i][0])
        agent_ys.append(agent.X_history[i][1])
        agent_zs.append(agent.X_history[i][2])


    # plot front view
    if True:
        ax = fig.add_subplot(3, 3, 3)
        ax.axis('equal')
        ax.plot(nxs, nys, color='black')
        ax.plot([-1, 1], [0, 0], color='black')
        ax.set_xlim([-1.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_axis_off()
        ax.set_title('Front View')

        # sample average
        ax.plot(xs, zs, color='black', linewidth=1.0, linestyle='--')

        # monte carlo samples
        for m in range(M):
            sample_xs = []
            sample_zs = []
            for i in range(len(ps[m])):
                sample_xs.append(ps[m][i][0])
                sample_zs.append(ps[m][i][2])
            ax.plot(sample_xs, sample_zs, color=style.color.sample, lw=style.lw.sample, alpha=style.alpha.sample)

        # ground truth
        ax.plot(agent_xs, agent_zs, color='black', lw=0.5)


    # plot side view
    if True:
        ax = fig.add_subplot(3, 3, 6)
        ax.axis('equal')
        ax.plot(nxs, nys, color='black')
        ax.plot([-1, 1], [0, 0], color='black')
        ax.set_xlim([-1.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_axis_off()
        ax.set_title('Side View')

        # sample average
        ax.plot(ys, zs, color='black', linewidth=1.0, linestyle='--')

        # monte carlo samples
        for m in range(M):
            sample_ys = []
            sample_zs = []
            for i in range(len(ps[m])):
                sample_ys.append(ps[m][i][1])
                sample_zs.append(ps[m][i][2])
            ax.plot(sample_ys, sample_zs, color=style.color.sample, lw=style.lw.sample, alpha=style.alpha.sample)

        # ground truth
        ax.plot(agent_ys, agent_zs, color='black', lw=0.5)


    # plot top view
    if True:
        ax = fig.add_subplot(3, 3, 9)
        ax.axis('equal')
        ax.plot(nxs, nys, color='black')
        ax.plot(nxs, -nys, color='black')
        ax.set_xlim([-1.02, 1.02])
        ax.set_ylim([-1.02, 1.02])
        ax.set_axis_off()
        ax.set_title('Top View')

        # sample average
        ax.plot(xs, ys, color='black', linewidth=1.0, linestyle='--')

        # monte carlo samples
        for m in range(M):
            sample_xs = []
            sample_ys = []
            for i in range(len(ps[m])):
                sample_xs.append(ps[m][i][0])
                sample_ys.append(ps[m][i][1])
            ax.plot(sample_xs, sample_ys, color=style.color.sample, lw=style.lw.sample, alpha=style.alpha.sample)

        # ground truth
        ax.plot(agent_xs, agent_ys, color='black', lw=0.5)




    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
