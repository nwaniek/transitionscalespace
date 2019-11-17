#!/usr/bin/env python

"""
This file simulates trajectory computation based on a previously learned
distribution of grid fields. The start and goal locations are the same as in
hemisphere_geodesic.py, but computed by propagating and backtracking (Dijkstra).

See hemisphere_walker.py for learning the distribution.
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


def xs2symbols(xs, ys, zs):
    coords = np.zeros((len(xs), 3))
    for i in range(len(xs)):
        coords[i, 0] = xs[i]
        coords[i, 1] = ys[i]
        coords[i, 2] = zs[i]

    symbols = []
    for i in range(len(xs)):
        symbols.append(modules.Symbol(coords[i, :]))

    return symbols



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

    # particle setup
    particles_json_path = 'data/particles.1M.010.json'
    Nsymbols = 5000
    mindist = 0.10
    # output setup
    trajectorysamples_json_path = 'data/trajectorysamples.1M.010.json'


    #particles_json_path = 'data/particles.1M.005.json'
    #Nsymbols = 10000
    #mindist = 0.05

    particles_args = {
            'dist_type'             : 'euclidean',
            'mindist'               : mindist,
            'maxdist'               : 2.0 * 0.1,
            'mindist_rnd_threshold' : 0.9,
            'mindist_rnd_chance'    : 0.1,
            'alpha'                 : 0.1,
            'alpha_decay'           : 0.9999,
            'mem'                   : 0.9,
        }
    particles = PushPullParticleSystem(**particles_args)
    # load_json will also load the particle state_dict
    particles.load_json(particles_json_path)

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

    print("Computing ground truth geodesic")
    for timestep in range(1000):
        # move around
        agent.move(random_walk=False)
        dist = np.linalg.norm(agent.X - Target)
        if dist <= agent.speed/2:
            # reached goal, or would overshoot
            break

    # get some random points on the hemisphere
    print("Picking symbols randomly")
    sym_xs, sym_ys, sym_zs = pick_npoints_on_hemisphere(N=Nsymbols)
    if False:
        ax.scatter(sym_xs, sym_ys, sym_zs, color='#7a7a7a', s=0.3)
    symbols = xs2symbols(sym_xs, sym_ys, sym_zs)

    # get indices of start and target symbols
    i_start_sym, _ = utils.get_closest_symbol(symbols, Start)
    i_target_sym, _ = utils.get_closest_symbol(symbols, Target)


    # create transition layer and associate it with all symbols
    print("Generating transition layers")
    layer = modules.TransitionLayer(particles_args['mindist'], -1.0, -1.0, pointgen_fn='particles', particles=particles)
    layer.associate(symbols, 0)

    # record everything!11
    recorder = utils.Recorder(1)

    # find sequence using the layer. Note that this is not guaranteed to work if
    # there is no viable trajectory...
    print("Searching trajectory...")
    hits = algorithms.find_sequence_on_scale([layer], 0, symbols, i_start_sym, i_target_sym, recorder)
    print("Trajectory found.")

    # Monte Carlo Sampling: compute M sample trajectories
    print("Monte Carlo Sampling")
    M = 50
    ps = []
    for i in range(M):
        ps.append([])
        p = symbols[i_target_sym]
        ps[i].append(p)
        while not (p == symbols[i_start_sym]):
            p = symbols[p.getRandomParent()]
            ps[i].append(p)

    # compute sample average
    avg_trace = []
    Ns = len(ps[0])
    for n in range(Ns):
        avg = ps[0][n].coord.copy()
        for m in range(1,M):
            avg += ps[m][n].coord
        avg /= np.linalg.norm(avg)
        avg_trace.append(avg)


    # draw calls
    clean_all(ax, vis_sphere, vis_agent, vis_coord, vis_trace)
    if vis_particles is not None:
        ax.collections.remove(vis_particles)

    # draw grid field centers
    vis_particles = draw_particles(ax, particles, color='gray', s=1)
    vis_sphere, _, _, _, _, _ = draw_all(ax, agent=agent, A=Start, B=Target, plot_trace=True, plot_agent=False)

    # draw monte carlo samples
    for k in range(M):
        for i in range(1, len(ps[k])):
            s = ps[k][i-1]
            t = ps[k][i]

            v = t.coord - s.coord
            draw_vector(ax, s.coord, v,
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

    # write all samples + symbol locations to file
    d = {
        'M_samples': M,
        'trajectory_length': len(ps[0]),
        'trajectory_data': [],
        'symbol_data': []
    }
    for m in range(M):
        td = []
        for n in range(len(ps[0])):
            td.append([float(ps[m][n].coord[0]), float(ps[m][n].coord[1]), float(ps[m][n].coord[2])])
        d['trajectory_data'].append(td)

    for i in range(len(sym_xs)):
        d['symbol_data'].append([float(sym_xs[i]), float(sym_ys[i]), float(sym_zs[i])])

    with open(trajectorysamples_json_path, 'w') as f:
        json.dump(d, f, indent=1)



    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


if __name__ == "__main__":
    main()
