#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# See LICENSE file for additional copyright and license details.

import os, datetime, sys, argparse, random
import numpy as np
import matplotlib.pyplot as plt

import modules
import pointgen
import utils
import plotter
import style

DEMO_NAME = "demo_05_watermaze"


def get_parent_objs(layers, scale, symbols, ss):
    # return set of all parents for all symbols simultaneously
    ps = []
    for s in ss:
        for p in s.parents:
            ps.append(symbols[p])

    # make elements unique
    ps = list(set(ps))
    return ps



def find_sequence_on_scale(layers, scale, symbols, start, target, args, recorder):
    """Find a sequence from start to target on a certain scale."""

    # initializiation
    # active_symbols = [start]

    # start really from the start here
    active_symbols = [start]
    targets = [target]

    tick = 0
    any_target_found = False
    recorder.record_expansion(scale, tick, start, targets, active_symbols, utils.intersect(targets, active_symbols))

    any_target_found = utils.intersect(targets, active_symbols) != []
    while not any_target_found:
        # update 'retrieval tick' -> this emulates refractory periods of place
        # cells. This comes from marking symbols as "expanded"
        for s in active_symbols:
            symbols[s].retrieval_tick = tick

        # predict next batch of symbols, and remove currently active ones
        next_symbols = layers[scale].expand(active_symbols, symbols, tick)
        active_symbols = [s for s in next_symbols if s not in active_symbols and symbols[s].retrieval_tick < 0]

        # update time
        tick += 1

        # record everything!!1
        recorder.record_expansion(scale, tick, start, targets, active_symbols, utils.intersect(targets, active_symbols))

        # check if we reached the destination
        any_target_found = utils.intersect(targets, active_symbols) != []
        if any_target_found:
            break

    return utils.intersect(targets, active_symbols)


def backtrack_on_scale(layers, scale, symbols, start, targets, args, recorder):
    """Backtrack on a certain scale from target to start.

    This involves region based computing and propagation of information

    start and target are indices"""

    # this is the 'global' start
    start_obj = symbols[start]

    # setup symbol and fetch all parents
    ss = [symbols[t] for t in targets]
    ps = get_parent_objs(layers, scale, symbols, ss)

    recorder.record_backtrack(scale, [utils.get_symbol_id(symbols, s) for s in ss],
                                     [utils.get_symbol_id(symbols, p) for p in ps])

    # this works, because we always start searching from the initial symbol. start is
    # thus the parent node of _every_ trajectory
    while not (start_obj in ps):
        ss = ps
        ps = get_parent_objs(layers, scale, symbols, ss)

        # record everything!!1
        recorder.record_backtrack(scale, [utils.get_symbol_id(symbols, s) for s in ss],
                                         [utils.get_symbol_id(symbols, p) for p in ps])

    return ss, ps



def spawn_symbol(cfg, coord, rnd_dist, nscales):
    dest = coord + utils.sample_unit_sphere(rnd_dist)

    # check not to spawn a symbol outside the environment
    while np.linalg.norm(dest) >= cfg.world_radius:
        dest = coord + utils.sample_unit_sphere(rnd_dist)

    return modules.Symbol(dest, nscales)


def generate_symbols_on_trajectory(cfg, Xs, Ys, mindist, nscales):
    # sample symbol location from unit sphere
    i = 0
    denom = 20.0

    coord = np.array([Xs[i], Ys[i]])
    symbols = [spawn_symbol(cfg, coord, mindist/denom, nscales)]

    for i in range(1, len(Xs)):
        coord = np.array([Xs[i], Ys[i]])
        _, dist = utils.get_closest_symbol(symbols, coord)
        if dist >= mindist:
            symbols.append(spawn_symbol(cfg, coord, mindist/denom, nscales))

    return symbols


def main(args) :
    # get a string for this simulation and generate the directory
    global DEMO_NAME

    # grid periods to use in this demo
    periods = [0.2]
    for i in range(1, args.nscales):
        periods.append(periods[-1] * np.sqrt(2))

    basename = os.path.basename(args.filename)
    dir_str = "{}/{}_{}.d".format(args.output_dir, DEMO_NAME, basename)
    if args.save_figures:
        print("Saving figures to directory {}".format(dir_str))
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

    # load HDF5
    print("Loading trajectory data from file '{}'".format(args.filename))
    cfg, Xs, Ys = utils.load_trajectory_hdf5(args.filename, verbose=False)

    # generate symbols, and select start and targets
    print("Generating symbols along trajectory.")
    symbols = generate_symbols_on_trajectory(cfg, Xs, Ys, args.mindist, args.nscales)

    global_start, _ = utils.get_closest_symbol(symbols, np.array([Xs[0], Ys[0]]))
    global_target, _ = utils.get_closest_symbol(symbols, np.array([Xs[-1], Ys[-1]]))


    # create transition layers and associate each with all available symbols
    print("Generating transition layers")
    z = 0
    layers = []
    for period in periods:
        print("Generating layer {} of {}".format(z+1, len(periods)))
        layers.append(modules.TransitionLayer(period, cfg.world_radius * 2.0, cfg.world_radius * 2.0, pointgen.hex))

        # this is required due to the circular world, centered at 0.0
        # shift all transitions
        for t in layers[-1].ts:
            t.coord -= cfg.world_radius
        # remove transitions outside of the environment
        ts = []
        for t in layers[-1].ts:
            dist = np.linalg.norm(t.coord)
            if dist <= (cfg.world_radius + 0.1):
                ts.append(t)
        layers[-1].ts = ts
        # compute neighborhood
        layers[-1].compute_neighborhoods()

        print("associating layer {}".format(z))
        layers[-1].associate(symbols, z)
        z += 1


    # plot that only shows the transition centers
    fig, axs = plotter.setup_watermaze(cfg, len(periods), ion=False)
    for scale in range(args.nscales):
        # exploit the symbol plotter here
        plotter.watermaze(axs[scale], cfg, Xs[0], Ys[0], plot_start_target=False)
        plotter.symbols(axs[scale], layers[scale].ts, range(len(layers[scale].ts)))
    if args.save_figures:
        fig.savefig("{}/transition_centers.svg".format(dir_str))
    plt.show()


    # result plotting
    fig, axs = plotter.setup_watermaze(cfg, len(periods), ion=False)

    for ax in axs:
        plotter.watermaze(ax, cfg, Xs[0], Ys[0])
        plotter.symbols(ax, symbols, [])
        plotter.trajectory(ax, Xs, Ys)
        plotter.start_target(ax, symbols, [global_start], [global_target])


    print("Running algorithm for each layer individually.")
    recorder = utils.Recorder(args.nscales)
    for scale in range(args.nscales):

        # reset the symbols for this scale
        for s in symbols:
            s.reset()

        #
        # Algorithm: expansion only, then backtracking during sample computation
        #
        hits = find_sequence_on_scale(layers, scale, symbols, global_start, global_target, args, recorder)
        print("Scale {}: Trajectory found.".format(scale))

        ts = []

        # Algorithm: compute M sample trajectories
        for i in range(args.M):
            # backtracking ala Dijkstra
            p = symbols[global_target]
            ts.append(layers[scale].ts[p.t[scale]])

            s = symbols[global_target]
            # TODO: is the following line required? I think not
            ts.append(layers[scale].ts[s.t[scale]])

            while not (s == symbols[global_start]):
                rnd_p = p.getRandomParent()

                p = symbols[rnd_p]

                ts.append(layers[scale].ts[p.t[scale]])
                plotter.sample_segment(axs[scale], s, p)

                s = p
                ts.append(layers[scale].ts[s.t[scale]])

        ts = list(set(ts))
        cols = utils.Config()
        cols.transition = 'gray'
        for t in ts:
            plotter.transition_domain(axs[scale], t, periods[scale])

        axs[scale].axis('off')

    if args.save_figures:
        fig.savefig("{}/results.svg".format(dir_str))
    plt.show()



if __name__ == "__main__":
    random.seed()
    np.random.seed()

    parser = argparse.ArgumentParser(prog='demo_05_watermaze.py', description="propagate through the transition system")
    parser.add_argument('filename'       , type=str            , help='Trajectory file , generated with "generate_trajectory.py"')

    parser.add_argument('--M'            , type=int            , default=10            , help='Number of Monte Carlo backtracks for trajectory generation')
    parser.add_argument('--save-figures' , dest='save_figures' , action='store_true'   , help='Save figures to file')
    parser.add_argument('--output-dir'   , type=str            , default='output'      , help='Output directory')

    parser.add_argument('--nscales'      , type=int            , default=5             , help='Number of scales')
    parser.add_argument('--mindist'      , type=float          , default=0.05          , help='Minimal distance between trajectory and symbol')
    parser.set_defaults(plot_transition_centers=False, save_figures=False, live_plot=True)

    args = parser.parse_args()
    main(args)


