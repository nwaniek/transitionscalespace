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

DEMO_NAME = "demo_03_scalespace_refinement_descending"

def get_parent_objs(layers, scale, symbols, ss):
    # return set of all parents for all symbols simultaneously
    ps = []
    for s in ss:
        for p in s.parents:
            ps.append(symbols[p])

    # make elements unique
    ps = list(set(ps))
    return ps


def find_sequence_on_scale(layers, scale, symbols, start, targets, args, recorder):
    """Find a sequence from start to target on a certain scale."""

    # initializiation
    # active_symbols = [start]

    # this is the horror...
    # XXX: change to [start] to prevent back-activation of symbols?
    active_symbols = layers[scale].ts[symbols[start].t[scale]].domain

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



def backtrack_on_scale(layers, periods, scale, symbols, start, targets, args, recorder):
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



def find_sequence(layers, periods, symbols, global_start, global_target, args, recorder):
    """Main Algorithm"""

    # find in largest scale. As a result, symbols will have timings and parents
    scale = len(layers)-1

    targets = [global_target]
    while scale >= 0:
        # reset all symbols (but not their transition association)
        for s in symbols:
            s.reset()

        hits = find_sequence_on_scale(layers, scale, symbols, global_start, targets, args, recorder)
        subgoals, _ = backtrack_on_scale(layers, periods, scale, symbols, global_start, hits, args, recorder)

        # short term memory update for next iteration, and pointer/object foo
        targets = [utils.get_symbol_id(symbols, s) for s in subgoals] # I wish I had PPP (Proper Pointers in Python)

        # record what happened
        recorder.record_search(scale, [global_start], targets, hits)
        recorder.record_subgoals(scale, targets)

        # drop down one scale
        scale -= 1



def main(args) :
    # get a string for this simulation and generate the directory
    global DEMO_NAME

    dir_str = "{}/{}_{}.d".format(args.output_dir, DEMO_NAME, args.pointgen)
    if args.save_figures:
        print("Saving figures to directory {}".format(dir_str))
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

    # grid periods to use in this demo
    periods = [0.2]
    for i in range(1, args.nscales):
        periods.append(periods[-1] * np.sqrt(2))

    # generate symbols
    print("Generating symbols.")
    nscales = len(periods)
    mindist = periods[0]/4
    if args.pointgen == 'hammersley':
        symbols = utils.gen_symbols(args.W, args.H, args.N, nscales=nscales)
    else:
        symbols = utils.gen_symbols(args.W, args.H, N=args.N, nscales=nscales, method='rmind1', mindist=mindist)

    # create transition layers and associate each with all available symbols
    print("Generating transition layers")
    z = 0
    layers = []
    for period in periods:
        print("Generating layer {} of {}".format(z+1, len(periods)))
        layers.append(modules.TransitionLayer(period, args.W, args.H, pointgen.hex))
        layers[-1].associate(symbols, len(layers)-1)
        z += 1

    # select desired start and target symbols
    global_start, _  = utils.get_closest_symbol(symbols, np.array([args.startX, args.startY]))
    global_target, _ = utils.get_closest_symbol(symbols, np.array([args.targetX, args.targetY]))


    ##
    ## Algorithm start #################################################################
    ##
    print("Running algorithm.")
    recorder = utils.Recorder(nscales)
    find_sequence(layers, periods, symbols, global_start, global_target, args, recorder)
    print("Done.")
    ##
    ##  Algorithm end ##################################################################
    ##


    # plot that only shows the transition centers
    fig, axs = plotter.setup(args, len(periods), ion=False)
    for scale in range(args.nscales):
        # exploit the symbol plotter here
        plotter.rectangular_scene(axs[scale], args.W, args.H)
        plotter.symbols(axs[scale], layers[scale].ts, range(len(layers[scale].ts)))
    plt.draw()
    if args.save_figures:
        fig.savefig("{}/transition_centers.svg".format(dir_str))
    plt.show()

    # plotting
    fig, axs = plotter.setup(args, len(periods), ion=False)

    for ax in axs:
        plotter.rectangular_scene(ax, args.W, args.H)
        plotter.symbols(ax, symbols, [])

        # global start and target
        plotter.start_target(ax, symbols, [global_start], [global_target])

    scale = nscales-1
    while scale >= 0:

        # select axes to plot to
        ax = axs[nscales - scale - 1]

        # subgoals
        plotter.subgoals(ax, symbols, recorder.searches[scale][-1].targets)

        ts = []
        for b in recorder.backtracks[scale]:
            # symbol's associated transition
            for s in b.ss:
                t = utils.get_transition_obj(layers, symbols, scale, s)
                ts.append(t)
        ts = list(set(ts))

        for t in ts:
            plotter.transition_domain(ax, t, periods[scale])

        # plot initial transition area
        ts = []
        b = recorder.backtracks[scale][-1]
        for s in b.ps:
            t = utils.get_transition_obj(layers, symbols, scale, s)
            ts.append(t)
        ts = list(set(ts))
        for t in ts:
            # tmp_colors = utils.Config()
            # tmp_colors.transition = 'green'
            plotter.transition_domain(ax, t, periods[scale])

        scale -= 1

    plt.draw()
    if args.save_figures:
        fig.savefig("{}/results.svg".format(dir_str))
    plt.show()



if __name__ == "__main__":
    random.seed()
    np.random.seed()

    parser = argparse.ArgumentParser(prog='demo_03_scalespace_refinement_descending.py', description="propagate through the transition system")
    parser.add_argument('--datadir', default='trajectory_data', type=str, help="Directory for output files")
    parser.add_argument('--W', type=float, default=1.0, help='Width of the environment')
    parser.add_argument('--H', type=float, default=1.0, help='Height of the environment')
    parser.add_argument('--startX', type=float, default=0.1, help='Select start symbol closest to this X')
    parser.add_argument('--startY', type=float, default=0.2, help='Select start symbol closest to this Y')
    parser.add_argument('--targetX', type=float, default=0.8, help='Select target symbol closest to this X')
    parser.add_argument('--targetY', type=float, default=0.8, help='Select target symbol closest to this Y')
    parser.add_argument('--N', type=int, default=500, help='Number of symbols to generate')
    parser.add_argument('--M', type=int, default=10, help='Number of Monte Carlo backtracks for trajectory generation')
    parser.add_argument('--pointgen', type=str, default='rand', help='Method for symbol placement. One of "hammersley", "rand"')
    parser.add_argument('--save-figures', dest='save_figures', action='store_true', help='Save figures to file')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--maxticks', type=int, default=100, help='Maximum ticks to limit algorithm execution')
    parser.add_argument('--nscales', type=int, default=5, help='Number of scales')
    parser.set_defaults(save_figures=False)

    args = parser.parse_args()
    main(args)


