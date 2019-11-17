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

DEMO_NAME = "demo_06_temporal_compression"


def find_sequence(layers, periods, symbols, global_start, global_target, args, recorder):
    """Main Algorithm"""

    # number of scales available
    nscales = len(periods)

    # active symbols - everything on scale 0
    active_symbols = [global_start] # layers[0].ts[symbols[global_start].t[0]].domain
    targets = [global_target]

    i = 0
    tick = 0
    scale = 0
    any_target_found = utils.intersect(targets, active_symbols)
    while not any_target_found:
        for s in active_symbols:
            symbols[s].retrieval_tick = tick

        # fetch all symbols that follow from a given set of active symbols
        next_symbols = layers[scale].expand(active_symbols, symbols, tick)

        # the next active symbols are all those that are not already in the set
        # of active symbols, and for which the retrieval tick is not yet set
        active_symbols = [s for s in next_symbols if s not in active_symbols and symbols[s].retrieval_tick < 0]

        recorder.record_expansion(
                scale, tick,
                global_start, targets,
                active_symbols, utils.intersect(targets, active_symbols))

        # increase scale if we didn't find anything on this one
        if i >= args.max_i:
            scale += 1
            if scale >= nscales:
                scale = nscales - 1
            i = 0
        i += 1
        tick += 1

        # check if we reached the destination
        any_target_found = utils.intersect(targets, active_symbols) != []
        if any_target_found:
            break


def main(args) :
    # get a string for this simulation and generate the directory
    global DEMO_NAME

    # grid periods to use in this demo
    periods = [0.2]
    for i in range(1, args.nscales):
        periods.append(periods[-1] * np.sqrt(2))

    dir_str = "{}/{}_{}.d".format(args.output_dir, DEMO_NAME, args.pointgen)
    if args.save_figures:
        print("Saving figures to directory {}".format(dir_str))
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

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


    # plotting results
    fig, axs = plotter.setup(args, len(periods), ion=False)

    for ax in axs:
        plotter.rectangular_scene(ax, args.W, args.H)
        plotter.symbols(ax, symbols, range(len(symbols)))

        # global start and target
        plotter.start_target(ax, symbols, [global_start], [global_target])


    scale = nscales-1
    while scale >= 0:
        col_tmp = utils.Config()
        col_tmp.active = 'red'

        ts = []
        for e in recorder.expansions[scale]:
            plotter.symbols(axs[scale], symbols, e.active_symbols, only_active=False)
            for s in e.active_symbols:
                ts.append(layers[scale].ts[symbols[s].t[scale]])

        ts = list(set(ts))
        for t in ts:
            plotter.transition_domain(axs[scale], t, periods[scale])

        scale -= 1


    # find a few samples
    # MonteCarloSamples = [list() for n in range(nscales)]
    print("Monte Carlo sampling.")
    for i in range(args.M):
        # backtracking ala Dijkstra
        p = symbols[global_target]
        s = symbols[global_target]

        while not (s == symbols[global_start]):
            p = symbols[p.getRandomParent()]
            plotter.sample_segment(axs[-1], s, p)
            s = p

    plt.draw()
    if args.save_figures:
        fig.savefig("{}/results.svg".format(dir_str))
    plt.show()



if __name__ == "__main__":
    random.seed()
    np.random.seed()

    parser = argparse.ArgumentParser(prog='demo_04_scalespace_refinement_ascending.py', description="propagate through the transition system")
    parser.add_argument('--datadir'      , default='trajectory_data' , type=str            , help="Directory for output files")
    parser.add_argument('--W'            , type=float                , default=1.0         , help='Width of the environment')
    parser.add_argument('--H'            , type=float                , default=1.0         , help='Height of the environment')
    parser.add_argument('--startX'       , type=float                , default=0.1         , help='Select start symbol closest to this X')
    parser.add_argument('--startY'       , type=float                , default=0.2         , help='Select start symbol closest to this Y')
    parser.add_argument('--targetX'      , type=float                , default=0.8         , help='Select target symbol closest to this X')
    parser.add_argument('--targetY'      , type=float                , default=0.8         , help='Select target symbol closest to this Y')
    parser.add_argument('--N'            , type=int                  , default=500         , help='Number of symbols to generate')
    parser.add_argument('--M'            , type=int                  , default=10          , help='Number of Monte Carlo backtracks for trajectory generation')
    parser.add_argument('--pointgen'     , type=str                  , default='rand'      , help='Method for symbol placement. One of "hammersley", "rand"')
    parser.add_argument('--save-figures' , dest='save_figures'       , action='store_true' , help='Save figures to file')
    parser.add_argument('--output-dir'   , type=str                  , default='output'    , help='Output directory')
    parser.add_argument('--max-i'        , type=int                  , default=0           , help='Number of expansions per layer before ascent')
    parser.add_argument('--nscales'      , type=int                  , default=5           , help='Number of scales')
    parser.set_defaults(save_figures=False)

    args = parser.parse_args()
    main(args)


