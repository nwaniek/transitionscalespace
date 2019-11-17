#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# See LICENSE file for additional copyright and license details.

import os, datetime, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import matplotlib.patches as patches
from matplotlib.path import Path

import modules
import pointgen
import utils
import plotter

DEMO_NAME = "demo_02_wavepropagation"

def main(args) :
    global DEMO_NAME

    # get a string for this simulation and generate the directory
    dir_str = "{}/{}_{}_{:4.2f}.d".format(args.output_dir, DEMO_NAME, args.pointgen, args.period)
    if args.save_figures:
        print("Saving figures to directory {}".format(dir_str))
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

    # generate symbols
    print("Generating symbols.")
    mindist = 0.02
    if args.pointgen == 'hammersley':
        symbols = utils.gen_symbols(args.W, args.H, args.N)
    else:
        symbols = utils.gen_symbols(args.W, args.H, N=args.N, method='rmind1', mindist=mindist)

    # create transition layer and associate it with all available symbols
    print("Creating layer.")
    layers = []
    layers.append(modules.TransitionLayer(args.period, args.W, args.H, pointgen.hex))
    layers[0].associate(symbols)

    # select some random symbols
    start, _  = utils.get_closest_symbol(symbols, np.array([args.startX, args.startY]))
    target, _ = utils.get_closest_symbol(symbols, np.array([args.targetX, args.targetY]))

    # initialize the symbols
    current_symbols = [start]

    # plotting setup
    fig, ax = plotter.setup(args, 1, figsize=(4,4))
    ax = ax[0]
    ax.axis('off')

    # pure retrieval to target
    target_found = False
    tick = 0
    print("Running algorithm.")
    while tick <= args.maxticks:
        ax.clear()
        ax.axis('off')

        # update 'retrieval tick' -> this emulates refractory periods of place
        # cells. This comes from marking symbols as "expanded"
        for s in current_symbols:
            symbols[s].retrieval_tick = tick

        # plotting
        plotter.everything(args, ax, symbols, current_symbols, start, target)
        if args.save_figures:
            fig.savefig("{}/{:03}.svg".format(dir_str, tick))


        # predict next batch of symbols, and remove currently active ones
        next_symbols = layers[0].expand(current_symbols, symbols, tick)
        current_symbols = [s for s in next_symbols if s not in current_symbols and symbols[s].retrieval_tick < 0]

        # update time
        tick += 1

        # check to see if we reached the destination
        target_found = target in current_symbols
        if target_found:
            break

    print("Done.")
    if not target_found:
        print("EE: Target not found")

    plotter.everything(args, ax, symbols, current_symbols, start, target)
    plt.draw()

    if target_found:
        print("Generating Monte Carlo samples")
        # get some monte carlo samples
        for i in range(args.M):
            # backtracking ala Dijkstra
            p = symbols[target]
            s = symbols[target]

            while not (s == symbols[start]):
                p = symbols[p.getRandomParent()]
                plotter.sample_segment(ax, s, p)
                s = p

        # plot the transition regions using a backtracking sample ala Dijkstra
        ts = []
        p = symbols[target]
        s = symbols[target]
        ts.append(layers[0].ts[p.t[0]])
        while not (s == symbols[start]):
            p = symbols[p.getRandomParent()]
            ts.append(layers[0].ts[p.t[0]])
            s = p

        for t in ts:
            plotter.transition_domain(ax, t, args.period)

    plt.ioff()
    if args.save_figures:
        fig.savefig("{}/{:03}.svg".format(dir_str, tick))
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='demo_02_wavepropagation.py', description="propagate through the transition system")
    parser.add_argument('--datadir', default='trajectory_data', type=str, help="Directory for output files")
    parser.add_argument('--W', type=float, default=1.0, help='Width of the environment')
    parser.add_argument('--H', type=float, default=1.0, help='Height of the environment')
    parser.add_argument('--period', type=float, default=0.07, help='Distance between transition centers')
    parser.add_argument('--startX', type=float, default=0.2, help='Select start symbol closest to this X')
    parser.add_argument('--startY', type=float, default=0.2, help='Select start symbol closest to this Y')
    parser.add_argument('--targetX', type=float, default=0.8, help='Select target symbol closest to this X')
    parser.add_argument('--targetY', type=float, default=0.8, help='Select target symbol closest to this Y')
    parser.add_argument('--N', type=int, default=500, help='Number of symbols to generate')
    parser.add_argument('--M', type=int, default=10, help='Number of Monte Carlo backtracks for trajectory generation')
    parser.add_argument('--pointgen', type=str, default='rand', help='Method for symbol placement. One of "hammersley", "rand"')
    parser.add_argument('--save-figures', dest='save_figures', action='store_true', help='Save figures to file')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--maxticks', type=int, default=100, help='Maximum ticks to limit algorithm execution')
    parser.add_argument('--plot-transition-centers', dest='plot_transition_centers', action='store_true', help='plot transition centers')
    parser.add_argument('--live-plot', dest='live_plot', action='store_true', help='Plot results while simulating')
    parser.add_argument('--plot-hexfields', dest='plot_hexfields', action='store_true')
    parser.set_defaults(plot_transition_centers=False, save_figures=False, live_plot=True, plot_hexfields=False)

    args = parser.parse_args()
    main(args)

