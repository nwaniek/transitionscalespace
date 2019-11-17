#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# See LICENSE file for additional copyright and license details.

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import matplotlib.patches as patches

import modules
import pointgen
import utils


def simulate_linear_track(symbols, grid_period, W, H, output_dir, live_plot=False):
    """Simulate a linear track, on which the goal is to go from start to finish"""

    # color configuration
    colors = utils.Config()

    colors.symbol_hit = 'black'
    colors.symbol_hit_alpha = 0.6
    colors.symbol_miss = 'grey'
    colors.symbol_miss_alpha = 0.4

    colors.transition_hit = '#2882cd'
    colors.transition_hit_alpha = 0.6
    colors.transition_miss = 'grey'
    colors.transition_miss_alpha = 0.4

    # create transition layer and associate it with all available symbols
    layer = modules.TransitionLayer(grid_period, W, H, pointgen.flat)
    layer.associate(symbols)

    # select some random symbol and activate it as well as all others in the area
    current_symbols = layer.getClosestTransition(np.array((0,0))).domain

    # select the rightmost symbols as final target location
    target_symbol, _ = utils.get_closest_symbol(symbols, np.array([W, 0.0]))

    # visualization setup
    fig = plt.figure(figsize=(20, 3))
    gs = gspec.GridSpec(1,1)
    gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.025, hspace=0.05)
    ax = plt.subplot(gs[0])

    if live_plot:
        plt.ion()
        plt.show()

    # pure retrieval to target
    ticks = 0
    while not target_symbol in current_symbols:
        ax.clear()

        # update 'retrieval tick' -> this emulates refractory periods of place
        # cells, which this model predicts are significantly different than grid
        # cell refractory periods
        for s in current_symbols:
            symbols[s].retrieval_tick = ticks

        # plot all transitions for which the precondition is met
        ts = layer.getDefinedTransitions(current_symbols)
        layer.plot_with_highlight(ax, symbols, highlighted=ts, color=colors.transition_miss, color_highlight=colors.transition_hit, marker='o', markersize=10)

        # replot all symbols. currently active symbols will get a different color
        for i in range(len(symbols)):
            if i in current_symbols:
                ax.plot(symbols[i].coord[0], symbols[i].coord[1], '.', color=colors.symbol_hit, alpha=colors.symbol_hit_alpha)
            else:
                ax.plot(symbols[i].coord[0], symbols[i].coord[1], '.', color=colors.symbol_miss, alpha=colors.symbol_miss_alpha)

        # predict next batch of symbols, and remove currently active ones
        next_symbols = layer.expand(current_symbols, symbols, ticks)
        current_symbols = [s for s in next_symbols if s not in current_symbols and symbols[s].retrieval_tick < 0]

        ax.set_xlim([-0.15, W+.15])
        ax.set_ylim([-0.1, 0.1])
        ax.set_aspect('equal')

        # drawing
        plt.title("tick %i" % (ticks))
        plt.draw()
        if live_plot:
            plt.pause(0.1)
        fig.savefig("{}/{:03}.svg".format(output_dir, ticks))

        # update tick counter
        ticks += 1

    # draw final state
    ax.clear()
    ts = layer.getDefinedTransitions(current_symbols)
    layer.plot_with_highlight(ax, symbols, highlighted=ts, color=colors.transition_miss, color_highlight=colors.transition_hit, marker='o', markersize=10)


    # replot all symbols. currently active symbols will get a different color
    for i in range(len(symbols)):
        if i in current_symbols:
            ax.plot(symbols[i].coord[0], symbols[i].coord[1], '.', color=colors.symbol_hit, alpha=colors.symbol_hit_alpha)
        else:
            ax.plot(symbols[i].coord[0], symbols[i].coord[1], '.', color=colors.symbol_miss, alpha=colors.symbol_miss_alpha)

    ax.set_xlim([-0.15, W+.15])
    ax.set_ylim([-0.1, 0.1])
    ax.set_aspect('equal')
    plt.title("tick %i" % (ticks))
    plt.draw()
    if live_plot:
        plt.pause(0.1)
    fig.savefig("{}/{:03}.svg".format(output_dir, ticks))
    if live_plot:
        plt.ioff()
        plt.show()

    return ticks


if __name__ == "__main__":
    # name of this demo
    DEMO_NAME = "demo_01_linear_track"

    # width and height of the environment to consider
    W = 10.0
    H = 0.02

    # grid periods
    grid_periods = [0.2, np.sqrt(2) * 0.2, 0.4, np.sqrt(2) * 0.4, 0.8, np.sqrt(2) * 0.8, 1.6]

    # create symbols
    symbols = utils.gen_symbols(W, H)
    ticks = []

    # simulate each grid period
    for gp in grid_periods:
        # define output directory and create, if necessary
        dir_str = "output/{}_{:1.4f}.d".format(DEMO_NAME, gp)
        if not os.path.exists(dir_str):
            os.mkdir(dir_str)

        # run the simulation
        ticks.append(simulate_linear_track(symbols, gp, W, H, dir_str, live_plot=False))

        # 'reset' all symbols
        for s in symbols:
            s.reset()

    # print statistics for plotting
    print(grid_periods)
    print(ticks)

