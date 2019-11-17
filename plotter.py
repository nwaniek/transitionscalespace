#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# See LICENSE file for additional copyright and license details.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import matplotlib.patches as patches
from matplotlib.path import Path

import scenes
import utils
import style

# visualization setup
def setup(args, nscales, ion=False, figsize=(19,9)):

    fig = plt.figure(figsize=figsize)
    gs = gspec.GridSpec(1, nscales)
    gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)


    ax = [None for i in range(nscales)]
    for i in range(nscales):
        ax[i] = plt.subplot(gs[i])
        ax[i].set_xlim([-0.01, args.W+.01])
        ax[i].set_ylim([-0.01, args.H+.01])
        ax[i].set_aspect('equal')
        ax[i].axis('off')

    if ion:
        plt.ion()
        plt.show()

    return fig, ax


def setup_watermaze(cfg, nscales, ion=False):
    W = cfg.world_radius
    H = cfg.world_radius

    fig = plt.figure(figsize=(19,7))
    gs = gspec.GridSpec(1, nscales)
    gs.update(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01)

    ax = [None for i in range(nscales)]
    for i in range(nscales):
        ax[i] = plt.subplot(gs[i])
        ax[i].set_xlim([-W-.05, W+.05])
        ax[i].set_ylim([-H-.05, H+.05])
        ax[i].set_aspect('equal')
        ax[i].axis('off')

    if ion:
        plt.ion()
        plt.show()

    return fig, ax



# plot all symbols, colorize the currently active ones
def symbols(ax, symbols, current_symbols, only_active=False):
    # replot all symbols. currently active symbols will get a different color
    for i in range(len(symbols)):
        if i in current_symbols:
            ax.plot(symbols[i].coord[0], symbols[i].coord[1], '.', lw=style.lw.active, color=style.color.active, alpha=style.alpha.active, zorder=style.zorder.symbol)
        else:
            if not only_active:
                ax.plot(symbols[i].coord[0], symbols[i].coord[1], '.', lw=style.lw.inactive, color=style.color.inactive, alpha=style.alpha.inactive, zorder=style.zorder.symbol)


def rectangular_scene(ax, W, H):
    patch = patches.Rectangle((0, 0), W, H, color=None, facecolor="None", edgecolor=style.color.scene, lw=style.lw.scene, zorder=style.zorder.scene)
    ax.add_patch(patch)


def agent_to_path(x, y, **kwargs):
    verts = [(x-0.05, y-0.05),
             (x+0.05, y-0.05),
             (x+0.05, y+0.05),
             (x-0.05, y+0.05),
             ( 0.0,  0.0)]
    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY]

    path = Path(verts, codes)
    return patches.PathPatch(path, **kwargs)


def watermaze(ax, cfg, startX, startY, plot_start_target=True):
    if plot_start_target:
        # render the starting location
        agent_patch = agent_to_path(startX, startY, facecolor=style.color.start, zorder=style.zorder.watermaze_start)
        ax.add_patch(agent_patch)

        # render the target platform
        platform_patch = patches.Circle((cfg.platform_x, cfg.platform_y), cfg.platform_radius, edgecolor='black', linestyle='solid', facecolor=style.color.target, lw=0.5, zorder=style.zorder.watermaze_target)
        ax.add_patch(platform_patch)

    # get a patch for the scene to render nicely
    scene = scenes.Circular(cfg.world_radius)
    scenepatch = scene.getScenePatch(linestyle='solid', edgecolor='black', facecolor='none', lw=style.lw.scene, zorder=style.zorder.scene)
    ax.add_patch(scenepatch)



def start_target(ax, symbols, starts, targets, subgoals=False):
    for start in starts:
        spatch = patches.Circle(symbols[start].coord,  radius=0.03, edgecolor='black', facecolor=style.color.start,  alpha=style.alpha.start, zorder=style.zorder.start_symbol)
        sitem = patches.Circle(symbols[start].coord,  radius=0.01, edgecolor='black', facecolor='black', alpha=style.alpha.start, zorder=style.zorder.start_symbol+1)
        ax.add_patch(spatch)
        ax.add_patch(sitem)

    for target in targets:
        tpatch = patches.Circle(symbols[target].coord, radius=0.03, edgecolor='black', facecolor=style.color.target, alpha=style.alpha.target, zorder=style.zorder.target_symbol)
        titem = patches.Circle(symbols[target].coord, radius=0.01, edgecolor='black', facecolor='black', alpha=style.alpha.target, zorder=style.zorder.target_symbol+1)
        ax.add_patch(tpatch)
        ax.add_patch(titem)


def subgoals(ax, symbols, ss):
    for s in ss:
        circ = patches.Circle(symbols[s].coord, radius = 0.03, edgecolor='black', facecolor=style.color.subgoal, alpha=style.alpha.subgoal, zorder=style.zorder.subgoal)
        ax.add_patch(circ)


def everything(args, ax, syms, current_symbols, start, target):
    rectangular_scene(ax, args.W, args.H)
    start_target(ax, syms, [start], [target])
    symbols(ax, syms, current_symbols)
    plt.draw()
    plt.pause(1.0)


def transition_domain(ax, t, period):
    hex = patches.RegularPolygon(t.coord, 6, radius=np.sqrt(4/3) * period/2, facecolor=style.color.domain, alpha=style.alpha.domain, zorder=style.zorder.domain)
    ax.add_patch(hex)


def sample_segment(ax, s0, s1):
    ax.plot([s0.coord[0], s1.coord[0]], [s0.coord[1], s1.coord[1]], color=style.color.sample, lw=style.lw.sample, alpha=style.alpha.sample, zorder=style.zorder.sample)


def trajectory(ax, Xs, Ys):
    # trajectory data
    ax.plot(Xs, Ys, color=style.color.trajectory, lw=style.lw.trajectory)
