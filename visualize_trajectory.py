#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# See LICENSE file for additional copyright and license details.


import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

import agent as ag
import scenes
import utils



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



def main(args):

    # local color configuration
    colors = utils.Config()
    colors.start      = '#add3ff'
    colors.target     = '#88e275'
    colors.trajectory = '#d0d0d0'


    # load the data
    filepath = "{}/{}".format(args.datadir, args.filename)
    cfg, Xs, Ys = utils.load_trajectory_hdf5(filepath)

    # setup figure
    fig = plt.figure()
    ax0 = plt.subplot2grid((1,1), (0,0))
    ax0.axis('square')
    ax0.autoscale(True)

    # render the agent's initial position
    agent_patch = agent_to_path(Xs[0], Ys[0], facecolor=colors.start)
    ax0.add_patch(agent_patch)

    # render the platform
    platform_patch = patches.Circle((cfg.platform_x, cfg.platform_y), cfg.platform_radius, edgecolor='black', linestyle='solid', facecolor=colors.target, lw=0.5)
    ax0.add_patch(platform_patch)

    # render trajectory from agent
    ax0.plot(Xs, Ys, color=colors.trajectory, lw=1.0)

    # get a patch for the scene to render nicely
    scene = scenes.Circular(cfg.world_radius)
    scenepatch = scene.getScenePatch(linestyle='solid', edgecolor='black', facecolor='none', lw=1.5)
    ax0.add_patch(scenepatch)

    # configure limits
    ax0.set_xlim([-cfg.world_radius - .1 , cfg.world_radius + .1])
    ax0.set_ylim([-cfg.world_radius - .1 , cfg.world_radius + .1])


    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='visualize_trajectory.py', description="Generate random watermaze trajectory")
    parser.add_argument('filename', type=str, help='Filename with trajectory data')
    parser.add_argument('--datadir', default='trajectory_data', type=str, help="Directory for output files")
    args = parser.parse_args()

    main(args)
