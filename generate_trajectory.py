#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# See LICENSE file for additional copyright and license details.


import os, sys, argparse
import numpy as np

import agent as ag
import scenes
import utils


def main(cfg):
    #  get a scene and an agent
    scene = scenes.Circular(cfg.world_radius)
    agent = ag.Agent2D(cfg.agent_x, cfg.agent_y)

    # platform
    P = np.array([cfg.platform_x, cfg.platform_y])

    # integration time step (in s). this is only required to retrieve good values
    # from the samplers that are used within the Agent class, as this class is also
    # used in other simulations where it is necessary to specify real units.
    dt   = cfg.dt

    # data from the agent
    Xs = []
    Ys = []

    # max ticks prevents the agent to roam infinitely
    t = 0
    while t < cfg.maxticks:
        agent.iter(dt, scene)
        Xs.append(agent.X[0])
        Ys.append(agent.X[1])

        # if the agent found the platform, then we're done
        dist_to_platform = np.linalg.norm(agent.X[0:2] - P)
        if dist_to_platform <= cfg.platform_radius:
            break
        t += 1

    if t < cfg.maxticks:
        print("Target platform found after {} ticks".format(t))
        utils.save_trajectory_to_hdf5(cfg, Xs, Ys)
    else:
        print("Target platform not found during maximum allowance time")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='generate_trajectory.py', description="Generate random watermaze trajectory")
    parser.add_argument('filename', type=str, help='Filename for output')
    parser.add_argument('--world_radius', default=1.0, type=float, help='Radius of the environment')
    parser.add_argument('--platform_x', default=0.5, type=float, help='X coordinate of the target platform')
    parser.add_argument('--platform_y', default=0.5, type=float, help='Y coordiante of the target platform')
    parser.add_argument('--platform_radius', default=0.1, type=float, help='Radius of the target platform')
    parser.add_argument('--agent_x', default=0.0, type=float, help='Initial X coordiante of the agent')
    parser.add_argument('--agent_y', default=-0.5, type=float, help='Initial Y coordinate of the agent')
    parser.add_argument('--datadir', default='trajectory_data', type=str, help="Directory for output files")
    parser.add_argument('--maxticks', default=10000, type=int, help="Maximum ticks to allow the agent to find the platform")
    parser.add_argument('--dt', default=0.1, type=float, help="Integration time width")
    args = parser.parse_args()
    main(args)
