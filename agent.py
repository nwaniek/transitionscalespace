#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# See LICENSE file for additional copyright and license details.

import numpy as np

class Agent2D:
    """Implementation of an agent that walks through a two dimensional world."""

    def __init__(self, x=0.0, y=0.0):
        # initial pose information
        self.X = np.array([x, y, 0.0])
        self.theta = np.deg2rad(90)


    def sample_speed(self, dt):
        mean_speed = 25.0 # cm/s
        max_speed  = 100.0 # cm/s

        # shape and slope for the gamma distribution
        k = 2.0
        th = mean_speed / k

        s = np.random.gamma(k, th)
        while (s >= max_speed):
            s = np.random.gamma(k, th)

        return dt * s


    def sample_theta(self, dt, factor):
        # sample an angular velocity
        return self.theta + factor * np.random.laplace(0, 0.15)


    def iter(self, dt, scene):
        # the /100 compensates for the fact that sample_speed returns units in
        # based on CM, but the 'scene' is measured in meters
        speed = self.sample_speed(dt) / 100

        newX = np.array([0.0, 0.0, 0.0])

        # the 'r' helps to get higher rotational values out of sample_theta in
        # case the agent gets stuck at a wall
        r = 1.0
        while True:
            theta = self.sample_theta(dt, r)
            newX[0] = self.X[0] + speed * np.cos(theta)
            newX[1] = self.X[1] + speed * np.sin(theta)
            r += 0.1

            if scene.isValidMove(self.X, newX):
                break

        self.X = newX
        self.theta = theta



