#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# See LICENSE file for additional copyright and license details.

import numpy as np
from mathutils import *

# XXX: move distance functions to mathutils

def dist_euclidean(A, B):
    """Euclidean distance between A and B"""
    return np.linalg.norm(A - B)


def closest_euclidean(X, Pts):
    """Closest point in Pts to X.

    Args:
        X(np.array): coordiante of a query point
        Pts(np.array): matrix of coordinates of other points

    Returns:
        int: index into Pts
    """
    if Pts.size <= 0:
        return np.inf
    else:
        return np.min(np.linalg.norm(Pts - X, axis=1))


def hammersley(N, width, height, t=1):
    """Gernerate quasi-random 2D points according to Hammersley.

    Args:
        N(int): number of points to generate
        width(float): Specifies [0, width) for the x-coordinate
        height(float): Specifies [0, height) for the y-coordinate
        t(int): number of bits to truncate. Default=1

    Returns:
        np.array(shape=[N, 2]): matrix of coordinates
    """
    Pts = np.empty(shape=[N, 2])
    for k in range(N):
        u = 0
        p = 0.5
        kk = k
        while kk > 0:
            if (kk & 1):
                u = u + p
            p *= 0.5
            kk >>= t
        v = (k + 0.5) / N
        Pts[k,0] = u * width
        Pts[k,1] = v * height
    return Pts


def random_uniform(N, width, height):
    """Gernerate 2D points that are uniformly random distributed.

    Args:
        N(int): number of points to generate
        width(float): Specifies [0, width) for the x-coordinate
        height(float): Specifies [0, height) for the y-coordinate

    Returns:
        np.array(shape=[N, 2]): matrix of coordinates
    """
    X = np.random.uniform(0, width, N)
    Y = np.random.uniform(0, height, N)
    return np.vstack([X.ravel(), Y.ravel()]).T


def random_mindist(N, mindist, width, height):
    """Create random 2D points with a minimal distance to each other.

    Args:
        N(int): number of points to generate
        mindist(float): Minimal distance between each point
        width(float): Specifies [0, width) for the x-coordinate
        height(float): Specifies [0, height) for the y-coordinate

    Returns:
        np.array(shape=[N, 2]): matrix of coordinates
    """
    Pts = np.empty(shape=[0, 2])
    n = 0
    while n < N:
        X = random_uniform(1, width, height)
        # rejection sampling
        if closest_euclidean(X, Pts) > mindist:
            Pts = np.vstack((Pts, X))
            n = n+1
    return Pts


def hex(width, height, period=0.2, random_offset=False):
    """Generate points that are arranged on a hexagonal lattice.

    Args:
        width(float): width of a bounding box around the environment
        height(float): height of a bounding box around the environment
        period(float): inter-point distance
        random_offset(bool): add a random offset to the initial point that is
            usually centered at (0,0) (default=False)
    """
    Pts = []

    x_offset = 0
    y_offset = 0
    if random_offset:
        x_offset = np.random.uniform() * period/2.0
        y_offset = np.random.uniform() * period/2.0

    i = -1
    y = y_offset - period * np.sqrt(0.5)
    while y <= height + np.sqrt(0.5)*period:
        j = 1
        x = x_offset - (i % 2) * 0.5 * period
        while x < width + .5*period:
            Pts.append([x, y])
            x += period
        y += period/2.0 * np.sqrt(3)
        i += 1

    return np.asarray(Pts)


def flat(width, height, period=0.2, random_offset=False):
    Pts = []

    x_offset = 0
    y_offset = 0
    if random_offset:
        x_offset = np.random.uniform() * period/2.0
        y_offset = np.random.uniform() * period/2.0

    y = 0
    x = 0
    while x < width + .5 * period:
        Pts.append([x, y])
        x += period
    return np.asarray(Pts)


def square(width, height, period=0.2, random_offset=False):
    """Generate points that are arranged on a square lattice

    Args:
        width(float): width of a bounding box around the environment
        height(float): height of a bounding box around the environment
        period(float): inter-point distance
        random_offset(bool): add a random offset to the initial point that is
            usually centered at (0,0) (default=False)
    """
    Pts = []

    x_offset = 0
    y_offset = 0
    if random_offset:
        x_offset = np.random.uniform() * period/2.0
        y_offset = np.random.uniform() * period/2.0

    # start points slightly outside the bounding box to avoid some issues with
    # assigning symbols and transitions
    i = -1
    y = y_offset - 0.25 * period
    while y <= height + period:
        x = x_offset - 0.25 * period
        while x <= width + period:
            Pts.append([x, y])
            x += period
        y += period
        i += 1

    return np.asarray(Pts)


def pick_point_on_hemisphere():
    """Pick a random point on the upper part of a hemisphere."""
    xs, ys, zs = pick_random_hemisphere(N=1)
    return np.array([xs[0], ys[0], zs[0]])


def pick_npoints_on_hemisphere(N = 2500):
    """Randomly pick N points on the upper part of a hemisphere.

    see http://mathworld.wolfram.com/SpherePointPicking.html
    """
    xs, ys, zs = [], [], []
    us, vs = np.random.rand(N) * 2 * np.pi, np.arccos(np.random.rand(N))
    for i in range(N):
        x, y, z = uv2xyz(us[i], vs[i])
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return xs, ys, zs
