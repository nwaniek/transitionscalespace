#!/usr/bin/env python

import numpy as np

##
# Math utilities
##


def uv2xyz(u, v, r=1.0):
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def random_rotmat_x():
    x = np.random.rand(1) * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x),  np.cos(x)]])
    return Rx

def random_rotmat_y():
    y = np.random.rand(1) * 2 * np.pi
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])

    return Ry

def random_rotmat_z(theta=None):
    if theta is None:
        theta = np.random.rand(1) * 2 * np.pi
    Rz = np.array([[np.cos(theta), np.sin(theta), 0],
                   [-np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    return Rz


def random_rotmat():
    Rx = random_rotmat_x()
    Ry = random_rotmat_y()
    Rz = random_rotmat_z()
    return Rz.dot(Ry.dot(Rx))


def rodrigues(u, theta):
    """Rodrigues formula to compute a rotation theta around unit vector u."""
    ux = u[0]
    uy = u[1]
    uz = u[2]
    W = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]])
    I = np.identity(3)
    R = I + np.sin(theta) * W + (1 - np.cos(theta)) * W**2
    return R


