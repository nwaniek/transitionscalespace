#!/usr/bin/env python

import numpy as np
from mathutils import *

##
# Agent in 3D space on a sphere
##
class Agent3DSphere:

    def __init__(self, speed=0.05, A=None, B=None):
        """Initialize an agent that can walk on a 3D sphere.

        If A and B are not None, then the agent will tangential to the geodesic
        between the two points."""

        self.speed = speed

        # initialize the agent for unit position
        if A is None or B is None:
            # R = random_rotmat()
            R = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

            self.up    = np.array([0, 0, 1])
            self.right = np.array([1, 0, 0])
            self.fwd   = np.cross(self.up, self.right)

            self.up    = R.dot(self.up)
            self.right = R.dot(self.right)
            self.fwd   = R.dot(self.fwd)

            # this is only valid on spherical world
            self.X     = self.up.copy()

        else:
            # tangent to the geodesic
            self.X = A
            self.up = self.X
            self.right = -np.cross(A, B)
            self.fwd = np.cross(self.up, self.right)
            self.fwd /= np.linalg.norm(self.fwd)

        self.X_history = [self.X]

        # XXX theta not really used.
        self.theta = 0.0


    def normalize(self):
        self.up    /= np.linalg.norm(self.up)
        self.right /= np.linalg.norm(self.right)
        self.fwd   /= np.linalg.norm(self.fwd)


    @staticmethod
    def get_proposal(up, right, fwd):
        # get random rotation around up vector via Rodrigues
        theta = np.random.laplace(0.0, 0.4)

        while theta > np.pi:
            theta -= 2*np.pi
        while theta < -np.pi:
            theta += 2*np.pi

        # theta = max(-1.0, min(1, theta))
        R = rodrigues(up, theta)

        # rotate all vectors
        up    = R.dot(up)
        right = R.dot(right)

        # normalize all vectors
        up    = up / np.linalg.norm(up)
        right = right / np.linalg.norm(right)

        # compute new forward vector
        fwd = np.cross(up, right)
        fwd = fwd / np.linalg.norm(fwd)

        # return the proposal
        return theta, up, right, fwd


    def check_proposal(self, up, right, fwd):
        # in this scenario, we don't want to go below Z == 0
        X_new = self.X + self.speed * fwd
        return X_new[2] > 0.001


    def move(self, random_walk=True):
        """Move the agent forward and get a new heading direction"""

        newX = self.X + self.speed * self.fwd
        # new coordinate on unit sphere
        self.X = newX / np.linalg.norm(newX)
        self.up = self.X.copy()
        # store to history
        self.X_history.append(self.X)

        # get proposal for next movement direction
        if random_walk:
            theta, p_up, p_right, p_fwd = Agent3DSphere.get_proposal(self.up, self.right, self.fwd)
            ok = self.check_proposal(p_up, p_right, p_fwd)
            if ok:
                # we're good to go
                # self.up    = p_up
                self.fwd   = p_fwd
                # self.right = p_right
                # self.right = np.cross(self.fwd, self.up)
            else:
                # reflect to disallow movement out of arena
                self.fwd   = -p_fwd

            self.up    = p_up
            self.right = np.cross(self.fwd, self.up)
            self.right = self.right / np.linalg.norm(self.fwd)
            self.theta = theta

        else:
            # keep direction
            self.fwd = np.cross(self.up, self.right)
            self.fwd = self.fwd / np.linalg.norm(self.fwd)
