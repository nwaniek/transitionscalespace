#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# See LICENSE file for additional copyright and license details.

import numpy as np
from numpy import linalg as LA
from matplotlib.path import Path
import matplotlib.patches as patches
from primitives import Ray, HitRecord, AABB, Wall, Cylinder, PolyWall

# A scene describes a world in which an agent can move around. It can be built
# using primitives (objects).
class Scene:
    def __init__(self):
        self.objects = []

    # add objects to the scene
    def addObject(self, obj):
        self.objects.append(obj)

    # trace a ray through the scene. this is useful to check if a certain
    # movement would lie inside or outside the scene. for instance, if the hit
    # distance towards an object is less than the movement distance, the agent
    # would run into a wall.
    def trace(self, ray):
        # default return value
        hr = HitRecord(-1, np.array([np.inf, np.inf, np.inf]))

        min_dist = np.inf
        i = 0
        for o in self.objects:
            dist, local_hr = o.intersect(ray)
            if dist >= 0.0 and dist <= min_dist:
                min_dist = dist
                hr.id = i
                hr.hit = local_hr.hit
            i = i + 1

        return min_dist, hr


    # this method checks if a certain movement from A to B is valid or not
    # within the scene
    def isValidMove(self, A, B):
        # get the direction of the movement ray
        D = B - A
        dist = LA.norm(D)

        # trace a ray to get the hit record towards scene objects
        ray = Ray(A, D / dist)
        d, hr = self.trace(ray)

        # we can move if we don't hit a wall
        return dist < d


# Definition of a square scene. The scene itself is simple, therefore we change
# the implementation of the isValidMove and just check if the target location is
# within the box
class Square(Scene):
    def __init__(self):
        super(Square, self).__init__()

        # a square world consists of an AABB in which the agent moves
        self.aabox = AABB(np.array([-1.0, -1.0, 0.0]),
                      np.array([ 1.0,  1.0, 0.0]))

        self.addObject(self.aabox)

    def isInside(self, A):
        return not ((A[0] < self.aabox.X0[0]) or (A[0] > self.aabox.X1[0]) or
                    (A[1] < self.aabox.X0[1]) or (A[1] > self.aabox.X1[1]))


    def isValidMove(self, A, B):
        return self.isInside(B)


    def getScenePatch(self, **kwargs):
        verts = [
                (-1.0, -1.0),
                ( 1.0, -1.0),
                ( 1.0,  1.0),
                (-1.0,  1.0),
                ( 0.0,  0.0)]
        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.CLOSEPOLY]

        scenepath = Path(verts, codes)
        return patches.PathPatch(scenepath, **kwargs)



# The circular scene is also simple to test, therefore again the isValidMove
# function is overridden
class Circular(Scene):
    def __init__(self, r=1.0):
        super(Circular, self).__init__()

        self.r = r
        self.cyl = Cylinder(np.array([0.0, 0.0, 0.0]),
                            np.array([0.0, 0.0, 1.0]),
                            r)

        self.addObject(self.cyl)


    def isInside(self, A):
        return (A[0]**2 + A[1]**2) <= self.cyl.r**2


    def isValidMove(self, A, B):
        return self.isInside(B)


    def getScenePatch(self, **kwargs):
        return patches.Circle((0.0, 0.0), self.r, **kwargs)



class TMaze(Scene):
    def __init__(self):
        super(TMaze, self).__init__()

        # the polywall is constructed by all points of the T in
        # counter-clockwise direction
        self.polywall = PolyWall([
                    [-0.33, -1.00, 0.0],
                    [ 0.33, -1.00, 0.0],
                    [ 0.33,  0.33, 0.0],
                    [ 1.00,  0.33, 0.0],
                    [ 1.00,  1.00, 0.0],
                    [-1.00,  1.00, 0.0],
                    [-1.00,  0.33, 0.0],
                    [-0.33,  0.33, 0.0]
                    ])

        self.addObject(self.polywall)

    def getScenePatch(self, **kwargs):
        verts = [
                (-0.33, -1.00),
                ( 0.33, -1.00),
                ( 0.33,  0.33),
                ( 1.00,  0.33),
                ( 1.00,  1.00),
                (-1.00,  1.00),
                (-1.00,  0.33),
                (-0.33,  0.33),
                ( 0.00,  0.00)]
        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.CLOSEPOLY]

        scenepath = Path(verts, codes)
        return patches.PathPatch(scenepath, **kwargs)




class Triangular(Scene):
    def __init__(self):
        super(Triangular, self).__init__()


        self.addObject(
                PolyWall([
                    [-1.0,  0.0, 0.0],
                    [ 1.0, -1.0, 0.0],
                    [ 1.0,  1.0, 0.0]
                    ])
                )


    def getScenePatch(self, **kwargs):
        verts = [
                (-1.0,  0.0),
                ( 1.0, -1.0),
                ( 1.0,  1.0),
                ( 0.0,  0.0)]
        codes = [Path.MOVETO,
                 Path.LINETO,
                 Path.LINETO,
                 Path.CLOSEPOLY]

        scenepath = Path(verts, codes)
        return patches.PathPatch(scenepath, **kwargs)

