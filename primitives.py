#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# See LICENSE file for additional copyright and license details.

import numpy as np

# A minimalistic ray tracer (RT) computes if a simulated agent makes valid moves
# within a world. We operate already in 3D space for future developments,
# although the examples presented in here are on flat 2D surface.

# A ray consists of an origin point, and a directional vector (both 3D vectors)
class Ray:
    def __init__(self, origin, dir):
        self.origin = origin
        self.dir = dir

# A hit record indicates if something was hit by a ray. It has an ID (integer)
# and hit information (3D point where the hit occurred)
class HitRecord:
    def __init__(self, id, hit):
        self.id = id
        self.hit = hit

    def __repr__(self):
        return "HitRecord: {{id: {}, hit: {}}}".format(self.id, self.hit)


# Worlds can be specified by adding primitives. The agent always starts at 0/0
# within the world and is not allowed to cross through walls defined by
# primitives.

# A triangle has three points (in 3D space) by which it is specified
class Triangle:
    def __init__(self, A, B, C):
        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.C = np.asarray(C)

    def intersect(self, ray):
        # result: hit record
        hr = HitRecord(-1, np.array([np.inf, np.inf, np.inf]))

        # Möller-Trumbore ray triangle intersection
        e1 = self.B - self.A
        e2 = self.C - self.A

        # compute determinant
        P = np.cross(ray.dir, e2)
        det = e1.dot(P)
        # if det is near zero, ray is inside the triangle
        if det == 0.0:
            return np.inf, hr

        # ignore negative determinante. this only means that the triangle was
        # hit on the backface - which is OK for us here
        inv_det = 1 / det

        # calculate distance from A to ray origin
        T = ray.origin - self.A
        u = T.dot(P) * inv_det
        #intersection lies outside of the triangle?
        if u < 0.0 or u > 1.0:
            return np.inf, hr

        Q = np.cross(T, e1)
        v = ray.dir.dot(Q) * inv_det
        # intersection lies outside of triangle?
        if v < 0.0 or v > 1.0:
            return np.inf, hr

        # compute the distance.
        # Note that the distance needs to be positive, otherwise the triangle
        # would be hit walking backwards on the ray. however, as mentioned
        # above, we have ignored negative determinantes which means the
        # hit is still accepted even if the triangle is hit on the backside!

        dist = e2.dot(Q) * inv_det
        if dist > 0.0:
            hr.hit = ray.origin + dist * ray.dir
            return dist, hr

        # no hit
        return np.inf, hr



# A plane requires an origin point and a normal vector. Planes can be used
# quickly to create square worlds
class Plane:
    def __init__(self, origin, normal):
        self.origin = origin
        self.normal = normal


    def intersect(self, ray):
        # result: hit record
        hr = HitRecord(-1, np.array([np.inf, np.inf, np.inf]))

        dt = ray.dir.dot(self.normal)
        if dt == 0.0:
            return np.inf, hr
        else:
            dist = (self.origin - ray.origin).dot(self.normal) / dt
            hr.hit = ray.origin + dist * ray.dir
            return dist, hr



# A cylinder is given by its origin, upright direction vector, and radius
class Cylinder:
    def __init__(self, origin, dir, r):
        self.origin = origin
        self.dir = dir
        self.r = r


    def intersect(self, ray):
        # result: hit record
        hr = HitRecord(-1, np.array([np.inf, np.inf, np.inf]))

        dp = ray.origin - self.origin
        E = ray.dir - ray.dir.dot(self.dir) * self.dir
        D = dp - dp.dot(self.dir) * self.dir

        A = E.dot(E)
        B = 2.0 * (E.dot(D))
        C = D.dot(D) - self.r**2.0

        q = B*B - 4*A*C
        if q < 0.0:
            return np.inf, hr
        if A == 0.0:
            return np.inf, hr


        q = np.sqrt(q)
        t0 = (-B + q) / (2.0 * A)
        t1 = (-B - q) / (2.0 * A)

        if t0 < 0.0 and t1 < 0.0:
            return np.inf, hr

        if t0 < 0.0 and t1 >= 0.0:
            hr.hit = ray.origin + t1 * ray.dir
            return t1, hr

        if t1 < 0.0 and t0 >= 0.0:
            hr.hit = ray.origin + t0 * ray.dir
            return t0, hr

        if t1 > t0:
            hr.hit = ray.origin + t0 * ray.dir
            return t0, hr

        hr.hit = ray.origin + t1 * ray.dir;
        return t1, hr



# A wall between two points and a specific height
class Wall:
    def __init__(self, X0, X1, height=0.5):
        self.X0 = np.asarray(X0)
        self.X1 = np.asarray(X1)
        self.height = height

        # A wall is made of two triangles
        self.T0 = Triangle(X0, X1, np.array([X1[0], X1[1], height]))
        self.T1 = Triangle(X0, np.array([X1[0], X1[1], height]), np.array([X0[0], X0[1], height]))


    def intersect(self, ray):
        # result: hit record
        hr = HitRecord(-1, np.array([np.inf, np.inf, np.inf]))

        t0, hr0 = self.T0.intersect(ray)
        t1, hr1 = self.T1.intersect(ray)

        if np.isinf(t0) and np.isinf(t1):
            return np.inf, hr
        elif t0 < t1:
            hr.hit = hr0.hit
            return t0, hr
        else:
            hr.hit = hr1.hit
            return t1, hr


# A wall of a list of 3D points
class PolyWall:
    def __init__(self, Xs, height=0.5):
        self.Xs = Xs
        self.walls = []

        # convert all points to consecutive walls
        for i in range(len(Xs)):
            self.walls.append(Wall(Xs[i], Xs[(i+1) % len(Xs)], height))

    def intersect(self, ray):
        dist_min = np.inf
        dist = np.inf

        hr = HitRecord(-1, np.array([np.inf, np.inf, np.inf]))

        # test ray intersection with each wall
        for w in self.walls:
            dist, local_hr = w.intersect(ray)
            if dist < dist_min:
                dist_min = dist
                hr.hit = local_hr.hit

        return dist_min, hr



class AABB:
    """An Axis Aligned Bounding Box from X0 to X1"""
    def __init__(self, X0, X1, height=0.5):
        self.X0 = X0
        self.X1 = X1
        self.height = height

        # the AAbox is made of four walls
        self.north = Wall(np.array([X0[0], X1[1], 0.0]), X1, height)
        self.east = Wall(X1, np.array([X1[0], X0[1], 0.0]), height)
        self.south = Wall(np.array([X1[0], X0[1], 0.0]), X0, height)
        self.west = Wall(X0, np.array([X0[0], X1[1], 0.0]), height)


    def intersect(self, ray):
        dist_min = np.inf
        dist = np.inf

        hr = HitRecord(-1, np.array([np.inf, np.inf, np.inf]))

        # test ray intersection with each wall
        for w in [self.north, self.east, self.south, self.west]:
            dist, local_hr = w.intersect(ray)
            if dist < dist_min:
                dist_min = dist
                hr.hit = local_hr.hit

        return dist_min, hr

