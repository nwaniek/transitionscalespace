#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mathutils import uv2xyz


def get_hemisphere_mesh():
    u = np.linspace(-np.pi,  np.pi, 50)
    v = np.linspace(0, .5 * np.pi, 50)
    x, y, z = uv2xyz(u, v)
    return x, y, z

def draw_hemisphere(ax, stride=5):
    x, y, z = get_hemisphere_mesh()
    #return ax.plot_surface(x, y, z, rstride=stride, cstride=stride, color='#d8d8d8', linewidth=0, alpha=0.5, edgecolor='none')
    return ax.plot_surface(x, y, z, rstride=stride, cstride=stride, color='white', linewidth=0, alpha=0.3, edgecolor='none')



def draw_hidden_aspect_cube(ax, max=1.0):
    for direction in (-1, 1):
        for point in np.diag(direction * max * np.array([1,1,1])):
            ax.plot([point[0]], [point[1]], [point[2]], 'w')

def hide_axes(ax):
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def draw_point(ax, X, color='black'):
    return ax.scatter(X[0], X[1], X[2], color=color)


def draw_particles(ax, particles, color='gray', s=10):
    xs = []
    ys = []
    zs = []

    for i in range(len(particles)):
        X = particles[i].X
        xs.append(X[0])
        ys.append(X[1])
        zs.append(X[2])

    return ax.scatter(xs, ys, zs, color=color, s=s)


def draw_vector(ax, X, v, color, lw, scale=0.1, **kwargs):
    xs = np.array([X[0], X[0] + v[0] * scale])
    ys = np.array([X[1], X[1] + v[1] * scale])
    zs = np.array([X[2], X[2] + v[2] * scale])
    xs = xs.flatten()
    ys = ys.flatten()
    zs = zs.flatten()
    return ax.plot3D(xs, ys, zs=zs, color=color, linewidth=lw, **kwargs)


def draw_agent_coord(ax, agent):
    c1 = draw_vector(ax, agent.X, agent.up, 'blue', 0.5)
    c2 = draw_vector(ax, agent.X, agent.right, 'green', 0.5)
    c3 = draw_vector(ax, agent.X, agent.fwd, 'red', 1.0)
    return (c1, c2, c3)


def draw_trace(ax, agent, color='black', lw=0.2):
    if len(agent.X_history) < 2:
        return None

    xs = []
    ys = []
    zs = []
    for X in agent.X_history:
        xs.append(X[0]*1.0001)
        ys.append(X[1]*1.0001)
        zs.append(X[2]*1.0001)

    return ax.plot3D(xs, ys, zs=zs, color=color, linewidth=lw)


def draw_all(ax, agent, A=None, B=None, plot_trace=True, plot_agent=True, stride=4):
    vis_sphere = draw_hemisphere(ax, stride=stride)

    vis_agent = None
    vis_coord = None
    vis_trace = None
    if agent is not None:
        if plot_agent:
            vis_agent = draw_point(ax, agent.X)
            vis_coord = draw_agent_coord(ax, agent)

        if plot_trace:
            vis_trace = draw_trace(ax, agent, lw=1.0)

    vis_A = None
    if A is not None:
        vis_A = draw_point(ax, A, color='green')

    vis_B = None
    if B is not None:
        vis_B = draw_point(ax, B, color='blue')

    return vis_sphere, vis_agent, vis_coord, vis_trace, vis_A, vis_B


def clean_all(ax, vis_sphere, vis_agent, vis_coord, vis_trace, vis_A=None, vis_B=None):
    if vis_sphere:
        ax.collections.remove(vis_sphere)
    if vis_agent:
        ax.collections.remove(vis_agent)
    if vis_coord:
        for c in range(3):
            for l in vis_coord[c]:
                l.remove()
    if vis_trace:
        for t in vis_trace:
            t.remove()
    if vis_A:
        ax.collections.remove(vis_A)
    if vis_B:
        ax.collections.remove(vis_B)

