#!/usr/bin/env/python

# SPDX-License-Identifier: MIT
# See LICENSE file for additional copyright and license details.

import datetime
import numpy as np
import h5py
import modules
import pointgen

class Config:
    def __init__(self):
        self.d = dict()

    def __item__(self, itm):
        return self.d[itm]



class ExpansionItem:
    def __init__(self, scale, tick, start, targets, active_symbols, hits):
        self.scale = scale
        self.tick = tick
        self.start = start
        self.targets = targets
        self.active_symbols = active_symbols
        self.hits = hits


class BacktrackItem:
    def __init__(self, scale, ss, ps):
        self.scale = scale
        self.ss  = ss # symbols
        self.ps = ps # parents

    def __repr__(self):
        return "{}: ({}, {})".format(self.scale, self.ss, self.ps)


class SubgoalsItem:
    def __init__(self, scale, subgoals):
        self.scale = scale
        self.subgoals = subgoals

    def __repr__(self):
        return "{}: {}".format(self.scale, self.subgoals)


class SearchItem:
    def __init__(self, scale, starts, targets, hits):
        self.scale = scale
        self.starts = starts
        self.targets = targets
        self.hits = hits


class Recorder:
    def __init__(self, nscales=1):
        self.expansions = [list() for n in range(nscales)]
        self.backtracks = [list() for n in range(nscales)]
        self.searches = [list() for n in range(nscales)]
        self.subgoals = [None for n in range(nscales)]

    def record_expansion(self, scale, tick, start, targets, active_symbols, hits):
        self.expansions[scale].append(ExpansionItem(scale, tick, start, targets, active_symbols, hits))

    def record_backtrack(self, scale, ss, ps):
        self.backtracks[scale].append(BacktrackItem(scale, ss, ps))

    def record_subgoals(self, scale, subgoals):
        self.subgoals[scale] = SubgoalsItem(scale, subgoals)

    def record_search(self, scale, starts, targets, hits):
        self.searches[scale].append(SearchItem(scale, starts, targets, hits))



def save_trajectory_to_hdf5(cfg, Xs, Ys, verbose=True):
    filepath = "{}/{}".format(cfg.datadir, cfg.filename)
    if verbose:
        print("Saving results to '{}'".format(filepath))

    # store results to file
    f5 = h5py.File(filepath, 'w')

    # meta information
    f5.attrs['datetime'] = str(datetime.datetime.now())

    # environment and simulation setup
    f5.attrs['world_type'] = 'scenes.Circular'
    f5.attrs['world_radius'] = cfg.world_radius
    f5.attrs['platform_x'] = cfg.platform_x
    f5.attrs['platform_y'] = cfg.platform_y
    f5.attrs['platform_radius'] = cfg.platform_radius
    f5.attrs['dt'] = cfg.dt
    f5.attrs['maxticks'] = cfg.maxticks

    # data
    f5.create_dataset('agent_x', data=Xs)
    f5.create_dataset('agent_y', data=Ys)

    # done writing
    f5.close()


def load_trajectory_hdf5(filepath, verbose=True):
    if verbose:
        print("Loading results from '{}'".format(filepath))

    # initialize return values
    cfg = Config()
    Xs = []
    Ys = []

    # open file and read meta information
    f5 = h5py.File(filepath, 'r')
    cfg.creation_date = f5.attrs['datetime']

    # read environment and simulation data
    cfg.world_type = f5.attrs['world_type']
    cfg.world_radius = f5.attrs['world_radius']
    cfg.platform_x = f5.attrs['platform_x']
    cfg.platform_y = f5.attrs['platform_y']
    cfg.platform_radius = f5.attrs['platform_radius']
    cfg.dt = f5.attrs['dt']
    cfg.maxticks = f5.attrs['maxticks']

    # data
    Xs = f5['agent_x']
    Ys = f5['agent_y']

    # don't close file here

    return cfg, Xs, Ys


def gen_symbols(W, H, N=500, method='hammersley', mindist=0.035, nscales=1):
    """Generate N symbols.

    Args:
        W(int): width of the environment
        H(int): height of the environment
        N(int): number of symbols to generate
        method(string): method by which the symbol locations are generated. One
            of 'hammersley', 'rmind1', 'rmind2', 'runif'.

    Returns:
        []: List of Symbol objects

    """

    if method == 'runif':
        symbol_coords = pointgen.random_uniform(N, W, H)
    if method == 'rmind1':
        symbol_coords = pointgen.random_mindist(N, mindist, W, H)
    if method == 'rmind2':
        symbol_coords = pointgen.random_mindist(100, 0.08, W, H)
    if method == 'hammersley':
        symbol_coords = pointgen.hammersley(N, W, H)
    return [modules.Symbol(symbol_coords[i, :], nscales=nscales) for i in range(symbol_coords.shape[0])]


def target_reached(symbols, target):
    """Test if the target area was already found"""
    for s in symbols:
        if s in targets:
            return True
    return False


def sample_unit_sphere(radius):
    """Retrieve a normally distributed sample from the unit sphere"""
    l = np.sqrt(np.random.uniform(0, radius))
    a = np.pi * np.random.uniform(0, 2)
    x = l * np.cos(a)
    y = l * np.sin(a)
    return x, y


def get_closest_symbol(symbols, coord):
    """Return the symbol with shortest Euclidean distance to coord"""
    ret_s = 0
    ret_c = np.linalg.norm(coord - symbols[0].coord)

    for i in range(1, len(symbols)):
        c = np.linalg.norm(coord - symbols[i].coord)
        if c <= ret_c:
            ret_c = c
            ret_s = i

    return ret_s, ret_c


def get_symbol_id(symbols, s):
    for i in range(len(symbols)):
        if symbols[i] == s:
            return i
    return -1



def get_transition_obj(layers, symbols, scale, sid):
    t_id = symbols[sid].t[scale]
    return layers[scale].ts[t_id]



def get_parent_obj(layers, scale, symbols, s, origin, strategy='rand'):
    if strategy == 'rand':
        return symbols[s.getRandomParent()]

    elif strategy == 'mean':
        # strategy to select symbol closest to grid field center
        pid = s.getRandomParent()
        if pid == origin:
            return symbols[pid]

        parent_transition = symbols[pid].t[scale]
        transition_coord = layers[scale].ts[parent_transition].coord
        sid, _ = utils.get_closest_symbol(symbols, transition_coord)
        return symbols[sid]

    else:
        print("Unknown parent selection strategy")
        sys.exit(-1)


def intersect(l1, l2):
    return list(set(l1) & set(l2))
