#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# See LICENSE file for additional copyright and license details.

import random
import numpy as np
import pointgen
import algorithms


class Symbol():
    def __init__(self, coord, nscales=1):
        """Initialze a new symbol and pin it to a coordinate. A symbol
        corresponds to a place cell.

        Args:
            coord(np.array(shape=[1,2])): static coordinate of the symbol.
        """

        # symbol is located relative to a spatial coordinate
        self.coord = coord

        # reset stuff
        self.reset(nscales=nscales, reset_transition=True)


    def updateParents(self, ps, tick):
        """Add parents to the list of parents.

        This is for backtracking when a target in the graph search was found.
        The parents may not be updated

        In the standard MTS model, we don't need to store any distance or cost,
        because we assume unit cost between neighboring symbols. Therefore, as
        soon as a symbol gets expanded and it has not received any parents, this
        is the minimal cost path. The cost here corresponds to temporal
        expansion time!
        """

        # XXX: maybe check here for >= ?
        if (len(self.parents) == 0) or (self.parent_tick == tick):
            self.parents += ps
            # parent tick simply stores when the parents were received
            self.parent_tick = tick


    def getRandomParent(self):
        if len(self.parents) == 0:
            return []
        return random.choice(self.parents)


    def getWeightedRandomParent(self):
        if len(self.parents) == 0:
            return []

        # sort all parents by the weight of the connectivity
        # TODO: implement


    def reset(self, nscales=1, reset_transition=False):
        # monitor if already retrieved or not. This corresponds to removing the
        # symbol from the open set (or Queue Q) in Dijkstra's algorithm. If the
        # retrival tick >= 0, then we don't expand this symbol any more
        self.retrieval_tick = -1

        # parent symbols (ids) during graph expansion.
        self.parents = []
        # this stores when this symbol received parents. this is purely for
        # maintenance
        self.parent_tick = -1

        if reset_transition:
            # symbol is linked to a transition t on each scale
            self.t = [None for n in range(nscales)]


class Transition():
    def __init__(self, coord):
        """Initialize a new transition and pin it to a coordinate. A transition
        is part of a bundle that, in turn, corresponds to a grid cells.

        Args:
            coord(np.array(shape=[1,2])): static coordinate of the symbol.
        """
        # each static transition is anchored at a certain location
        self.coord = coord

        # transition is defined for some symbols (this only contains indexes)
        self.domain = []

        # transition leads to some symbols (this only contains indexes)
        self.image = []

        # each transition has a set of neighbor transitions. This is only used
        # to accelerate the computation of image
        self.nbrs = []


    def isDefinedForAny(self, symbols):
        """Determine if this transition is defined for a set of symbols"""
        any = False
        for s in symbols:
            any = s in self.domain
            if any:
                break
        return any


    def intersectDomain(self, symbols):
        """Return the intersection of symbols with the domain of this transition"""

        intersection = []
        for s in symbols:
            if s in self.domain:
                intersection.append(s)
        return intersection


    def intersectImage(self, symbols):
        """Return the intersection of symbols with the image of this transition"""

        intersection = []
        for s in symbols:
            if s in self.image:
                intersection.append(s)
        return intersection


    def leadsTo(self):
        return self.image


    def distEuclidean(self, coord):
        """Euclidean distance between this transition and a coordinate in the
        input space."""
        return np.linalg.norm(self.coord - coord)


    def plot(self, ax, **kwargs):
        """Plot the transition center

        Args:
            self(Transition): the static transition
            ax(matplotlib.axes.Axes): axis to plot onto
            color: color to use during plotting
        """
        col = kwargs.pop('color', 'grey')
        marker = kwargs.pop('marker', 'o')
        markersize = kwargs.pop('markersize', 5)
        alpha = kwargs.pop('alpha', 1.0)
        ax.plot(self.coord[0], self.coord[1], color=col, marker=marker, markersize=markersize, alpha=alpha)


    def plot_highlighted(self, ax, symbols, **kwargs):
        """Plot the transition information on an axis object as highlighted information.

        Args:
            self(Transition): the static transition
            ax(matplotlib.axes.Axes): axis to plot onto
            symbols: list of symbols
            color_transition: Color for plotting the transition center
            color_domain: Color for plotting symbols the transition is defined for
            color_image: Color for plotting symbols the transition leads to

        """
        col = kwargs.pop('color_highlight', 'black')
        marker = kwargs.pop('marker', 'o')
        markersize = kwargs.pop('markersize', 5)
        ax.plot(self.coord[0], self.coord[1], color=col, marker=marker, markersize=markersize)


class TransitionLayer():
    def __init__(self, period, width, height, pointgen_fn=pointgen.hex, particles=None):
        """Initialize a static layer of transition cells.

        Args:
            period(float): distance (e.g. in m) between transition centers
            width(float): width of a bounding box around the environment
            height(float): height of a bounding box around the environment
            pointgen_fn(function): function to generate points given width,
                height, and period. This alters the lattice of transition
                centers. Default=pointgen.hex

        """
        # period between transition centers
        self.period = period
        # world information required to compute sufficiently many centers
        self.width = width
        self.height = height
        # point generation function for transition centers.
        self.pointgen_fn = pointgen_fn
        self.particles = particles

        # initialize fields
        self.grid_coords = []
        self.ts = []

        self.compute_centers()
        self.compute_neighborhoods()


    def size(self):
        return len(self.ts)


    def associate(self, symbols, scale=0):
        """Associate this transition layer with symbols.

        This computes the defined-for and leads-to sets for each transition.

        Args:
            symbol(list(Symbol)): list of symbols to associate with
        """
        self._compute_domain(symbols, scale)
        self._compute_image()


    def expand(self, active_symbols, all_symbols, tick, flag=0):
        """Given a set of active symbols, returns expanded symbols (that is, the next set of symbols)

        It is assumed that all symbols are 'true' in the sense of the logic of
        Multi-Transition Systems. In terms of Dijkstra, this is one expansion.

        Args:
            current_symbol(list(int)): list of symbols that are currently expanded
            all_symbols(list(Symbols)): all symbols that are in use
            tick(int): time tick, required for parent updating
        """
        # iterate over all transitions, and determine if they are defined for
        # the input symbol
        predicted_symbols = []
        for transition in self.ts:
            if transition.isDefinedForAny(active_symbols):

                # get the image of the transition
                image = transition.leadsTo()

                # add the image to the set of next symbols
                for s in image:
                    # TODO: add forward cost in here?

                    #if all_symbols[s].flag == flag:
                    predicted_symbols.append(s)

                # update parent information for backtracking
                syms = transition.intersectDomain(active_symbols)
                for s in image:
                    # to avoid biasing
                    all_symbols[s].updateParents(syms, tick)

        # make elements unique
        predicted_symbols = set(predicted_symbols)
        return list(predicted_symbols)


    def getTransitionBySymbol(self, symbol):
        """The the transition that is defined for a symbol"""
        for t in self.ts:
            if t.isDefinedForAny([symbol]):
                return t
        return None


    def getClosestTransition(self, coord):
        """Get the transition closest to a coordinate"""
        closest_id   = 0
        closest_dist = self.ts[0].distEuclidean(coord)
        for i in range(1, len(self.ts)):
            dist = self.ts[i].distEuclidean(coord)
            if dist < closest_dist:
                closest_id = i
                closest_dist = dist

        return self.ts[closest_id]


    def getDefinedTransitions(self, symbols):
        """Return a list of all transitions whose domain contains any of the symbols."""

        ts = []
        for i, transition in enumerate(self.ts):
            if transition.isDefinedForAny(symbols):
                ts += [i]

        ts = set(ts)
        return list(ts)


    def compute_centers(self):
        """Create all transition centers for this layer.

        Given that this is a static transition layer, the centers will be placed
        on a perfectly symmetrical hexagonal grid lattice.
        """

        if not (type(self.pointgen_fn) == str):
            print("Computing centers")
            # create coords using pointgen_fn
            self.grid_coords = self.pointgen_fn(self.width, self.height, self.period)

        elif self.pointgen_fn == 'particles':
            print("Using particle locations as transition centers")
            # get coords from particles
            self.grid_coords = np.zeros((len(self.particles), len(self.particles[0].X)))
            for i in range(len(self.particles)):
                self.grid_coords[i, :] = self.particles[i].X

        else:
            # what?
            raise RuntimeError(f"Unknown pointgen_fn '{self.pointgen_fn}'")

        # initialize all transitions
        self.ts = [Transition(self.grid_coords[i, :]) for i in range(self.grid_coords.shape[0])]


    def compute_neighborhoods(self):
        """Compute all neighbors of each transition.

        This is mostly to speed up association with symbols later and retrieval
        of information.
        """

        print("Computing neighborhood")
        for a in self.ts:
            a.nbrs = []
            for i,b in enumerate(self.ts):
                if a == b:
                    continue

                alpha = 0.01
                # TODO: include real voronoi clustering here. number changes
                # because learned particles might have imperfect neighbor
                # distances
                if (type(self.pointgen_fn) == str) and (self.pointgen_fn == 'particles'):
                    alpha = 0.2

                # use alpha to capture numerical inaccuracies, and for
                # transitions that were placed using the push-pull particle
                # system
                if pointgen.dist_euclidean(a.coord, b.coord) <= (1 + alpha) * self.period:
                    a.nbrs.append(i)


    def _compute_domain(self, symbols, scale=0):
        """For each transition, compute the set of symbols it is defined for."""

        print("Computing domain")
        for si, s in enumerate(symbols):
            # transition with minimal distance
            min_td = np.inf
            # index of transition with minimal distance
            min_ti = -1
            # find by brute force. This does not scale, but is not important in
            # this demo
            for ti, t in enumerate(self.ts):
                d = pointgen.dist_euclidean(s.coord, t.coord)
                if d < min_td:
                    min_td = d
                    min_ti = ti

            # if this transition was closest to the symbol, it is defined for it
            if min_td < np.inf:
                self.ts[min_ti].domain.append(si)
                s.t[scale] = min_ti
            else:
                print("EE: found symbol without assigned transition")

        # make elements in each list unique
        t.domain = list(set(t.domain))


    def _compute_image(self):
        print("Computing image")
        """For each transition, compute the set of symbols it leads to"""
        # simply go through neighbors and collect their list of symbols they are
        # defined for, and collect them as the set of leads to
        for t in self.ts:
            for n in t.nbrs:
                for s in self.ts[n].domain:
                    t.image.append(s)
            # make elements unique
            t.image = list(set(t.image))


    def plot(self, ax, color='#000000'):
        """Plot all transitions stored within the layer

        Args:
            self(Transition): the static transition
            ax(matplotlib.axes.Axes): axis to plot onto
            color: color to use during plotting
        """
        for t in self.ts:
            t.plot(ax, color)


    def plot_with_highlight(self, ax, symbols, highlighted=[], **kwargs):
        for i,t in enumerate(self.ts):
            if i in highlighted:
                t.plot_highlighted(ax, symbols, **kwargs)
            else:
                t.plot(ax, **kwargs)



