#!/usr/bin/env python

import random
import numpy as np
#try:
#    import ujson as json
#except:
import json



def dist(X, Y, dist_type='angle'):
    # compute distance between particle and some input
    if dist_type == 'angle':
        # this computes the angular distance between two inputs. this is
        # primarily useful for particles that live on a unit sphere.
        # otherwise, results are not guaranteed to be useful
        return np.arccos(X.T.dot(Y))
    elif dist_type == 'euclidean':
        return np.linalg.norm(X - Y)
    else:
        raise RuntimeError(f"Unknown dist_type '{dist_type}'")



class Particle:
    def __init__(self, X, **kwargs):
        """Initialize a particle.

        Args:
            X (array-like): Initial position of the particle
            dist (str): distance computation function. One of 'euclidean', 'angle'
        """
        self.X = X
        self.set_config(**kwargs)
        self.reset_push_pull()


    def set_config(self, **kwargs):
        self.dist_type = kwargs.get('dist_type', 'angle')
        self.alpha = kwargs.get('alpha', 0.2)
        self.alpha_decay = kwargs.get('alpha_decay', 0.99999)
        self.mem = kwargs.get('mem', 0.8)


    def dist(self, X):
        return dist(self.X, X, self.dist_type)


    def normalize(self):
        """Normalize X.

        This is only useful when the agent that uses the particle system lives
        on a unit sphere.
        """
        self.X /= np.linalg.norm(self.X)


    def apply(self):

        # compute new position
        Xnew = self.X + (self.alpha * self.pull - 5.0 * self.alpha * self.push)
        Xold = self.X

        # smoothed update + renormalize (XXX: only useful for unit sphere agents)
        tmp = self.mem * Xold + (1.0 - self.mem) * Xnew
        tmp /= np.linalg.norm(tmp)

        # test if this is ok. TODO: no hard thresholding here! this only
        # works in the hemisphere example
        if True:
            if tmp[2] >= 0:
                self.X = tmp
        else:
            self.X = tmp

        # slowly decay alpha
        self.alpha *= self.alpha_decay


    def reset_push_pull(self):
        self.push = np.zeros(self.X.shape)
        self.pull = np.zeros(self.X.shape)


    def from_json(self, state_dict):
        self.set_config(**state_dict)
        X = state_dict['X']
        self.X = np.asarray(X)
        self.reset_push_pull()


    def to_json(self):
        state_dict = {
                # meta
                'dist_type'   : self.dist_type,
                'alpha'       : self.alpha,
                'alpha_decay' : self.alpha_decay,
                'mem'         : self.mem,
                # data
                'X'           : [float(self.X[0]), float(self.X[1]), float(self.X[2])],
        }
        return state_dict



class PushPullParticleSystem:

    def __init__(self, **kwargs):
        super(PushPullParticleSystem, self).__init__()
        self.particles = []
        self.set_config(**kwargs)


    def set_config(self, **kwargs):

        # settings for particle interactions
        self.mindist = kwargs.get('mindist', 0.1)
        self.maxdist = kwargs.get('maxdist', 2.0 * self.mindist)
        self.dist_type = kwargs.get('dist_type', 'euclidean')

        # update and interaction strenghts for particles. will be passed forward
        # during construction of new particles
        self.alpha = kwargs.get('alpha', 0.15)
        self.alpha_decay = kwargs.get('alpha_decay', 0.99999)
        self.mem = kwargs.get('mem', 0.8)

        # threshold and random chance for creation of new symbol
        self.mindist_rnd_threshold = kwargs.get('mindist_rnd_threshold', 0.9)
        self.mindist_rnd_chance = kwargs.get('mindist_rnd_chance', 0.1)


    def __len__(self):
        return len(self.particles)


    def __getitem__(self, i):
        assert i < len(self), "Index out of bounds"
        return self.particles[i]

    def save_json(self, path : str):
        # create state dict with configuration
        state_dict = {
                # meta
                'dist_type'             : self.dist_type,
                'mindist'               : self.mindist,
                'maxdist'               : self.maxdist,
                'mindist_rnd_threshold' : self.mindist_rnd_threshold,
                'mindist_rnd_chance'    : self.mindist_rnd_chance,
                'alpha'                 : self.alpha,
                'alpha_decay'           : self.alpha_decay,
                'mem'                   : self.mem,
                # data
                'particles'             : list(),
        }

        data = self.particles[0].to_json()
        state_dict['particles'].append(data)

        for p in self.particles:
            state_dict['particles'].append(p.to_json())

        with open(path, 'w') as f:
            json.dump(state_dict, f, indent=1)


    def load_json(self, path : str):
        with open(path, 'r') as f:
            state_dict = json.load(f)

        self.set_config(**state_dict)
        for p in state_dict['particles']:
            self.particles.append(Particle(np.array([0,0,0])))
            self.particles[-1].from_json(p)


    def avg_dist(self):
        """Compute the average distance between closest particles"""
        raise NotImplementedError("Not yet done.")
        mean = 0.0
        std = 0.0
        return mean, std


    def spawn_particle(self, X):
        particle_args = {
                'dist'        : self.dist_type,
                'alpha'       : self.alpha,
                'alpha_decay' : self.alpha_decay,
                'mem'         : self.mem,
                }
        self.particles.append(Particle(X, **particle_args))
        return self.particles[-1]


    def update(self, X):
        """Compute updates in the particle system.

        Args:
            X (array-like): Euclidean coordinate / input

        Returns:
            int: index of the
        """

        # no particle yet, so let's create the first one
        if len(self) == 0:
            self.spawn_particle(X)
            return 0

        # compute distance from all particles to target
        ds = [p.dist(X) for p in self.particles]

        # find all particles that are within a treshold-distance. store only the
        # indices
        rs = []
        for i in range(len(ds)):
            if ds[i] <= self.mindist:
                rs.append(i)

        # if we have no particle close to this location, let's spawn a new one
        if len(rs) == 0:
            self.spawn_particle(X)
            rs.append(len(self)-1)
        else:
            # close to minimal distance and some luck? create a new particle
            md = min(ds)
            rnd = False
            if md > (self.mindist_rnd_threshold * self.mindist):
                rnd = random.random() > (1 - self.mindist_rnd_chance)
            if rnd:
                self.spawn_particle(X)

        # there's at least one particle with its receptive field covering X. We
        # thus need to find the winner. This effectively implements
        # winner-take-all dynamics
        w = rs[0]
        for i in range(1, len(rs)):
            if ds[rs[i]] < ds[w]:
                w = rs[i]

        # reset all particles' push pull dynamics
        for i in range(len(rs)):
            ri = rs[i]
            self[ri].reset_push_pull()

        # compute local push-pull interactions of particles
        for i in range(len(rs)):
            # skip processing the winner. XXX: is this correct? pull closer to current state?
            if w == rs[i]:
                continue

            # get real index
            ri = rs[i]

            # local interactions between all particles. O(N^2) time :E could be
            # reduced (because of symmetry), but i'm just too lazy
            for j in range(len(rs)):
                # get real particle indices
                rj = rs[j]

                # skip self-reinforcing interactions
                if ri == rj:
                    continue

                # vector for direction
                v = self[rj].X - self[ri].X
                v /= np.linalg.norm(v)

                # distance
                d = dist(self[rj].X, self[ri].X)

                # (weighted) push or pull?
                if d <= self.mindist:
                    self[ri].push += d * v
                elif d <= self.maxdist:
                    self[ri].pull += d * v


        # update according to push/pull
        for i in range(len(rs)):
            # skip processing the winner. XXX: is this correct? pull closer to current state?
            if w == rs[i]:
                continue

            # get real particle index
            ri = rs[i]

            self[ri].apply()



        return w


