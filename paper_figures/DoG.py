#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def _g(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x-mu) / sigma)**2)

def _mh(x, mu, sigma):
    # return 2 / (np.sqrt(3) * np.pi**0.25) * (1 - (x / sigma)**2) * np.exp(-x**2/(2*sigma**2))
    return 2 / (np.sqrt(3) * np.pi**0.25) * (1 - ((x - mu) / sigma)**2) * np.exp(-(x - mu)**2/(2*sigma**2))

def _g2nd(x, mu, sigma):
    return (-1) * - (sigma**2 - (x - mu)**2) / (sigma**5 * np.sqrt(2 * np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))

def dog(x, mu, sigma1, sigma2):
    return _g(x, mu, sigma1) - _g(x, mu, sigma2)

def conv(f, g, dx):
    c = np.convolve(f, g, 'same')
    c /= np.sum(c) * dx
    return c


xmin = -5
xmax = -xmin
ymin = -0.11
ymax =  0.28

nelems = 1000
dx = (xmax - xmin) / nelems
X = np.linspace(xmin, xmax, nelems)

mu0 = -1.0


def make_a(K, s):
    return (1 - K**2) / (2 * K**2 * s**2)

def make_c(K, s):
    return np.log(K)

def get_roots(K, s):
    a = make_a(K, s)
    c = make_c(K, s)
    r = np.sqrt(-4*a*c)/(2*a)
    return (r, -r)


# kernel size difference setup
approximate_LoG = False
K      = 1.600 if approximate_LoG else 2.0
sigma0 = 0.805 if approximate_LoG else 1/np.exp(1.0)*2
sigmaX = np.sqrt(2) * sigma0

# figure setup
f, ax = plt.subplots(1, 3)
f.set_figheight(3.5)
f.set_figwidth(11)

if True:
    ax_idx = 0
    X = np.linspace(xmin, xmax, nelems)

    _sigma0 = np.sqrt(2)
    G0 = _g(X, 0, _sigma0)
    GX = _g(X, 0, _sigma0 * np.sqrt(2))

    ymax = 0.3
    r = 2.0*_sigma0

    ax[ax_idx].plot([X[0], X[-1]], [0, 0], linestyle='-', color='black', lw=0.5)
    ax[ax_idx].plot([0, 0], [ymin, ymax], linestyle='-', color='black', lw=0.5)

    ax[ax_idx].plot([r, r], [ymin, ymax], linestyle='--', color='magenta')
    ax[ax_idx].plot([r * np.sqrt(2), r * np.sqrt(2)], [ymin, ymax], linestyle=':', color='blue')

    ax[ax_idx].plot(X, G0, label='DoG G0')
    ax[ax_idx].plot(X, GX, label='DoG GX')


if True:
    ax_idx = 1
    X = np.linspace(0, xmax, nelems)

    G0 = dog(X, 0, sigma0, K*sigma0) #g1 - g2
    GX = dog(X, 0, sigmaX, K*sigmaX)

    ymax = 0.3
    r = get_roots(K, sigma0)[1]

    # size difference between kernels
    ax[ax_idx].plot([X[0], X[-1]], [0, 0], linestyle='-', color='black', lw=0.5)
    ax[ax_idx].plot([r, r], [ymin, ymax], linestyle='--', color='magenta')
    ax[ax_idx].plot([r * np.sqrt(2), r * np.sqrt(2)], [ymin, ymax], linestyle=':', color='blue')

    ax[ax_idx].plot(X, G0, label='DoG G0')
    ax[ax_idx].plot(X, GX, label='DoG GX')


# merging the fields
if True:
    ax_idx = 2
    X = np.linspace(xmin, xmax, nelems)

    G0 = dog(X, +mu0, sigma0, K*sigma0) #g1 - g2
    G1 = dog(X, -mu0, sigma0, K*sigma0) #g3 - g4
    GX = dog(X, 0, sigmaX, K * sigmaX)

    ymax = 0.3

    ax[ax_idx].plot([xmin, xmax], [0, 0], linestyle='-', color='black', lw=0.5)
    ax[ax_idx].plot([0, 0], [ymin, ymax], linestyle='-', color='black', lw=0.5)

    ax[ax_idx].plot(X, G0, label='DoG G0')
    ax[ax_idx].plot(X, G1, label='DoG G1')
    ax[ax_idx].plot(X, GX, label='DoG GX')

plt.tight_layout()
plt.show()


