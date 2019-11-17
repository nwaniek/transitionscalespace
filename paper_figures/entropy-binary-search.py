#!/usr/bin/env python

"""
This script generates figures of the entropy of binary search with respect to
various overlaps of receptive fields.
"""

import numpy as np
import matplotlib.pyplot as plt


def H_epsilon_dk(k, eps):
    """Derivative with respect to k of the entropy and overlap eps"""
    return eps * (np.log(k) - np.log(1 + k * eps))

def H_epsilon(k, eps):
    """Entropy of binary search at split k with overlap eps"""
    return k * eps * np.log(k) - (1 + k * eps) * np.log(1 + k * eps)

def H_split(k, N, eps):
    """Entropy of the split in binary search including overlap, specified by
    eps"""
    return (k / N) * (np.log(k) + H_epsilon(k, eps)) + ((N - k) / N) * (np.log(N - k) + H_epsilon(N - k, eps))

def H(N, eps):
    # return np.log(N) + N * eps * np.log(N) - (1 + N * eps) * np.log(1 + N * eps)
    return - N * (1 / N + eps) * np.log(1 / N + eps)


if __name__ == "__main__":

    # number of fields/bins to assume
    N = 1000
    # overlap width. note that an overlap of 1/(2*N) means 50% overlap of fields
    e = 1e-5
    # configuration of different overlaps to consider, starting at zero overlap
    epsilon = [0.0, 1*e, 5*e, 10*e]

    # Figure setup
    fig = plt.figure(figsize=(20,4))
    ax = fig.subplots(1, 4)

    #
    # First part of plots: entropy and H_epsilon for different epsilons
    #
    M = 100
    Xs = np.zeros(M)
    H_base = np.zeros(M)
    Ht = np.zeros(M)
    He = np.zeros(M)
    eps = 0

    for i in range(M):
        eps = e * i

        Xs[i] = N * eps * 100
        H_base[i] = H(N, 0)
        Ht[i] = H(N, eps)
        He[i] = H_epsilon(N, eps)

    l0 = ax[0].plot(Xs, H_base, ls='-', color='black')
    l1 = ax[0].plot(Xs, Ht, ls='--', color='black')
    ax[0].set_xlabel(r'Overlap $\varepsilon$ in % of receptive field size')
    ax[0].set_ylabel(r'$H_N$')
    ax[0].set_title(r'Entropy $H_N$ over overlap $\varepsilon$')
    ax[0].legend(['baseline (0 overlap)', 'with overlap'])
    # ax[0,0].plot(Xs, He, ls=':')

    #
    # Entropy and difference (maximized)
    #
    styles = ['-', '--', '-.', ':']
    for i, eps in enumerate(epsilon):
        Xs = np.arange(1, N+1)
        Ht = np.zeros(len(Xs))
        Hs = np.zeros(len(Xs))
        Hd = np.zeros(len(Xs))
        He = np.zeros(len(Xs))
        for k in range(1, N+1):
            Xs[k-1] = k
            Ht[k-1] = H(N, eps)
            Hs[k-1] = H_split(k, N, eps)
            He[k-1] = H_epsilon(N, eps)
            Hd[k-1] = Ht[k-1] - Hs[k-1]

        ax[1].plot(Xs, Hs, ls=styles[i], color='black')
        ax[2].plot(Xs, Hd, ls=styles[i], color='black')

    ax[1].set_title(r'$H_c(k)$ over $k$ for different $\varepsilon$')
    ax[2].set_title(r'$H_N - H_c(k)$ over $k$ for different $\varepsilon$')
    ax[1].set_ylabel(r'$H_N$')
    ax[2].set_ylabel(r'$H_N - H_c(k)$')
    ax[1].legend([fr'$\varepsilon$ = {eps * N * 100:1.0f} %' for eps in epsilon])
    ax[2].legend([fr'$\varepsilon$ = {eps * N * 100:1.0f} %' for eps in epsilon])

    #
    # zero crossing of kHe(k) - (N-k)He(N-k)
    #
    styles = ['-', ':', '-.', '--']
    ls = []
    for i, _eps in enumerate(epsilon): #, 0.1, 0.3, 0.5]):
        print(_eps)
        Xs = np.arange(1, N+1)
        He = np.zeros(len(Xs))

        for k in range(0, N+1):
            Xs[k-1] = k
            #He[k-1] = H_epsilon_dk(k, _eps)
            He[k-1] = k * H_epsilon(k, _eps) - (N - k) * H_epsilon(N - k, _eps)

        ax[3].plot(Xs, He, ls=styles[i], color='black', label=f"" if _eps == 0.0 else fr"$\varepsilon$ = {_eps * N * 100:1.0f} %")

    ax[3].set_title(r'Zero-crossing of $kH_e(k) + (N - k)H_e(N-k)$ for $\varepsilon > 0$')
    ax[3].set_ylabel(r'$\frac{d}{dk} kH_e(k) + (N - k)H_e(N-k)$')
    ax[3].legend()





    #
    # fix xticks
    #
    for i in [1,2,3]:
        ax[i].set_xticks([0, N/2, N])
        ax[i].set_xticklabels([0, 'N/2', 'N'])


    plt.show()

