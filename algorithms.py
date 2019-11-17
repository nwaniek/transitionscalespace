#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# See LICENSE file for additional copyright and license details.

import utils


def get_parent_objs(layers, scale, symbols, ss):
    # return set of all parents for all symbols simultaneously
    ps = []
    for s in ss:
        for p in s.parents:
            ps.append(symbols[p])

    # make elements unique
    ps = list(set(ps))
    return ps



def find_sequence_on_scale(layers, scale, symbols, start, target, recorder):
    """Find a sequence from start to target on a certain scale."""

    # initialize
    active_symbols = [start]
    targets = [target]

    tick = 0
    any_target_found = False
    recorder.record_expansion(scale, tick, start, targets, active_symbols, utils.intersect(targets, active_symbols))

    any_target_found = utils.intersect(targets, active_symbols) != []
    while not any_target_found:
        # update 'retrieval tick' -> this emulates refractory periods of place
        # cells. This comes from marking symbols as "expanded"
        for s in active_symbols:
            symbols[s].retrieval_tick = tick

        # predict next batch of symbols, and remove currently active ones
        next_symbols = layers[scale].expand(active_symbols, symbols, tick)
        active_symbols = [s for s in next_symbols if s not in active_symbols and symbols[s].retrieval_tick < 0]

        # update time
        tick += 1

        # record everything!!1
        recorder.record_expansion(scale, tick, start, targets, active_symbols, utils.intersect(targets, active_symbols))

        # check if we reached the destination
        any_target_found = utils.intersect(targets, active_symbols) != []
        if any_target_found:
            break

    return utils.intersect(targets, active_symbols)



# def backtrack_on_scale(layers, scale, symbols, start, targets, recorder):
#     """Backtrack on a certain scale from target to start.
#
#     This involves region based computing and propagation of information
#
#     start and target are indices"""
#
#     # this is the 'global' start
#     start_obj = symbols[start]
#
#     # setup symbol and fetch all parents
#     ss = [symbols[t] for t in targets]
#     ps = get_parent_objs(layers, scale, symbols, ss)
#
#     recorder.record_backtrack(scale, [utils.get_symbol_id(symbols, s) for s in ss],
#                                      [utils.get_symbol_id(symbols, p) for p in ps])
#
#     # this works, because we always start searching from the initial symbol. start is
#     # thus the parent node of _every_ trajectory
#     while not (start_obj in ps):
#         ss = ps
#         ps = get_parent_objs(layers, scale, symbols, ss)
#
#         # record everything!!1
#         recorder.record_backtrack(scale, [utils.get_symbol_id(symbols, s) for s in ss],
#                                          [utils.get_symbol_id(symbols, p) for p in ps])
#
#     return ss, ps
