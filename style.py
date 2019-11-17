#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# See LICENSE file for additional copyright and license details.

import utils

# Color configuration
color = utils.Config()
color.start             = '#add3ff'
color.target            = '#88e275'
color.transition        = '#ce1cd3'
color.domain            = '#e9e9e9'
color.sample            = '#0468b9'
color.scene             = 'black'
color.active            = 'black'
color.inactive          = '#b9b9b9'
color.trajectory        = 'black'
color.subgoal           = '#ce1cd3'

# Alpha configuration
alpha = utils.Config()
alpha.sample            = 0.1
alpha.trajectory        = 1.0
alpha.domain            = 1.0
alpha.symbol            = 1.0
alpha.active            = 1.0
alpha.inactive          = 1.0
alpha.start             = 1.0
alpha.target            = 1.0
alpha.subgoal           = 1.0

# line width
lw = utils.Config()
lw.active               = 0.3
lw.inactive             = 0.2
lw.scene                = 2.0
lw.sample               = 0.3
lw.trajectory           = 0.75

# zorder plotting configuration
zorder = utils.Config()
zorder.scene            = 2
zorder.watermaze_target = 6
zorder.watermaze_start  = 6
zorder.trajectory       = 5
zorder.sample           = 7
zorder.symbol           = 4
zorder.domain           = 1
zorder.start_symbol     = 10
zorder.target_symbol    = 10
zorder.subgoal          = 3
