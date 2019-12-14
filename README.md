# transitionscalespace

The repository contains code that was used to generate data and figures for
Waniek, Transition scale-spaces: A computational theory for the discretized
entorhinal cortex, 2019.

Please see below how to create raw data and figures that are presented in the
Experiments section of the paper. See the code in folder ```paper_figures``` for
code that was used to generate some of the other figures in the paper, such as
the entropy of binary search with overlapping receptive fields. If you want to
create particles on a hemisphere similar to the ones illustrated in the Appendix
of the paper, then have a look at README_hemisphere.md for some working
parameters. You can use those in combination with the ```hemisphere_...```
scripts.

If you have any questions regarding the code, please don't hesitate from
contacting me.

# How to cite

If you use this code, or base new experiments on it, please make sure to cite
the corresponding article:

```
@article{waniek2019tss,
  author = {Waniek, Nicolai},
  title = {Transition scale-spaces: A computational theory for the discretized entorhinal cortex},
  publisher = {MIT Press},
  doi = {10.1162/neco\_a\_01255},
  note ={PMID: 31835003},
  journal = {Neural Computation},
  volume = {0},
  year = {0},
  pages = {1-65},
  url = {https://doi.org/10.1162/neco_a_01255 },
  eprint = {https://doi.org/10.1162/neco_a_01255}
}
```


# License

All code in this repository is under MIT License. See LICENSE file for details.


# Examples for TSS / Code used to generate data and figures in main body of the paper

Brief descriptions of the algorithms can be found in the paper. Of course you
can also just have a look at the file. Most of the code is support-code to
record activity of nodes in the graph. Have a look at the comments in the file
to discern what is algorithm and what is support-code.

```
./demo_01_linear_track.py
./demo_02_wavepropagation.py --N 250 --startX 0.2 --startY 0.2 --targetX 0.8 --targetY 0.8 --W 1.0 --H 1.0 --period 0.2 --M 300 --pointgen rand --save-figures
./demo_03_scalespace_refinement_descending.py --N 250 --startX 0.4 --startY 0.1 --targetX 1.8 --targetY 4.8 --W 2.0 --H 5.0 --M 200 --max-i 2 --nscales 5 --save-figures
./demo_04_scalespace_refinement_ascending.py --N 250 --startX 0.4 --startY 0.1 --targetX 1.8 --targetY 4.8 --W 2.0 --H 5.0 --M 200 --max-i 2 --nscales 5 --save-figures
./demo_05_watermaze.py trajectory_data/005.hdf5 --M 100 --save-figures
```

Code to generate figures from paper can be found in subfolder
```paper_figures```


# Hemisphere Examples / Code used to generate the geodesic figure in the appendix

To recreate the example of geodesic computation given in the appendix of the
paper, you first need to generate data for the field distribution on the
hemisphere, then compute samples from start to target, and finally visualize.
Or, after checking out this repository, directly jump to the third and final
step of visualization. In detail, the steps are as follows.

First, run the file ```hemisphere_walker.py```, which will spawn a virtual agent
on a hemisphere and keep it walking according to some statistics similar to a
rodent. While walking around, grid fields will be placed and self-organized
according to particle dynamics that were described in [1].  Specifically, the
closest particle to the agent will be moved closed to the agent (but slower than
the agent's movement). All other particles in the vicinity of the agent will be
repelled. After finishing the execution for some time (see line 58 of the file),
the particle locations will be stored to a .json file.

Second, run the file ```hemisphere_tss.py```. Despite the name, this file will
not create the TSS but only perform trajectory sampling using Dijkstra and
backtracking. After finishing, this file will write its results to a file. To
change the file name, adapt line 61 of the file.

Third, and finally, visualize everything using ```hemisphere_visualizer.py```.
If you changed the filenames in the other scripts, make sure to also adapt this
file at lines 47 -- 48.


[1] Waniek Hexagonal Grid Fields Optimally Encode Transitions in Spatiotemporal
Sequences, 2018, Neural Comput. 2018 Oct;30(10):2691-2725. doi:
10.1162/neco_a_01122, https://www.mitpressjournals.org/doi/abs/10.1162/neco_a_01122
