## Comparing Loss Functions for Partial Label Learning

This repository is a fork of the public repository [LW loss](https://github.com/hongwei-wen/LW-loss-for-partial-label) and it contains all experiments related to partial label learning for paper [Towards Unbiased Exploration in Partial Label Learning](TODO).

### Command to reproduce experiments with the large consistent synthetic dataset (Section 6.2, Figure 7)

`python zs29.py`

### Command to reproduce experiments with the CIFAR10 (Section 6.3, Figure 8)

`python zs31.py`

### Command to reproduce experiments with the CIFAR100 (Section 6.3, Figure 9)

`python zs30.py`

### Command to reproduce experiments with real PLL datasets (Section 6.5, Table 6)

TODO


### Reproducing plots

Once the experiments terminated, the plots (Figures 7, 8, 9) can be generated from the logs by running

`python plot.py`
