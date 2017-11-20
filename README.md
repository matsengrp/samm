# SAMM

SAMM is a method for estimating somatic hypermutation rates of nucleotide motifs.
The method is based on a survival analysis model and uses MCEM to find parameters that
maximize the likelihood of the data.

## Installation
Note: `samm` requires Biopython version 1.65.

If you plan on simulating somatic hypermutation on a tree via `gctree` or simulating germline sequences via `partis`, you will need to run
```git submodule update --init --recursive```

To visualize the model by creating the rainbow barcharts, the code calls out to `R`. Required packages include `ggplot2`, `dplyr`, `gridExtra`, `methods`, `lazyeval`.
