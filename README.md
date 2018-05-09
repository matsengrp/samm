# SAMM

SAMM is a method for estimating somatic hypermutation rates of nucleotide motifs.
The method is based on a survival analysis model and uses MCEM to find parameters that maximize the likelihood of the data.

## Installation

Note: `samm` requires `python` 2.7.

The necessary `python` packages required to run `samm` are
```
- numpy
- scipy
- pandas
- biopython >= 1.65
- ete3
```

To visualize the model by creating the rainbow barcharts, the code calls out to `R`.
Install the required packages by running
```
R --vanilla --slave -e 'install.packages(c("seqinr", "ggplot2", "dplyr", "gridExtra", "tidyr"), repos="http://cran.rstudio.com/")'
```

## Running `scons` scripts

If you want to run the `scons` scripts, you will need a file `data_paths.py` that has two directory paths in it:

- `SCRATCH_DIR`: where intermediate output from Gibbs sampling will be sent
- `CUI_DATA_PATH`: where `partis`-processed data from Cui et al. (2016) lives, if running `get_hedgehog` or `imputed_ancestors_comparison`

## Feature-rich simulation

If you plan on simulating somatic hypermutation on a tree via [gctree](https://github.com/matsengrp/gctree/) or simulating germline sequences via [partis](https://github.com/psathyrella/partis), you will need to run
```
git submodule update --init --recursive
```
then install these packages according to their instructions.

## Imputing ancestors

Imputing ancestors with parsimony requires `PyQt4`, which is most easily installed via `conda`.

## Tutorial

### Creating a mutation model

A mutation model is defined by the parameters in the survival model, which describes how the mutation rate changes when particular motifs are present.
To create a mutation model from scratch, you can run, for example,
```
mkdir _output
python generate_theta.py --motif-lens 3 --positions-mutating 1 --output-model _output/true_model.pkl
```
This creates a 3-mer motif model where the center position mutates (positions are indexed starting from zero).
If you want to have a hierarchical model, `--motif-lens` accepts a comma-separated list of motif lengths and `--positions-mutating` accepts a colon-separated list of comma-separated lists of positions that mutate. (For example, a 2,3-mer model where we include both offsetted 2-mers in the model can be specified with `--motif-lens 2,3 --positions-mutating 0,1:1`.)
If you would like to model individual parameters for different target nucleotides (e.g. a per-target model), then add `--per-target`.

You can also skip this step and use a pre-specified mutation model, but it should be in the proper pickled format.

### Simulating sequences

Now we would like to simulate sequences given a mutation model. This can be done by the command
```
python simulate_shm_star_tree.py --agg-motif-len 3 --input-model _output/true_model.pkl --n-naive 30 --tot-mutated 30 --output-mutated _output/mutated.csv --output-naive _output/naive.csv --random-gene-len 100
```
The above code generates 30 naive sequences each 100 nucleotides-long and generates 30 mutated sequences based on the naive sequences.
The argument `--agg-motif-len` specifies the length of the k-mer motifs in the aggregate form of the inputted mutation model (it assumes the motifs are centered).

In addition, use the following arguments to modify characteristics of the sequence generation process:

|  argument        |  description
|------------------------|----------------------------------------------------------------------------
|  `--min-percent-mutated` | the minimum amount to mutate a sequence
|  `--max-percent-mutated` | the maximum amount to mutate a sequence

### Fitting a model

To fit a model given naive and mutated sequences, run a command like the following:
```
python fit_samm.py --motif-lens 3 --positions-mutating 1 --penalty-params 0.1 --input-mutated _output/mutated.csv --input-naive _output/naive.csv --out-file _output/fitted.pkl --em-max-iters 5
```
Above we are fitting a 3-mer motif model where the center position mutates. (To fit a hierarchical model, see comment above.)
We only use a single penalty parameter. One can also provide many penalty parameters (in decreasing order) so that `samm` can tune the penalty parameter.
We only run the MCEM for 5 iterations above, as this is a tutorial. We recommend running at least 10 iterations for typical situations.

The code will take a couple minutes. If you plan on running a lot of data in `fit_samm.py`, we recommend using the multithreading option (`--num-threads`) and our job-submission option (`--num-jobs`).
Currently we assume the job scheduling system is Slurm.

### Visualizing the model

Finally, we can visualize the fitted model.
```
python plot_samm.py --input-pkl _output/fitted.pkl --output-pdf _output/fitted.pdf
```

### Computing the log-likelihood on a tree

To obtain the log-likelihood of a supplied tree under a 5mer model, use the module `likelihood_of_tree_from_shazam` from `samm_rank.py`.
It requires an `ete` tree as input as well as a `SHazaM`-style mutability and substitution `*.csv` file.
The likelihood computed is the marginal likelihood.
Chib's method is used to integrate out mutation order along all the branches, though determining `num_samples` and `num_tries` will be dataset-dependent.
The following code will rank a tree that was simulated via [bcr-phylo](https://github.com/matsengrp/bcr-phylo-benchmark#sequence-simulation):

```
import pickle
from samm_rank import likelihood_of_tree_from_shazam

with open('test/data/neutral_sim_lineage_tree.p', 'rb') as fh:
    tree = pickle.load(fh)

mutability_file = 'R/shmulate_params/mutability.csv'
substitution_file = 'R/shmulate_params/substitution.csv'

likelihood = likelihood_of_tree_from_shazam(
    tree,
    mutability_file,
    substitution_file,
)
```
