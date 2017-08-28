# samm tutorial

## Simulating data

### Creating a mutation model

A mutation model is defined by the parameters in the survival model, which describes how the mutation rate changes when particular motifs are present.
To create a mutation model from scratch, you can run, for example,
```
python generate_theta.py --motif-lens 3 --positions-mutating 1 --output-model _output/true_model.pkl
```
This creates a 3-mer motif model where the center position mutates (positions are indexed starting from zero).
If you want to have a hierarchical model, `--motif-lens` accepts a comma-separated list of motif lengths and `--positions-mutating` accepts a colon-separated list of comma-separated lists of positions that mutate. (For example, a 2,3-mer model where we include both offsetted 2-mers in the model can be specified with `--motif-lens 2,3 --positions-mutating 0,1:1`.)
If you would like to model individual parameters for different target nucleotides (e.g. a per-target model), then add `--per-target`.

You can also skip this step and use a pre-specified mutation model, but it should be in the proper pickled format.

### Simulating sequences

Now we would like to simulate sequences given a mutation model. This can be done by the command
```
python simulate_from_survival.py --agg-motif-len 3 --input-model _output/true_model.pkl --n-germlines <number-of-naive-sequences>
```
The argument `--agg-motif-len` specifies the length of the k-mer motifs in the aggregate form of the inputted mutation model (it assumes the motifs are centered).

In addition, use the following arguments to modify characteristics of the sequence generation process:

|  argument        |  description
|------------------------|----------------------------------------------------------------------------
|  --random-gene-len    |  the length of sequences
|  --min-percent-mutated | the minimum amount to mutate a sequence
|  --max-percent-mutated | the maximum amount to mutate a sequence

## Fitting a model

To fit a model given naive and mutated sequences, run a command like the following:
```
python fit_context_model.py --motif-lens 3 --positions-mutating 1 --penalty-params 0.1 --input-seqs _input/seqs.csv --input-genes _input/genes.csv
```
Above we are fitting a 3-mer motif model where the center position mutates. (To fit a hierarchical model, see comment above.)

## Visualizing the model

Finally, we can visualize the fitted model.
```
python create_bar_plots.py --input-pkl _output/fitted.pkl ?????
```
TBD: Needs to be updated.
