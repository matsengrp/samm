# Simulation plan

## Models to fit

- Basic
- `shazam`
- `mobeef`
    - 1-column no lasso
    - 1-column lasso
    - 4-column lasso
    - 1-column fused lasso (fusing on motifs then estimate probabilities using the same fusion graph)

## Simulations

### gctree simulated data

1. Run all models
    - choosing random sequence from each cluster
    - imputing ancestors with maximum parsimony
    - using known ancestors from `gctree`
2. Check differences in parameters
3. Compare results to truth

### simulated from survival model

1. Simulate survival model from random DNA sequences using a
    - star tree 1-column
    - star tree 4-column
2. Fit all models (see above)
3. Check differences in parameters
4. Compare results to truth

### parsimony survival model

1. Run `mobeef` model(s) on CTMC-generated data (multiple mutations) and parsimony-generated data (one mutation)
2. Compare between
3. Compare to truth

## Real data

### Datasets

- Cui et al (2016)
- Example dataset from `shazam` package
- MG data (Vander Heiden paper, p=0.5 paper)
- other publicly available datasets: [http://b-t.cr/t/publicly-available-airr-seq-data-sets/317](here)

1. Run all models
    - choosing best approach from simulations (random choice vs. imputed ancestors)
2. Check vs. known hot/cold spots
3. Compare between datasets for conserved/varying motifs
4. Predictive check with summary statistics
    - per-site mutability of known germline sequences, then also nucleotide content
    - motif content of observed sequences and simulated sequences
    - Vladimir rate idea
