# Simulation plan

## Models to fit

- Basic
- `shazam`
- `mobeef`
    - 1-column lasso
    - 4-column lasso
    - 1-column fused lasso

## Simulations

### gctree simulated data

1. Run all models
    - choosing random sequence from each cluster
    - imputing ancestors with maximum parsimony
    - using known ancestors from `gctree`
2. Check differences in parameters
3. Compare results to truth

### simulated from survival model

1. Run all models
    - star tree 1-column
    - star tree 4-column
2. Check differences in parameters
3. Compare results to truth

### parsimony survival model

1. Run `mobeef` model(s) on CTMC-generated data (multiple mutations) and parsimony-generated data (one mutation)
2. Compare between
3. Compare to truth

## Real data

## Datasets

- Cui et al (2016)
- Laura/Kate?
- Example dataset from `shazam` package
- website w/ publicly available datasets: [http://b-t.cr/t/publicly-available-airr-seq-data-sets/317](here)

1. Run all models
    - choosing best approach from simulations (random choice vs. imputed ancestors)
2. Check vs. known hot/cold spots
3. Predictive check??? TODO
4. Compare between datasets for conserved/varying motifs

