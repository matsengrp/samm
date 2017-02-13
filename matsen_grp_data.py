"""
Modules for processing real data

I'm not sure about the privacy of some of these data sets so I'm putting
all of this in a separate file.
"""

from itertools import product

LAURA_DATA_PATH = '/fh/fast/matsen_e/processed-data/partis/laura-mb-2016-12-22/v8'
KATE_DATA_PATH = '/fh/fast/matsen_e/processed-data/partis/laura-mb-2016-12-22/v8'

def get_atreca_seeded_data(pth, chain='h', ig_class='G'):
    """
    @param pth: prefix path to partis-processed data
    @param chain: 'h', 'k' or 'l'
    @param ig_class: 'G' or 'M' for heavy chain, o/w computed as chain.upper()

    @returns: two lists, one of paths to annotations files and one of corresponding paths
    to inferred germlines

    Notes:
    Data in LAURA_DATA_PATH and KATE_DATA_PATH are in similar forms, as are other data sets
    obtained via Atreca. Typically we have
    
    BF520.{0}-ig{1}/Hs-LN-{2}-5RACE-Ig{3}/

    where

    {0}: seed_id from 1--10 if chain is 'h' o/w in {3,10} if chain is 'l' o/w in 1--10 \ {3,10}
    {1}: chain in {'h', 'k', 'l'}
    {2}: run_id in {'C', 'D'}
    {3}: class in {'G', 'M'} if chain is 'h' o/w equals chain.upper()
    """

    seeds = range(1,11)
    runs = ['C', 'D']
    if chain=='k':
        ig_class = chain.upper()
        seeds = [1,2,4,5,6,7,8,9]
    elif chain=='l':
        ig_class = chain.upper()
        seeds = [3,10]

    file_path = pth + '/seeds/BF520.{0}-ig{1}/Hs-LN-{2}-5RACE-Ig{3}/partition-cluster-annotations.csv'
    gl_path = pth + '/Hs-LN-{0}-5RACE-Ig{1}/hmm/germline-sets'

    annotations_paths = []
    germline_paths = []
    for seed, run in product(seeds, runs):
        annotations_paths.append(file_path.format(seed, chain, run, ig_class))
        germline_paths.append(gl_path.format(run, ig_class, chain))

    return annotations_paths, germline_paths

