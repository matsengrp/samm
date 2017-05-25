import pickle
import sys
import argparse
import itertools
import numpy as np
import os
import os.path
import csv
import re
import random

from survival_model_simulator import SurvivalModelSimulatorSingleColumn
from survival_model_simulator import SurvivalModelSimulatorMultiColumn
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from common import *
from read_data import GERMLINE_PARAM_FILE

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='rng seed for replicability',
        default=1533)
    parser.add_argument('--random-gene-len',
        type=int,
        help='Create random germline genes of this length. If zero, load true germline genes',
        default=24)
    parser.add_argument('--agg-motif-len',
        type=int,
        help='length of motif -- assume center mutates',
        default=5)
    parser.add_argument('--input-model',
        type=str,
        default='_output/true_model.pkl',
        help='true model pickle file')
    parser.add_argument('--germline-path',
        type=str,
        help='germline file path',
        default=GERMLINE_PARAM_FILE)
    parser.add_argument('--output-file',
        type=str,
        help='simulated data destination file',
        default='_output/seqs.csv')
    parser.add_argument('--output-genes',
        type=str,
        help='germline genes used in csv file',
        default='_output/genes.csv')
    parser.add_argument('--lambda0',
        type=float,
        help='base hazard rate in cox proportional hazards model for a single motif (summing over targets)',
        default=0.1)
    parser.add_argument('--n-taxa',
        type=int,
        help='number of taxa to simulate',
        default=1)
    parser.add_argument('--n-germlines',
        type=int,
        help='number of germline genes to sample from (max 350)',
        default=2)
    parser.add_argument('--min-censor-time',
        type=float,
        help='Minimum censoring time',
        default=1)
    parser.add_argument('--min-percent-mutated',
        type=float,
        help='Minimum percent of sequence to mutate',
        default=0.05)
    parser.add_argument('--with-replacement',
        action="store_true",
        help='Allow same position to mutate multiple times')

    parser.set_defaults(with_replacement=False)
    args = parser.parse_args()

    return args

def _get_germline_nucleotides(args, nonzero_motifs=[]):
    if args.random_gene_len > 0:
        germline_genes = ["FAKE-GENE-%d" % i for i in range(args.n_germlines)]
        germline_nucleotides = [get_random_dna_seq(args.random_gene_len) for i in range(args.n_germlines)]
    else:
        # Read parameters from file
        params = read_germline_file(args.germline_path)

        # Select, with replacement, args.n_germlines germline genes from our
        # parameter file and place them into a numpy array.
        germline_genes = np.random.choice(params.index.values, size=args.n_germlines)

        # Put the nucleotide content of each selected germline gene into a
        # corresponding list.
        germline_nucleotides = [row[gene] for gene in germline_genes]

    return germline_nucleotides, germline_genes

def dump_germline_data(germline_nucleotides, germline_genes, args):
    # Write germline genes to file with two columns: name of gene and
    # corresponding sequence.
    with open(args.output_genes, 'w') as outgermlines:
        germline_file = csv.writer(outgermlines)
        germline_file.writerow(['germline_name','germline_sequence'])
        for gene, sequence in zip(germline_genes, germline_nucleotides):
            germline_file.writerow([gene,sequence])

def main(args=sys.argv[1:]):
    args = parse_args()

    # Randomly generate number of mutations or use default
    np.random.seed(args.seed)

    feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=[args.agg_motif_len],
        left_motif_flank_len_list=[[args.agg_motif_len/2]],
    )
    with open(args.input_model, 'r') as f:
        agg_theta, raw_theta = pickle.load(f)

    germline_nucleotides, germline_genes = _get_germline_nucleotides(args)

    if agg_theta.shape[1] == NUM_NUCLEOTIDES + 1:
        simulator = SurvivalModelSimulatorMultiColumn(agg_theta, feat_generator, lambda0=args.lambda0)
    else:
        probability_matrix = np.ones((agg_theta.size, NUM_NUCLEOTIDES)) * 1.0/3
        simulator = SurvivalModelSimulatorSingleColumn(agg_theta, probability_matrix, feat_generator, lambda0=args.lambda0)

    dump_germline_data(germline_nucleotides, germline_genes, args)

    # For each germline gene, run survival model to obtain mutated sequences.
    # Write sequences to file with three columns: name of germline gene
    # used, name of simulated sequence and corresponding sequence.
    with open(args.output_file, 'w') as outseqs:
        seq_file = csv.writer(outseqs)
        seq_file.writerow(['germline_name','sequence_name','sequence'])
        for run, (gene, sequence) in \
                enumerate(zip(germline_genes, germline_nucleotides)):

            full_data_samples = [
                simulator.simulate(
                    start_seq=sequence.lower(),
                    # censoring_time=args.min_censor_time + np.random.rand() * 0.25, # allow some variation in censor time
                    percent_mutated=args.min_percent_mutated + np.random.rand() * 0.1, # allow some variation in censor time
                    with_replacement=args.with_replacement,
                ) for i in range(args.n_taxa)
            ]

            # write to file in csv format
            num_mutations = []
            for i, sample in enumerate(full_data_samples):
                num_mutations.append(len(sample.mutations))
                seq_file.writerow([gene, "%s-sample-%d" % (gene, i) , sample.left_flank + sample.end_seq + sample.right_flank])
            print "Number of mutations: %f (%f)" % (np.mean(num_mutations), np.sqrt(np.var(num_mutations)))

if __name__ == "__main__":
    main(sys.argv[1:])
