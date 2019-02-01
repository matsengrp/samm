"""
Generate mutated sequences using the survival model on a given tree using
mutation model parameters. This can generate naive sequences using partis or
construct random nucleotide sequences.
NOTE: this does not output the usual files. Only for simulating sequences along a tree.
it will output the new tree with the ancestral states. It does not do anything else.
"""

import pickle
import sys
import argparse
import scipy
import numpy as np
import os
import os.path
import csv
import subprocess

from survival_model_simulator import SurvivalModelSimulatorSingleColumn
from survival_model_simulator import SurvivalModelSimulatorMultiColumn
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from simulate_germline import GermlineSimulatorPartis, GermlineMetadata
from simulate_shm_star_tree import _get_germline_nucleotides, create_simulator
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=1)
    parser.add_argument('--random-gene-len',
        type=int,
        help='Create random naive sequences of this length, only used if not using partis',
        default=24)
    parser.add_argument('--agg-motif-len',
        type=int,
        help='Length of k-mer motifs in the aggregate model -- assumes that the center position mutates',
        default=3)
    parser.add_argument('--input-tree',
        type=str,
        help='Input file with tree',
        default='_output/tree_in.pkl')
    parser.add_argument('--output-tree',
        type=str,
        help='Input file with tree',
        default='_output/tree_out.pkl')
    parser.add_argument('--output-naive',
        type=str,
        help='CSV file to output for naive sequences',
        default='_output/naive.csv')
    parser.add_argument('--input-model',
        type=str,
        help='Input file with true theta parameters',
        default='_output/true_model.pkl')
    parser.add_argument('--use-partis',
        action="store_true",
        help='Use partis to gernerate germline/naive sequences')
    parser.add_argument('--lambda0',
        type=float,
        help='Baseline constant hazard rate in cox proportional hazards model',
        default=0.1)
    parser.add_argument('--n-naive',
        type=int,
        help='Number of naive sequences to create, only used if not using partis',
        default=1)
    parser.add_argument('--n-subjects',
        type=int,
        help='Number of subjects (so number of germline sets) - used by partis',
        default=1)
    parser.add_argument('--with-replacement',
        action="store_true",
        help='Allow same position to mutate multiple times')

    parser.set_defaults(with_replacement=False, )
    args = parser.parse_args()
    return args

def run_survival(args, tree, germline_seq):
    simulator = create_simulator(args)

    tree.add_feature("sequence", germline_seq.lower())
    for node in tree.traverse("preorder"):
        sequence = node.sequence
        for children in node.children:
            num_to_mutate = scipy.random.poisson(children.dist * args.lambda0)
            percent_to_mutate = float(num_to_mutate)/len(sequence)
            full_seq_mutations = simulator.simulate(
                start_seq=sequence,
                percent_mutated=percent_to_mutate,
                with_replacement=args.with_replacement,
            )
            children.add_feature("sequence", full_seq_mutations.end_seq)
    print(tree.get_ascii(attributes=["sequence"], show_internal=True))

def get_random_germline(germline_seqs):
    germline_keys = germline_seqs.keys()
    rand_idx = np.random.choice(
        a=len(germline_keys),
        p=[germline_seqs[g_key].freq for g_key in germline_keys]
    )
    rand_key = germline_keys[rand_idx]
    germline_seq = germline_seqs[rand_key].val
    return germline_seq

def main(args=sys.argv[1:]):
    args = parse_args()
    np.random.seed(args.seed)

    germline_seqs = _get_germline_nucleotides(args)
    rand_germline_seq = get_random_germline(germline_seqs)

    with open(args.input_tree, 'r') as f:
        tree = pickle.load(f)
    run_survival(args, tree, rand_germline_seq)
    with open(args.output_tree, 'wb') as f:
        pickle.dump(tree, f)


if __name__ == "__main__":
    main(sys.argv[1:])
