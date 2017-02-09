#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    simulate sequences given germline sequences shmulate

'''

from __future__ import print_function

import subprocess
import sys
import argparse
import pandas as pd
import numpy as np
import os
import os.path
import csv
import pickle

from common import *
from Bio import SeqIO
from feature_generator import SubmotifFeatureGenerator

sys.path.append('gctree/bin')
from gctree import MutationModel, CollapsedTree

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    subparsers = parser.add_subparsers()

    ###
    # for simulating

    parser_simulate = subparsers.add_parser('simulate')

    parser_simulate.add_argument('--n-taxa',
        type=int,
        help='number of taxa to simulate',
        default=2)
    parser_simulate.add_argument('--param-path',
        type=str,
        help='parameter file path',
        default=GERMLINE_PARAM_FILE)
    parser_simulate.add_argument('--seed',
        type=int,
        help='rng seed for replicability',
        default=1533)
    parser_simulate.add_argument('--output-file',
        type=str,
        help='simulated data destination file',
        default='_output/seqs.csv')
    parser_simulate.add_argument('--output-genes',
        type=str,
        help='germline genes used in csv file',
        default='_output/genes.csv')
    parser_simulate.add_argument('--output-true-theta',
        type=str,
        help='true theta pickle file',
        default='_output/true_theta.pkl')
    parser_simulate.add_argument('--log-dir',
        type=str,
        help='log directory',
        default='_output')
    parser_simulate.add_argument('--n-germlines',
        type=int,
        help='number of germline genes to sample (maximum 350)',
        default=2)
    parser.add_argument('--motif-len',
        type=int,
        help='length of motif (must be odd)',
        default=5)
    parser_simulate.add_argument('--verbose',
        action='store_true',
        help='output R log')
    parser_simulate.add_argument('--mutability',
        type=str,
        default='gctree/S5F/Mutability.csv',
        help='path to mutability model file')
    parser_simulate.add_argument('--substitution',
        type=str,
        default='gctree/S5F/Substitution.csv',
        help='path to substitution model file')
    parser_simulate.add_argument('--lambda0',
        type=float,
        default=None,
        help='baseline mutation rate')
    parser_simulate.add_argument('--r',
        type=float,
        default=1.,
        help='sampling probability')
    parser_simulate.add_argument('--T',
        type=int,
        default=None,
        help='observation time, if None we run until termination and take all leaves')
    parser_simulate.add_argument('--frame',
        type=int,
        default=None,
        help='codon frame')
    parser_simulate.set_defaults(func=simulate)

    parser_simulate.set_defaults(subcommand=simulate)

    ###
    # for parsing

    args = parser.parse_args()

    return args


def run_gctree(args, germline_seq):
    ''' somewhat cannibalized gctree simulation '''

    if args.lambda0 is None:
        args.lambda0 = max([1, int(.01*len(germline_seq))])
    mutation_model = MutationModel(args.mutability, args.substitution)
    trials = 1000
    # this loop makes us resimulate if size too small, or backmutation
    for trial in range(trials):
        try:
            tree = mutation_model.simulate(germline_seq,
                                           lambda0=args.lambda0,
                                           r=args.r,
                                           N=args.n_taxa,
                                           T=args.T)
            collapsed_tree = CollapsedTree(tree=tree, frame=args.frame) # <-- this will fail if backmutations
            break
        except RuntimeError as e:
            print('{}, trying again'.format(e))
        else:
            raise

    return tree


def simulate(args):
    ''' simulate submodule '''

    # write empty sequence file before appending
    output_dir, _ = os.path.split(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read parameters from file
    params = read_bcr_hd5(args.param_path)

    # Find genes with "N" and remove them so gctree is happy
    genes_to_sample = params['gene'][[base in 'ACGT' for base in params['base']]].unique()

    # Randomly generate number of mutations or use default
    np.random.seed(args.seed)

    # Select, with replacement, args.n_germlines germline genes from our
    # parameter file and place them into a numpy array.
    germline_genes = np.random.choice(genes_to_sample,
            size=args.n_germlines)

    # Put the nucleotide content of each selected germline gene into a
    # corresponding list.
    germline_nucleotides = [''.join(list(params[params['gene'] == gene]['base'])) \
            for gene in germline_genes]

    # Write germline genes to file with two columns: name of gene and
    # corresponding sequence.
    with open(args.output_genes, 'w') as outgermlines:
        germline_file = csv.writer(outgermlines)
        germline_file.writerow(['germline_name','germline_sequence'])
        for gene, sequence in zip(germline_genes, germline_nucleotides):
            germline_file.writerow([gene,sequence])

    # For each germline gene, run shmulate to obtain mutated sequences.
    # Write sequences to file with three columns: name of germline gene
    # used, name of simulated sequence and corresponding sequence.
    with open(args.output_file, 'w') as outseqs:
        seq_file = csv.writer(outseqs)
        seq_file.writerow(['germline_name','sequence_name','sequence'])
        for run, (gene, sequence) in \
                enumerate(zip(germline_genes, germline_nucleotides)):
            # Creates a file with a single run of simulated sequences.
            # The seed is modified so we aren't generating the same
            # mutations on each run
            tree = run_gctree(args, sequence)
            i = 0
            for leaf in tree.iter_leaves():
                if leaf.frequency != 0:
                    i += 1
                    seq_file.writerow([gene, 'Run{0}-Sequence{1}'.format(run, i), str(leaf.sequence).lower()])

    # Dump the true "thetas," which are mutability * substitution
    feat_generator = SubmotifFeatureGenerator(motif_len=args.motif_len)
    motif_list = feat_generator.get_motif_list()
    context_model = MutationModel(args.mutability, args.substitution).context_model
    true_theta = np.empty((feat_generator.feature_vec_len, NUM_NUCLEOTIDES))
    true_theta.fill(-np.inf)
    for motif_idx, motif in enumerate(motif_list):
        mutability = args.lambda0 * context_model[motif.upper()][0]
        for nuc in NUCLEOTIDES:
            substitution = context_model[motif.upper()][1][nuc.upper()]
            if mutability > 0 and substitution > 0:
                true_theta[motif_idx, NUCLEOTIDE_DICT[nuc]] = np.log(mutability) + np.log(substitution)
    pickle.dump(true_theta, open(args.output_true_theta, 'w'))


def main(args=sys.argv[1:]):
    ''' run program '''

    args = parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    main(sys.argv[1:])
