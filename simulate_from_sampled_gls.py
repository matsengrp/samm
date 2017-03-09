#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    simulate sequences given germline sequences shmulate

'''

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
from read_data import *
from Bio import SeqIO
from submotif_feature_generator import SubmotifFeatureGenerator

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
    parser_simulate.add_argument('--output-seqs',
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
    parser_simulate.add_argument('--output-per-branch-genes',
        type=str,
        default=None,
        help='additionally output genes from single branches with intermediate ancestors instead of leaves from germline')
    parser_simulate.add_argument('--output-per-branch-seqs',
        type=str,
        default=None,
        help='additionally output genes from single branches with intermediate ancestors instead of leaves from germline')
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
            if args.verbose:
                print('{}, trying again'.format(e))
        else:
            raise

    return tree


def simulate(args):
    ''' simulate submodule '''

    # write empty sequence file before appending
    output_dir, _ = os.path.split(args.output_seqs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read parameters from file
    params = read_germline_file(args.param_path)

    # Find genes with "N" and remove them so gctree is happy
    genes_to_sample = [idx for idx, row in params.iterrows() if set(row['base'].lower()) == NUCLEOTIDE_SET]

    # Randomly generate number of mutations or use default
    np.random.seed(args.seed)

    # Select, with replacement, args.n_germlines germline genes from our
    # parameter file and place them into a numpy array.
    germline_genes = np.random.choice(genes_to_sample, size=args.n_germlines)

    # Put the nucleotide content of each selected germline gene into a
    # corresponding list.
    germline_nucleotides = [params.loc[gene]['base'] for gene in germline_genes]

    # Write germline genes to file with two columns: name of gene and
    # corresponding sequence.
    # For each germline gene, run shmulate to obtain mutated sequences.
    # Write sequences to file with three columns: name of germline gene
    # used, name of simulated sequence and corresponding sequence.
    with open(args.output_seqs, 'w') as outseqs, open(args.output_genes, 'w') as outgermlines, \
         open(args.output_per_branch_seqs, 'w') as outseqswithanc, \
         open(args.output_per_branch_genes, 'w') as outgermlineswithanc:
        gl_file = csv.writer(outgermlines)
        gl_file.writerow(['germline_name','germline_sequence'])
        gl_anc_file = csv.writer(outgermlineswithanc)
        gl_anc_file.writerow(['germline_name','germline_sequence'])
        seq_file = csv.writer(outseqs)
        seq_file.writerow(['germline_name','sequence_name','sequence'])
        seq_anc_file = csv.writer(outseqswithanc)
        seq_anc_file.writerow(['germline_name','sequence_name','sequence'])
        for run, (gene, sequence) in \
                enumerate(zip(germline_genes, germline_nucleotides)):
            # Creates a file with a single run of simulated sequences.
            # The seed is modified so we aren't generating the same
            # mutations on each run
            gl_file.writerow([gene,sequence])
            tree = run_gctree(args, sequence)
            for idx, descendant in enumerate(tree.traverse('preorder')):
                # internal nodes will have frequency zero, so for providing output
                # along a branch we need to consider these cases! otherwise the leaves
                # should have nonzero frequency
                seq_name = 'Run{0}-Sequence{1}'.format(run, idx)
                if descendant.is_root():
                    descendant.name = gene
                    gl_anc_file.writerow([descendant.name,descendant.sequence.lower()])
                else:
                    descendant.name = '-'.join([descendant.up.name, seq_name])
                    gl_anc_file.writerow([descendant.up.name,descendant.up.sequence.lower()])
                    if cmp(descendant.sequence.lower(), descendant.up.sequence.lower()) != 0:
                        seq_anc_file.writerow([descendant.up.name, seq_name, descendant.sequence.lower()])
                    if descendant.frequency != 0 and descendant.is_leaf() and cmp(descendant.sequence.lower(), sequence) != 0:
                        seq_file.writerow([gene, seq_name, descendant.sequence.lower()])

    # Dump the true "thetas," which are mutability * substitution
    feat_generator = SubmotifFeatureGenerator(motif_len=args.motif_len)
    motif_list = feat_generator.get_motif_list()
    context_model = MutationModel(args.mutability, args.substitution).context_model
    true_thetas = np.empty((feat_generator.feature_vec_len, 1))
    probability_matrix = np.empty((feat_generator.feature_vec_len, NUM_NUCLEOTIDES))
    true_thetas.fill(-np.inf)
    probability_matrix.fill(0.)
    for motif_idx, motif in enumerate(motif_list):
        mutability = args.lambda0 * context_model[motif.upper()][0]
        for nuc in NUCLEOTIDES:
            substitution = context_model[motif.upper()][1][nuc.upper()]
            if mutability > 0 and substitution > 0:
                true_thetas[motif_idx] = np.log(mutability)
                probability_matrix[motif_idx, NUCLEOTIDE_DICT[nuc]] = substitution
    pickle.dump([true_thetas, probability_matrix], open(args.output_true_theta, 'w'))


def main(args=sys.argv[1:]):
    ''' run program '''

    args = parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    main(sys.argv[1:])
