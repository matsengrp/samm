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
from Bio import SeqIO
import csv
from common import *

sys.path.append('gctree/bin')
from gctree import *

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
    parser_simulate.add_argument('--log-dir',
        type=str,
        help='log directory',
        default='_output')
    parser_simulate.add_argument('--n-germlines',
        type=int,
        help='number of germline genes to sample (maximum 350)',
        default=2)
    parser_simulate.add_argument('--n-mutes',
        type=int,
        help='number of mutations from germline (default: -1 meaning choose at random)',
        default=-1)
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
    parser_simulate.add_argument('--p',
        type=float,
        default=.49,
        help='branching probability')
    parser_simulate.add_argument('--lambda0',
        type=float,
        default=None,
        help='baseline mutation rate')
    parser_simulate.add_argument('--r',
        type=float,
        default=1.,
        help='sampling probability')
    parser_simulate.add_argument('--n',
        type=int,
        default=1,
        help='minimum simulation size')
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
    size = 0
    while size < args.n_taxa:
        # this loop makes us resimulate if we got backmutations
        trial = 1
        while trial < 10:
            try:
                tree = mutation_model.simulate(germline_seq, p=args.p, lambda0=args.lambda0, r=args.r, frame=args.frame, T=args.T)
                collapsed_tree = CollapsedTree(tree=tree, frame=args.frame) # <-- this will fail if backmutations
                break
            except RuntimeError:
                trial += 1
                continue
            else:
                raise
        if trial == 10:
            raise RuntimeError('repeated sequences in collapsed tree on {} attempts'.format(trial))
        size = sum(node.frequency for node in tree)

    return tree


def simulate(args):
    ''' simulate submodule '''

    # write empty sequence file before appending
    output_dir, _ = os.path.split(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read parameters from file
    params = read_bcr_hd5(args.param_path)

    # Randomly generate number of mutations or use default
    np.random.seed(args.seed)

    if args.n_mutes < 0:
        n_mute_vec = 10 + np.random.randint(20, size=args.n_germlines)
    else:
        n_mute_vec = [args.n_mutes] * args.n_germlines

    # Select, with replacement, args.n_germlines germline genes from our
    # parameter file and place them into a numpy array.
    # Here 'germline_gene' is something like IGHV1-2*01.
    germline_genes = np.random.choice(params['gene'].unique(),
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
        for run, (gene, sequence, n_mutes) in \
                enumerate(zip(germline_genes, germline_nucleotides, n_mute_vec)):
            # Creates a file with a single run of simulated sequences.
            # The seed is modified so we aren't generating the same
            # mutations on each run
            tree = run_gctree(args, sequence)
            i = 0
            for leaf in tree.iter_leaves():
                if leaf.frequency != 0:# and '*' not in Seq(leaf.sequence, generic_dna).translate():
                    i += 1
                    seq_file.writerow([gene, 'Run{0}-Sequence{1}'.format(run, i), str(leaf.sequence)])


def main(args=sys.argv[1:]):
    ''' run program '''

    args = parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    main(sys.argv[1:])
