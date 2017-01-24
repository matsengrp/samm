#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit a context-sensitive motif model via MCMC-EM
"""

import sys
import argparse
import os
import os.path
import csv

from models import ObservedSequenceMutations
from mcmc_em import MCMC_EM
from feature_generator import SubmotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='rng seed for replicability',
        default=1533)
    parser.add_argument('--input_file',
        type=str,
        help='sequence data in csv',
        default='_output/seqs.csv')
    parser.add_argument('--input_genes',
        type=str,
        help='genes data in csv',
        default='_output/genes.csv')
    parser.add_argument('--num_threads',
        type=str,
        help='number of threads to use during E-step',
        default=10)
    parser.add_argument('--motif_len',
        type=str,
        help='length of motif (must be odd)',
        default=5)
    parser.add_argument('--verbose',
        action='store_true',
        help='output R log')

    args = parser.parse_args()

    return args

def read_data(gene_file_name, seq_file_name):
    gene_dict = {}
    with open(gene_file_name, "r") as gene_csv:
        gene_reader = csv.reader(gene_csv, delimiter=',')
        gene_reader.next()
        for row in gene_reader:
            gene_dict[row[0]] = row[1]

    obs_data = []
    with open(seq_file_name, "r") as seq_csv:
        seq_reader = csv.reader(seq_csv, delimiter=",")
        seq_reader.next()
        for row in seq_reader:
            start_seq = gene_dict[row[0]].lower()
            end_seq = row[2]
            obs_data.append(
                ObservedSequenceMutations(
                    start_seq=start_seq[:len(end_seq)],
                    end_seq=end_seq,
                )
            )
    return gene_dict, obs_data

def main(args=sys.argv[1:]):
    args = parse_args()

    assert(args.motif_len % 2 == 1 and args.motif_len > 1)

    print "Reading data"
    gene_dict, obs_data = read_data(args.input_genes, args.input_file)
    feat_generator = SubmotifFeatureGenerator(submotif_len=args.motif_len)

    print "Running EM"
    em_algo = MCMC_EM(
        obs_data,
        feat_generator,
        MutationOrderGibbsSampler,
        num_threads=args.num_threads,
    )
    theta = em_algo.run(
        max_iters=10,
        verbose=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
