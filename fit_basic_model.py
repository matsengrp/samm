#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit a basic, order-free model by counting
"""

import sys
import argparse
import os
import os.path
import csv
import pickle
import logging as log
import time

import numpy as np
import scipy.stats

from models import ObservedSequenceMutations
from mcmc_em import MCMC_EM
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from common import *
from read_data import read_gene_seq_csv_data
from read_data import SAMPLE_PARTIS_ANNOTATIONS
from matsen_grp_data import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='rng seed for replicability',
        default=1533)
    parser.add_argument('--input-genes',
        type=str,
        help='genes data in csv',
        default='_output/genes.csv')
    parser.add_argument('--input-seqs',
        type=str,
        help='sequence data in csv',
        default='_output/seqs.csv')
    parser.add_argument('--sample-regime',
        type=int,
        default=1,
        choices=(1, 2, 3),
        help='1: take all sequences; 2: sample random sequence from cluster; 3: choose most highly mutated sequence (default: 1)')
    parser.add_argument('--scratch-directory',
        type=str,
        help='where to write dnapars files, if necessary',
        default='_output')
    parser.add_argument('--num-threads',
        type=str,
        help='number of threads to use during E-step',
        default=4)
    parser.add_argument('--motif-len',
        type=int,
        help='length of motif (must be odd)',
        default=5)
    parser.add_argument('--mutating-positions',
        type=str,
        help='which position in the motif is mutating; can be one of combination of -1, 0, 1 for 5\'/left end, central, or 3\'/right end',
        default='0')
    parser.add_argument('--theta-file',
        type=str,
        help='file with pickled true context model (default: None, for no truth)',
        default=None)
    parser.add_argument('--out-file',
        type=str,
        help='file to output fitted proportions',
        default='_output/basic_file.pkl')
    parser.add_argument('--log-file',
        type=str,
        help='file to output logs',
        default='_output/basic_log.txt')
    parser.add_argument('--per-target-model',
        action='store_true',
        help='Fit per target model')

    args = parser.parse_args()

    assert(args.motif_len % 2 == 1 and args.motif_len > 1)

    return args

def main(args=sys.argv[1:]):
    args = parse_args()

    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)

    np.random.seed(args.seed)
    feat_generator = HierarchicalMotifFeatureGenerator(motif_lens=[args.motif_len])

    obs_data, metadata = read_gene_seq_csv_data(
            args.input_genes,
            args.input_seqs,
            motif_len=args.motif_len,
            mutating_positions=[args.mutating_positions],
            sample=args.sample_regime
        )

    motif_list = feat_generator.motif_list

    mutations = {motif: {nucleotide: 0. for nucleotide in NUCLEOTIDES} for motif in motif_list}
    appearances = {motif: 0. for motif in motif_list}

    for obs_seq in obs_data:
        germline_motifs = feat_generator.create_for_sequence(obs_seq.start_seq, obs_seq.left_flank, obs_seq.right_flank)

        for key, value in germline_motifs.iteritems():
            appearances[motif_list[value[0]]] += 1

        for mut_pos, mut_nuc in obs_seq.mutation_pos_dict.iteritems():
            feat_idx = (germline_motifs[mut_pos])[0]
            mutations[motif_list[feat_idx]][mut_nuc] += 1

    # Print number of times the motifs appear in the starting sequence
    motif_appear_counts = [(m, k) for m, k in appearances.iteritems()]
    motif_appear_counts_sorted = sorted(motif_appear_counts, key=lambda x: x[1], reverse=True)
    log.info("Appearances in the starting sequence")
    for m,k in motif_appear_counts_sorted:
        log.info("%s: %d" % (m, k))

    if args.per_target_model:
        proportions = {motif: {nucleotide: 0. for nucleotide in NUCLEOTIDES} for motif in motif_list}
        probability_matrix = None
        for key in motif_list:
            for nucleotide in NUCLEOTIDES:
                if appearances[key] > 0:
                    proportions[key][nucleotide] = 1. * mutations[key][nucleotide] / appearances[key]
        prop_list = np.array([[proportions[motif_list[i]][nucleotide] for nucleotide in NUCLEOTIDES] for i in range(len(motif_list))])
    else:
        proportions = {motif: 0 for motif in motif_list}
        probability_dict = {motif: {nucleotide: 0. for nucleotide in NUCLEOTIDES} for motif in motif_list}
        for key in motif_list:
            num_mutations = sum(mutations[key].values())
            # if num_mutations > 0:
            if num_mutations > 0 and appearances[key] > 150:
                proportions[key] = 1. * num_mutations / appearances[key]
                for nucleotide in NUCLEOTIDES:
                    probability_dict[key][nucleotide] = 1. * mutations[key][nucleotide] / num_mutations
        prop_list = np.array([proportions[motif_list[i]] for i in range(len(motif_list))])
        probability_matrix = np.array([[probability_dict[motif_list[i]][nucleotide] for nucleotide in NUCLEOTIDES] for i in range(len(motif_list))])

        # Print theta values
        log.info("Fitted theta values")
        motif_theta_vals = [(motif_list[i], proportions[motif_list[i]]) for i in range(len(motif_list))]
        for m, t in sorted(motif_theta_vals, key=lambda x: x[1]):
            log.info("%s: %f" % (m, t))

    with open(args.out_file, 'w') as pickle_file:
        pickle.dump((prop_list, probability_matrix), pickle_file)

if __name__ == "__main__":
    main(sys.argv[1:])
