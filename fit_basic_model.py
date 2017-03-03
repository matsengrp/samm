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
from submotif_feature_generator import SubmotifFeatureGenerator
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
    parser.add_argument('--input-file',
        type=str,
        help='sequence data in csv',
        default='_output/seqs.csv')
    parser.add_argument('--input-genes',
        type=str,
        help='genes data in csv',
        default='_output/genes.csv')
    parser.add_argument('--sample-or-impute',
        default=None,
        choices=('sample-random', 'sample-highly-mutated', 'impute-ancestors'),
        help='sample sequence from cluster or impute ancestors?')
    parser.add_argument('--scratch-directory',
        type=str,
        help='where to write dnapars files, if necessary',
        default='_output')
    parser.add_argument('--input-partis',
        type=str,
        help='partis annotations file',
        default=SAMPLE_PARTIS_ANNOTATIONS)
    parser.add_argument('--use-partis',
        action='store_true',
        help='use partis annotations file')
    parser.add_argument('--num-threads',
        type=str,
        help='number of threads to use during E-step',
        default=4)
    parser.add_argument('--motif-len',
        type=int,
        help='length of motif (must be odd)',
        default=5)
    parser.add_argument('--theta-file',
        type=str,
        help='file with pickled true context model (default: None, for no truth)',
        default=None)
    parser.add_argument('--out-file',
        type=str,
        help='file to output fitted proportions',
        default='_output/prop_file.pkl')
    parser.add_argument('--log-file',
        type=str,
        help='file to output logs',
        default='_output/basic_log.txt')
    parser.add_argument('--chain',
        default='h',
        choices=('h', 'k', 'l'),
        help='heavy chain or kappa/lambda light chain')
    parser.add_argument('--igclass',
        default='G',
        choices=('G', 'M', 'K', 'L'),
        help='immunoglobulin class')
    parser.add_argument('--per-target-model',
        action='store_true',
        help='Fit per target model')

    args = parser.parse_args()

    assert(args.motif_len % 2 == 1 and args.motif_len > 1)

    return args

def main(args=sys.argv[1:]):
    args = parse_args()

    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)

    scratch_dir = os.path.join(args.scratch_directory, str(time.time()))
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)

    np.random.seed(args.seed)
    feat_generator = SubmotifFeatureGenerator(motif_len=args.motif_len)

    if args.use_partis:
        annotations, germlines = get_paths_to_partis_annotations(args.input_partis, chain=args.chain, ig_class=args.igclass)
        write_partis_data_from_annotations(args.input_genes, args.input_file, annotations, inferred_gls=germlines, chain=args.chain)

    obs_data = read_gene_seq_csv_data(args.input_genes, args.input_file, motif_len=args.motif_len, sample_or_impute=args.sample_or_impute, scratch_dir=scratch_dir)

    motif_list = feat_generator.get_motif_list()

    mutations = {motif: {nucleotide: 0. for nucleotide in NUCLEOTIDES} for motif in motif_list}
    appearances = {motif: 0. for motif in motif_list}

    for obs_seq in obs_data:
        germline_motifs = feat_generator.create_for_sequence(obs_seq.start_seq, obs_seq.left_flank, obs_seq.right_flank)

        for key, value in germline_motifs.iteritems():
            appearances[motif_list[value]] += 1

        for mut_pos, mut_nuc in obs_seq.mutation_pos_dict.iteritems():
            mutations[motif_list[germline_motifs[mut_pos]]][mut_nuc] += 1

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
        probability_matrix = {motif: {nucleotide: 0. for nucleotide in NUCLEOTIDES} for motif in motif_list}
        for key in motif_list:
            num_mutations = sum(mutations[key].values())
            if num_mutations > 0:
                proportions[key] = 1. * num_mutations / appearances[key]
                for nucleotide in NUCLEOTIDES:
                    probability_matrix[key][nucleotide] = 1. * mutations[key][nucleotide] / num_mutations
        prop_list = np.array([proportions[motif_list[i]] for i in range(len(motif_list))])

    pickle.dump((prop_list, probability_matrix), open(args.out_file, 'w'))

if __name__ == "__main__":
    main(sys.argv[1:])
