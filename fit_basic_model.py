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

import numpy as np
import scipy.stats

from models import ObservedSequenceMutations
from mcmc_em import MCMC_EM
from submotif_feature_generator import SubmotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from common import *
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
    parser.add_argument('--prop-file',
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

    args = parser.parse_args()

    assert(args.motif_len % 2 == 1 and args.motif_len > 1)

    return args

def main(args=sys.argv[1:]):
    args = parse_args()

    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)

    np.random.seed(args.seed)
    feat_generator = SubmotifFeatureGenerator(motif_len=args.motif_len)

    if args.use_partis:
        annotations, germlines = get_paths_to_partis_annotations(args.input_partis, chain=args.chain, ig_class=args.igclass)
        gene_dict, obs_data = read_partis_annotations(annotations, inferred_gls=germlines, chain=args.chain, motif_len=args.motif_len)
    else:
        gene_dict, obs_data = read_gene_seq_csv_data(args.input_genes, args.input_file, motif_len=args.motif_len)

    motif_list = feat_generator.get_motif_list()

    mutations = {motif: {nucleotide: 0. for nucleotide in 'acgt'} for motif in motif_list}
    proportions = {motif: {nucleotide: 0. for nucleotide in 'acgt'} for motif in motif_list}
    appearances = {motif: 0. for motif in motif_list}

    for obs_seq in obs_data:
        germline_motifs = feat_generator.create_for_sequence(obs_seq.start_seq)

        for key, value in germline_motifs.iteritems():
            appearances[motif_list[value[0]]] += 1

        for mut_pos, mut_nuc in obs_seq.mutation_pos_dict.iteritems():
            for mutation in germline_motifs[mut_pos]:
                mutations[motif_list[mutation]][mut_nuc] += 1

    for key in motif_list:
        for nucleotide in 'acgt':
            if appearances[key] > 0:
                proportions[key][nucleotide] = 1. * mutations[key][nucleotide] / appearances[key]

    prop_list = np.array([[proportions[motif_list[i]][nucleotide] for nucleotide in 'acgt'] for i in range(len(motif_list))])
    pickle.dump(prop_list, open(args.prop_file, 'w'))

    # Print the motifs with the highest and lowest proportions
    if args.theta_file is not None:
        theta = pickle.load(open(args.theta_file, 'rb'))
        threshold_prop_list = np.zeros(prop_list.shape)
        mean_prop = np.mean(prop_list, axis=0)
        sd_prop = np.sqrt(np.var(prop_list, axis=0))
        for i in range(theta.shape[0]):
            for idx, nucleotide in enumerate('acgt'):
                if np.abs(proportions[motif_list[i]][nucleotide] - mean_prop[idx]) > 0.5 * sd_prop[idx]:
                    log.info("%d: %f, %s, %f" % (i, np.max(theta[i,idx]), motif_list[i], proportions[motif_list[i]][nucleotide]))
                    threshold_prop_list[i][idx] = proportions[motif_list[i]][nucleotide]
    
        theta_flat = np.ravel(theta)
        prop_flat = np.ravel(prop_list)
        thresh_flat = np.ravel(threshold_prop_list)
        log.info("THETA")
        log.info(scipy.stats.spearmanr(theta_flat, prop_flat))
        log.info(scipy.stats.kendalltau(theta_flat, prop_flat))
    
        log.info("THRESHOLDED THETA")
        log.info(scipy.stats.spearmanr(theta_flat, thresh_flat))
        log.info(scipy.stats.kendalltau(theta_flat, thresh_flat))

if __name__ == "__main__":
    main(sys.argv[1:])
