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

import numpy as np
from scipy.stats import spearmanr

from models import ObservedSequenceMutations
from mcmc_em import MCMC_EM
from feature_generator import SubmotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from common import *

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
        help='file with pickled context model',
        default='_output/context_model.pkl')
    parser.add_argument('--prop-file',
        type=str,
        help='file to output fitted proportions',
        default='_output/prop_file.pkl')

    args = parser.parse_args()

    assert(args.motif_len % 2 == 1 and args.motif_len > 1)

    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    np.random.seed(args.seed)
    feat_generator = SubmotifFeatureGenerator(submotif_len=args.motif_len)

    gene_dict, obs_data = read_gene_seq_csv_data(args.input_genes, args.input_file)

    motif_list = feat_generator.get_motif_list()
    motif_list.append('EDGES')

    mutations = dict.fromkeys(motif_list, 0)
    appearances = dict.fromkeys(motif_list, 0)

    for obs_seq in obs_data:
        mutated_positions = obs_seq.mutation_pos_dict.keys()
        germline_motifs = feat_generator.create_for_sequence(obs_seq.start_seq)

        for key, value in germline_motifs.iteritems():
            appearances[motif_list[value[0]]] += 1

        for mut_pos in mutated_positions:
            for mutation in germline_motifs[mut_pos]:
                mutations[motif_list[mutation]] += 1

    proportions = {}
    for key in motif_list:
        if appearances[key] > 0:
            proportions[key] = 1. * mutations[key] / appearances[key]
        else:
            proportions[key] = 0.

    theta = pickle.load(open(args.theta_file, 'rb'))
    prop_list = [proportions[motif_list[i]] for i in range(theta.size)]

    # Print the motifs that are acting out of the ordinary
    mean_prop = np.mean(prop_list)
    sd_prop = np.sqrt(np.var(prop_list))
    for i in range(theta.size):
        if np.abs(proportions[motif_list[i]] - mean_prop) > 1.5 * sd_prop:
            print (i, theta[i], motif_list[i], proportions[motif_list[i]])

    print spearmanr(theta, prop_list)
    pickle.dump(np.array(prop_list), open(args.prop_file, 'w'))

if __name__ == "__main__":
    main(sys.argv[1:])
