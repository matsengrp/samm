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
import pickle
import logging as log

import numpy as np
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
        type=str,
        help='length of motif (must be odd)',
        default=5)
    parser.add_argument('--em-max-iters',
        type=int,
        help='number of EM iterations',
        default=20)
    parser.add_argument('--log-file',
        type=str,
        help='log file',
        default='_output/context_log.txt')
    parser.add_argument('--out-file',
        type=str,
        help='file with pickled context model',
        default='_output/context_model.pkl')
    parser.add_argument("--lasso-param",
        type=float,
        help="lasso parameter",
        default=0.1)

    args = parser.parse_args()

    assert(args.motif_len % 2 == 1 and args.motif_len > 1)

    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)
    np.random.seed(args.seed)
    feat_generator = SubmotifFeatureGenerator(submotif_len=args.motif_len)

    log.info("Reading data")
    gene_dict, obs_data = read_gene_seq_csv_data(args.input_genes, args.input_file)
    log.info("Number of sequences %d" % len(obs_data))
    log.info("Settings %s" % args)

    log.info("Running EM")
    em_algo = MCMC_EM(
        obs_data,
        feat_generator,
        MutationOrderGibbsSampler,
        num_threads=args.num_threads,
        burn_in=1,
        base_num_e_samples=2,
    )
    theta = em_algo.run(lasso_param=args.lasso_param, max_em_iters=args.em_max_iters)

    motif_list = SubmotifFeatureGenerator.get_motif_list(args.motif_len)
    log.info("==== FINAL theta ====")
    for i in range(theta.size):
        if np.abs(theta[i]) > ZERO_THRES:
            if i == theta.size - 1:
                log.info("%d: %f (EDGES)" % (i, theta[i]))
            else:
                log.info("%d: %f (%s)" % (i, theta[i], motif_list[i]))

    with open(args.out_file, "w") as f:
        pickle.dump(theta, f)

if __name__ == "__main__":
    main(sys.argv[1:])
