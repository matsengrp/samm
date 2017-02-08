#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit model using shmulate
"""
import subprocess

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
    parser.add_argument('--theta-file',
        type=str,
        help='file with pickled context model',
        default='_output/true_theta.pkl')
    parser.add_argument('--model-pkl',
        type=str,
        help='file to output fitted proportions (will also output csv)',
        default='_output/theta_shmulate.pkl')
    parser.add_argument('--log-file',
        type=str,
        help='file to output logs',
        default='_output/shmulate_log.txt')

    args = parser.parse_args()

    return args

def main(args=sys.argv[1:]):
    MOTIF_LEN = 5

    args = parse_args()
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)

    # Call Rscript
    command = 'Rscript'
    script_file = 'R/fit_shmulate_model.R'

    cmd = [command, script_file, args.input_file, args.input_genes, args.model_pkl.replace(".pkl", ".csv")]
    print "Calling:", " ".join(cmd)
    res = subprocess.call(cmd)

    # Read in the results from the shmulate model-fitter
    feat_gen = SubmotifFeatureGenerator(motif_len=MOTIF_LEN)
    motif_list = feat_gen.get_motif_list()
    motif_dict = dict()
    with open(args.model_pkl.replace(".pkl", ".csv"), "r") as model_file:
        csv_reader = csv.reader(model_file)
        # Assume header is ACGT
        header = csv_reader.next()
        for i in range(NUM_NUCLEOTIDES):
            header[i + 1] = header[i + 1].lower()

        for line in csv_reader:
            motif = line[0].lower()
            mutate_to_prop = {}
            for i in range(NUM_NUCLEOTIDES):
                mutate_to_prop[header[i + 1]] = line[i + 1]
            motif_dict[motif] = mutate_to_prop

    # Reconstruct theta in the right order
    # TODO: How do we compare the edge motifs?? What does shmulate even do with them?
    model_array = np.zeros((feat_gen.feature_vec_len, NUM_NUCLEOTIDES))
    for motif_idx, motif in enumerate(motif_list):
        for nuc in NUCLEOTIDES:
            val = motif_dict[motif][nuc]
            if val == "NA":
                val = -np.inf
            model_array[motif_idx, NUCLEOTIDE_DICT[nuc]] = val
    pickle.dump(model_array, open(args.model_pkl, 'w'))

    # Let's compare the true vs. fitted models
    true_theta = pickle.load(open(args.theta_file, 'rb'))
    theta_mask = get_possible_motifs_to_targets(motif_list, true_theta.shape)

    theta_shape = (theta_mask.sum(), 1)
    flat_model = model_array[theta_mask].reshape(theta_shape)
    # Convert the true model (parameterized in theta) to the the same scale as shmulate
    flat_true_model = np.exp(true_theta[theta_mask].reshape(theta_shape))

    log.info("THETA")
    log.info(scipy.stats.spearmanr(flat_model, flat_true_model))
    log.info(scipy.stats.kendalltau(flat_model, flat_true_model))

    log.info("THRESHOLDED THETA")
    log.info(scipy.stats.spearmanr(flat_model, flat_true_model))
    log.info(scipy.stats.kendalltau(flat_model, flat_true_model))

if __name__ == "__main__":
    main(sys.argv[1:])
