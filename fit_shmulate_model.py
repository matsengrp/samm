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
from submotif_feature_generator import SubmotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from read_data import read_shmulate_val
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
    parser.add_argument('--center-median',
        action='store_true',
        help='median center mutability vector?')

    args = parser.parse_args()

    return args

def main(args=sys.argv[1:]):
    MOTIF_LEN = 5

    args = parse_args()
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)

    # Call Rscript
    command = 'Rscript'
    script_file = 'R/fit_shmulate_model.R'

    cmd = [command, script_file, args.input_file, args.input_genes, args.model_pkl.replace(".pkl", "")]
    print "Calling:", " ".join(cmd)
    res = subprocess.call(cmd)

    # Read in the results from the shmulate model-fitter
    feat_gen = SubmotifFeatureGenerator(motif_len=MOTIF_LEN)
    motif_list = feat_gen.motif_list
    # Read target matrix
    target_motif_dict = dict()
    with open(args.model_pkl.replace(".pkl", "_target.csv"), "r") as model_file:
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
            target_motif_dict[motif] = mutate_to_prop

    # Read mutability matrix
    mut_motif_dict = dict()
    with open(args.model_pkl.replace(".pkl", "_mut.csv"), "r") as model_file:
        csv_reader = csv.reader(model_file)
        motifs = csv_reader.next()[1:]
        motif_vals = csv_reader.next()[1:]
        for motif, motif_val in zip(motifs, motif_vals):
            mut_motif_dict[motif.lower()] = motif_val

    # Read substitution matrix
    sub_motif_dict = dict()
    with open(args.model_pkl.replace(".pkl", "_sub.csv"), "r") as model_file:
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
            sub_motif_dict[motif] = mutate_to_prop

    # Reconstruct theta in the right order
    # TODO: How do we compare the edge motifs?? What does shmulate even do with them?
    target_model_array = np.zeros((feat_gen.feature_vec_len, NUM_NUCLEOTIDES))
    mut_model_array = np.zeros((feat_gen.feature_vec_len, 1))
    sub_model_array = np.zeros((feat_gen.feature_vec_len, NUM_NUCLEOTIDES))
    for motif_idx, motif in enumerate(motif_list):
        mut_model_array[motif_idx] = read_shmulate_val(mut_motif_dict[motif])
        log.info("%s:%f" % (motif, mut_model_array[motif_idx]))
        for nuc in NUCLEOTIDES:
            target_model_array[motif_idx, NUCLEOTIDE_DICT[nuc]] = read_shmulate_val(target_motif_dict[motif][nuc])
            sub_model_array[motif_idx, NUCLEOTIDE_DICT[nuc]] = read_shmulate_val(sub_motif_dict[motif][nuc])

    if args.center_median:
        if np.isfinite(np.median(mut_model_array)):
            mut_model_array -= np.median(mut_model_array)

    # keep mut_model_array in same position as mutabilities from fit_context
    pickle.dump((mut_model_array, (target_model_array, sub_model_array)), open(args.model_pkl, 'w'))

if __name__ == "__main__":
    main(sys.argv[1:])
