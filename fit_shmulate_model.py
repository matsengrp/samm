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
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='rng seed for replicability',
        default=1533)
    parser.add_argument('--input-seqs',
        type=str,
        help='sequence data in csv',
        default='_output/seqs.csv')
    parser.add_argument('--input-genes',
        type=str,
        help='genes data in csv',
        default='_output/genes.csv')
    parser.add_argument('--out-file',
        type=str,
        help='file to output fitted proportions (will also output csv)',
        default='_output/theta_shmulate.pkl')
    parser.add_argument('--intermediate-out-file',
        type=str,
        help='intermediate file with s5f model',
        default='_output/theta_intermed.pkl')
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

    cmd = [command, script_file, args.input_seqs, args.input_genes, args.out_file.replace(".pkl", "")]
    print "Calling:", " ".join(cmd)
    res = subprocess.call(cmd)

    # Read in the results from the shmulate model-fitter
    feat_gen = SubmotifFeatureGenerator(motif_len=MOTIF_LEN)
    motif_list = feat_gen.motif_list
    # Read target matrix
    target_motif_dict = dict()
    with open(args.out_file.replace(".pkl", "_target.csv"), "r") as model_file:
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
    with open(args.out_file.replace(".pkl", "_mut.csv"), "r") as model_file:
        csv_reader = csv.reader(model_file)
        motifs = csv_reader.next()[1:]
        motif_vals = csv_reader.next()[1:]
        for motif, motif_val in zip(motifs, motif_vals):
            mut_motif_dict[motif.lower()] = motif_val

    # Read substitution matrix
    sub_motif_dict = dict()
    with open(args.out_file.replace(".pkl", "_sub.csv"), "r") as model_file:
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
        mut_model_array[motif_idx] = _read_shmulate_val(mut_motif_dict[motif])
        for nuc in NUCLEOTIDES:
            target_model_array[motif_idx, NUCLEOTIDE_DICT[nuc]] = _read_shmulate_val(target_motif_dict[motif][nuc])
            sub_model_array[motif_idx, NUCLEOTIDE_DICT[nuc]] = _read_shmulate_val(sub_motif_dict[motif][nuc])

    with open(args.out_file, 'w') as shmulate_file:
        pickle.dump((mut_model_array, (target_model_array, sub_model_array)), shmulate_file)

    # temporary hack so scons is happy
    with open(args.intermediate_out_file, 'w') as shmulate_file:
        pickle.dump((mut_model_array, (target_model_array, sub_model_array)), shmulate_file)

def _read_shmulate_val(shmulate_value):
    return -np.inf if shmulate_value == "NA" else np.log(float(shmulate_value))

if __name__ == "__main__":
    main(sys.argv[1:])
