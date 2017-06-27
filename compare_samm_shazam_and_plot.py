#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare multiple samm and shazam fits
"""

import pickle
import numpy as np
import os
import csv
import sys
import argparse
import time

from multiprocessing import Pool
from common import *
from read_data import *
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from submotif_feature_generator import SubmotifFeatureGenerator
from likelihood_evaluator import LikelihoodComparer
from fit_shmulate_model import _read_shmulate_val

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input-gene-files',
        type=str,
        help='comma-separated gene files')
    parser.add_argument('--input-seq-files',
        type=str,
        help='comma-separated seq files')
    parser.add_argument('--in-shazam-mut',
        type=str,
        help='comma separated shazam mutability csv files')
    parser.add_argument('--in-shazam-sub',
        type=str,
        help='comma separated shazam substitution csv files',
        default=None)
    parser.add_argument('--in-samm',
        type=str,
        help='comma separated samm csv files')
    parser.add_argument('--in-samm-same-target',
        type=str,
        help='comma separated samm csv files',
        default=None)
    parser.add_argument('--log-file',
        type=str,
        help='log file',
        default=None)
    parser.add_argument('--out-thetas',
        type=str,
        help='output csv for avg thetas',
        default=None)
    parser.add_argument('--out-log-lik-file',
        type=str,
        help='output pdf log likelihood plot')
    parser.add_argument('--scratch-directory',
        type=str,
        help='where to write gibbs workers and dnapars files, if necessary',
        default='_output')
    parser.add_argument('--motif-lens',
        type=str,
        help='length of motif (must be odd)',
        default='5')
    parser.add_argument('--max-motif-len',
        type=str,
        help='full motif length',
        default=None)
    parser.add_argument('--max-position-mutating',
        type=str,
        help='full motif position mutating',
        default=None)
    parser.add_argument('--positions-mutating',
        type=str,
        help="""
        length of left motif flank determining which position is mutating; comma-separated within
        a motif length, colon-separated between, e.g., --motif-lens 3,5 --left-flank-lens 0,1:0,1,2 will
        be a 3mer with first and second mutating position and 5mer with first, second and third
        """,
        default=None)
    parser.add_argument('--num-val-burnin',
        type=int,
        help='Number of burn in iterations when estimating likelihood of validation data',
        default=2)
    parser.add_argument('--num-val-samples',
        type=int,
        help='Number of burn in iterations when estimating likelihood of validation data',
        default=4)
    parser.add_argument('--num-jobs',
        type=int,
        default=10)
    parser.add_argument('--num-cpu-threads',
        type=int,
        default=10)
    parser.add_argument('--per-target-model',
        action='store_true')

    parser.set_defaults(per_target_model=False, conf_int_stop=False)
    args = parser.parse_args()

    if args.per_target_model:
        args.theta_num_col = NUM_NUCLEOTIDES + 1
        args.keep_col0 = True
    else:
        args.theta_num_col = 1
        args.keep_col0 = False

    args.shazam_mut_files = args.in_shazam_mut.split(',')
    if args.in_shazam_sub is not None:
        args.shazam_sub_files = args.in_shazam_sub.split(',')
    else:
        args.shazam_sub_files = [None] * len(args.shazam_mut_files)

    args.motif_lens = [int(m) for m in args.motif_lens.split(',')]

    if args.max_motif_len is None:
        args.max_motif_len = max(args.motif_lens)
    else:
        args.max_motif_len = int(args.max_motif_len)

    if args.max_position_mutating is None:
        args.max_position_mutating = args.max_motif_len / 2
    else:
        args.max_position_mutating = int(args.max_position_mutating)

    if args.positions_mutating is None:
        # default to central base mutating
        args.max_left_flank = None
        args.max_right_flank = None
    else:
        args.positions_mutating = [[int(m) for m in positions.split(',')] for positions in args.positions_mutating.split(':')]
        for motif_len, positions in zip(args.motif_lens, args.positions_mutating):
            for m in positions:
                assert(m in range(motif_len))

        # Find the maximum left and right flanks of the motif with the largest length in the
        # hierarchy in order to process the data correctly
        args.max_left_flank = max(sum(args.positions_mutating, []))
        args.max_right_flank = max([motif_len - 1 - min(left_flanks) for motif_len, left_flanks in zip(args.motif_lens, args.positions_mutating)])

    args.scratch_dir = os.path.join(args.scratch_directory, str(time.time() + np.random.randint(10000)))
    if not os.path.exists(args.scratch_dir):
        os.makedirs(args.scratch_dir)

    return args

def get_shazam_theta(motif_len, mutability_file, target_file=None):
    """
    Take shazam csv files and turn them into our theta vector
    """

    # Read in the results from the shmulate model-fitter
    feat_gen = SubmotifFeatureGenerator(motif_len=motif_len)
    motif_list = feat_gen.motif_list

    # Read mutability matrix
    mut_motif_dict = dict()
    with open(mutability_file, "r") as model_file:
        csv_reader = csv.reader(model_file)
        motifs = csv_reader.next()[1:]
        motif_vals = csv_reader.next()[1:]
        for motif, motif_val in zip(motifs, motif_vals):
            mut_motif_dict[motif.lower()] = motif_val

    num_theta_cols = 1
    if target_file is not None:
        num_theta_cols = NUM_NUCLEOTIDES + 1
        # Read substitution matrix
        sub_motif_dict = dict()
        with open(target_file, "r") as model_file:
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

    motif_list = feat_gen.motif_list
    # Reconstruct theta in the right order
    theta = np.zeros((feat_gen.feature_vec_len, num_theta_cols))
    for motif_idx, motif in enumerate(motif_list):
        theta[motif_idx, 0] = _read_shmulate_val(mut_motif_dict[motif])
        if num_theta_cols > 1:
            for nuc in NUCLEOTIDES:
                theta[motif_idx, NUCLEOTIDE_DICT[nuc] + 1] = _read_shmulate_val(sub_motif_dict[motif][nuc])

    return theta

def write_data_for_r_plots(motif_len, thetas, out_file):
    feat_gen = SubmotifFeatureGenerator(motif_len=motif_len)
    motif_list = feat_gen.motif_list
    header = [
        'model',
        'motif',
        'target',
        'theta',
        'theta_lower',
        'theta_upper'
    ]
    data = []
    for model, theta in thetas.iteritems():
        for target in range(theta.shape[1]):
            for motif, tval in zip(motif_list, theta[:, target].ravel()):
                data.append([model, motif.upper(), target, tval, tval, tval])

    with open(out_file, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def calculate_val_log_ratios(
    args,
    input_val_gene_file,
    input_val_seq_file,
    shazam_theta,
    samm_theta
):
    def compare_thetas(theta_ref, theta_new):
        val_set_evaluator = LikelihoodComparer(
            val_set,
            full_feat_generator,
            theta_ref=theta_ref,
            num_samples=args.num_val_samples,
            burn_in=args.num_val_burnin,
            num_jobs=args.num_jobs,
            scratch_dir=args.scratch_dir,
            pool=args.all_runs_pool,
        )
        return val_set_evaluator.get_log_likelihood_ratio(theta_new)

    # sample=1 meaning take all data since we already sampled
    val_set, _ = read_gene_seq_csv_data(
        input_val_gene_file,
        input_val_seq_file,
        motif_len=args.max_motif_len,
        left_flank_len=args.max_left_flank,
        right_flank_len=args.max_right_flank,
        sample=1,
    )

    full_feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=[args.max_motif_len],
    )

    full_feat_generator.add_base_features_for_list(val_set)

    shazam_vs_samm = compare_thetas(shazam_theta, samm_theta)
    samm_vs_shazam = compare_thetas(samm_theta, shazam_theta)

    return {'shazam_ref': shazam_vs_samm, 'samm_ref': samm_vs_shazam}

def main(args=sys.argv[1:]):
    args = parse_args()

    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)

    if args.num_cpu_threads > 1:
        args.all_runs_pool = Pool(args.num_cpu_threads)
    else:
        args.all_runs_pool = None

    shazam_theta_list = [
        get_shazam_theta(
            args.max_motif_len,
            shazam_mut_csv,
            shazam_sub_csv
        ) for shazam_mut_csv, shazam_sub_csv in
        zip(args.shazam_mut_files, args.shazam_sub_files)
    ]
    samm_theta_list = [
        load_fitted_model(
            samm_pkl,
            args.max_motif_len,
            args.max_position_mutating,
            args.keep_col0,
            add_targets=False,
        ).agg_refit_theta for samm_pkl in \
        args.in_samm.split(',')
    ]

    write_data_for_r_plots(
        args.max_motif_len,
        {
            'shazam': np.mean(shazam_theta_list, axis=0),
            'samm': np.mean(samm_theta_list, axis=0),
        },
        args.out_thetas,
    )

    zipped_params = zip(
        args.input_gene_files.split(','),
        args.input_seq_files.split(','),
        shazam_theta_list,
        samm_theta_list
    )
    data = []
    for params in zipped_params:
        data.append(calculate_val_log_ratios(args, *params))

    with open(args.out_log_lik_file, 'w') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main(sys.argv[1:])
