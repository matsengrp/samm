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
    parser.add_argument('--in-shazam',
        type=str,
        help='pickle of shazam fit')
    parser.add_argument('--in-samm',
        type=str,
        help='pickle of samm fit')
    parser.add_argument('--in-samm-same-target',
        type=str,
        help='comma separated samm csv files',
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
        default=16)
    parser.add_argument('--num-val-samples',
        type=int,
        help='Number of burn in iterations when estimating likelihood of validation data',
        default=16)
    parser.add_argument('--num-jobs',
        type=int,
        default=10)
    parser.add_argument('--num-cpu-threads',
        type=int,
        default=10)
    parser.add_argument('--per-target-model',
        action='store_true')
    parser.add_argument('--center-median',
        action='store_true')

    parser.set_defaults(per_target_model=False, conf_int_stop=False)
    args = parser.parse_args()

    if args.per_target_model:
        args.theta_num_col = NUM_NUCLEOTIDES + 1
        args.keep_col0 = False
    else:
        args.theta_num_col = 1
        args.keep_col0 = True

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

def _get_shazam_theta(shazam_pkl, per_target=False, center_median=False):
    """
    Take shazam pickle return theta vector
    """

    with open(shazam_pkl, 'r') as f:
        theta_mut, (theta_target, theta_sub) = pickle.load(f)

    if per_target:
        theta = np.concatenate((theta_mut, theta_sub), axis=1)
    else:
        theta = theta_mut

    if np.any(np.isnan(theta)):
        print "SHAZAM contains nan in the estimates"
        theta[np.isnan(theta)] = 0

    if center_median:
        theta -= np.median(theta)

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

    if args.num_cpu_threads > 1:
        args.all_runs_pool = Pool(args.num_cpu_threads)
    else:
        args.all_runs_pool = None

    shazam_theta_list = [
        _get_shazam_theta(
            shazam_pkl,
            args.per_target_model,
            args.center_median,
        ) for shazam_pkl in
        args.in_shazam.split(',')
    ]
    samm_theta_list = [
        load_fitted_model(
            samm_pkl,
            args.max_motif_len,
            args.max_position_mutating,
            keep_col0=True,
            add_targets=False,
            center_median=args.center_median,
        ).agg_refit_theta for samm_pkl in \
        args.in_samm.split(',')
    ]

    if args.out_thetas is not None:
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
