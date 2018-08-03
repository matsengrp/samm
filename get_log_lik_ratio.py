"""
Compare log likelihood on test set (using the lower bound heuristic)
Comparing logistic vs samm vs shazam
"""

import sys
import argparse
import os
import os.path
import pickle
import time
import numpy as np
import itertools

SCRATCH_DIR = '/fh/fast/matsen_e/dshaw/_tmp/samm/'

#SAMM_PATH = '/home/jfeng2//mobeef'
#sys.path.insert(1, SAMM_PATH)

from read_data import read_gene_seq_csv_data, load_logistic_model
from fit_logistic_model import LogisticModel
from common import pick_best_model, NUM_NUCLEOTIDES
from likelihood_evaluator import LikelihoodComparer
from compare_simulated_shazam_vs_samm import ShazamModel
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator

FOLDER_SEED = 12

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=444)
    parser.add_argument('--input-naive',
        type=str,
        help='Input CSV file with naive sequences',
        default='shazam_vs_samm/_output/%d/processed_genes.csv' % FOLDER_SEED)
    parser.add_argument('--input-mutated',
        type=str,
        help='Input CSV file with naive sequences',
        default='shazam_vs_samm/_output/%d/processed_test_seqs.csv' % FOLDER_SEED)
    parser.add_argument('--input-samm',
        type=str,
        default='shazam_vs_samm/_output/%d/motif-3-5-flank-1--2/False/fitted_try.pkl' % FOLDER_SEED)
    parser.add_argument('--input-shazam',
        type=str,
        default='shazam_vs_samm/_output/%d/motif-3-5-flank-1--2/False/fitted_shazam_mut.csv' % FOLDER_SEED)
    parser.add_argument('--input-logistic',
        type=str,
        default='shazam_vs_samm/_output/%d/motif-3-5-flank-1--2/False/logistic_model.pkl' % FOLDER_SEED)
    parser.add_argument('--scratch-directory',
        type=str,
        help='Directory for writing temporary files',
        default=SCRATCH_DIR)
    parser.add_argument('--num-val-burnin',
        type=int,
        help='Number of burn-in iterations when estimating likelihood of validation data',
        default=16)
    parser.add_argument('--num-val-samples',
        type=int,
        help='Number of mutation order samples drawn per observation when estimating likelihood of validation data',
        default=16)
    parser.add_argument('--num-jobs',
        type=int,
        help='Number of jobs to submit to a Slurm cluster during E-step. (If using only 1 job, it does not submit to a cluster.)',
        default=10)

    parser.set_defaults(per_target_model=False, motif_len=5, left_flank=2, right_flank=2)
    args = parser.parse_args()

    args.scratch_dir = os.path.join(args.scratch_directory, str(time.time() + np.random.randint(10000)))
    if not os.path.exists(args.scratch_dir):
        os.makedirs(args.scratch_dir)

    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    print(args)
    np.random.seed(args.seed)

    thetas = []
    labels = []
    # get data
    obs_data, metadata = read_gene_seq_csv_data(
        args.input_naive,
        args.input_mutated,
        motif_len=args.motif_len,
        left_flank_len=args.left_flank,
        right_flank_len=args.right_flank,
    )

    ## get samm theta
    with open(args.input_samm, "r") as f:
        method_results = pickle.load(f)
        method_res = pick_best_model(method_results)
        feat_generator = method_res.refit_feature_generator
        per_target_model = method_res.refit_theta.shape[1] == NUM_NUCLEOTIDES + 1

    max_motif_len = args.motif_len
    full_feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=[max_motif_len],
        left_motif_flank_len_list=[[args.left_flank]],
    )
    full_feat_generator.add_base_features_for_list(obs_data)

    samm_theta = feat_generator.create_aggregate_theta(method_res.refit_theta, keep_col0=False)
    agg_possible_motif_mask = full_feat_generator.get_possible_motifs_to_targets(samm_theta.shape)
    samm_theta[~agg_possible_motif_mask] = -np.inf
    thetas += [samm_theta]
    labels += ['samm']

    # get shazam theta
    shazam_model = ShazamModel(max_motif_len, args.input_shazam, None, wide_format=True)

    shazam_theta = shazam_model.agg_refit_theta
    thetas += [shazam_theta]
    labels += ['shazam']

    # get logistic theta
    logistic_model = load_logistic_model(args.input_logistic)

    logistic_theta = logistic_model.agg_refit_theta
    thetas += [logistic_theta]
    labels += ['logistic']

    # do all comparisons
    cur_ref_idx = -1
    for idx in itertools.permutations(range(len(thetas)), 2):
        if idx[0] != cur_ref_idx:
            # need to update val_set_evaluator
            cur_ref_idx = idx[0]
            val_set_evaluator = LikelihoodComparer(
                obs_data,
                full_feat_generator,
                theta_ref=thetas[idx[0]],
                num_samples=args.num_val_samples,
                burn_in=args.num_val_burnin,
                num_jobs=args.num_jobs,
                scratch_dir=args.scratch_dir,
            )
        log_lik_ratio, lower_bound, upper_bound = val_set_evaluator.get_log_likelihood_ratio(thetas[idx[1]])
        print "{} with {} reference:".format(labels[idx[1]], labels[idx[0]])
        print "(lower, ratio, upper) = (%.4f, %.4f, %.4f)" % (lower_bound, log_lik_ratio, upper_bound)

if __name__ == "__main__":
    main(sys.argv[1:])
