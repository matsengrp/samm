import sys
import argparse
import pickle
import numpy as np
import scipy.stats

from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from confidence_interval_maker import ConfidenceIntervalMaker
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--fitted-models',
        type=str,
        help='fitted model pickle, comma separated, colon separated')
    parser.add_argument('--true-models',
        type=str,
        help='true model pickle file, colon separated')
    parser.add_argument('--agg-motif-len',
        type=int,
        default=3)
    parser.add_argument('--agg-pos-mutating',
        type=int,
        default=1)
    parser.add_argument('--stat',
        type=str,
        default="norm",
        choices=("norm", "coverage", "support", "pearson"),
    )
    parser.add_argument('--z',
        type=float,
        help="z statistic",
        default=1.96)

    args = parser.parse_args()
    args.fitted_models = args.fitted_models.split(":")
    for i, fmodels in enumerate(args.fitted_models):
        args.fitted_models[i] = fmodels.split(",")
    args.true_models = args.true_models.split(":")

    if args.stat == "norm":
        args.stat_func = _get_agg_norm_diff
    elif args.stat == "coverage":
        args.stat_func = _get_agg_coverage
    elif args.stat == "pearson":
        args.stat_func = _get_pearson

    return args

def load_fitted_model(file_name, agg_motif_len, agg_pos_mutating):
    with open(file_name, "r") as f:
        fitted_models = pickle.load(f)

    good_models = [f_model for f_model in fitted_models if f_model.has_refit_data and f_model.variance_est is not None]
    if len(good_models) == 0:
        return None
    max_idx = np.argmax([f_model.num_not_crossing_zero for f_model in good_models])# Take the one with the most nonzero and the largest penalty parameter
    best_model = good_models[max_idx]

    hier_feat_gen = HierarchicalMotifFeatureGenerator(
        motif_lens=best_model.motif_lens,
        feats_to_remove=best_model.model_masks.feats_to_remove,
        left_motif_flank_len_list=best_model.positions_mutating,
    )
    agg_feat_gen = HierarchicalMotifFeatureGenerator(
        motif_lens=[agg_motif_len],
        left_motif_flank_len_list=[[agg_pos_mutating]],
    )
    best_model.agg_refit_theta = create_aggregate_theta(hier_feat_gen, agg_feat_gen, best_model.refit_theta)
    if best_model.agg_refit_theta.shape[1] == NUM_NUCLEOTIDES + 1:
        best_model.agg_refit_theta = best_model.agg_refit_theta[:, 0:1] + best_model.agg_refit_theta[:, 1:]
    return best_model

def _collect_statistics(fitted_models, args, raw_true_theta, agg_true_theta, stat_func):
    dense_agg_feat_gen = HierarchicalMotifFeatureGenerator(
        motif_lens=[args.agg_motif_len],
        left_motif_flank_len_list=[[args.agg_pos_mutating]],
    )

    possible_agg_mask = get_possible_motifs_to_targets(
        dense_agg_feat_gen.motif_list,
        mask_shape=agg_true_theta.shape,
        mutating_pos_list=[args.agg_pos_mutating] * dense_agg_feat_gen.feature_vec_len,
    )
    statistics = [stat_func(fmodel, dense_agg_feat_gen, raw_true_theta, agg_true_theta, possible_agg_mask) for fmodel in fitted_models if fmodel is not None]
    return statistics

def _get_pearson(fmodel, full_feat_generator, raw_true_theta, agg_true_theta, possible_agg_mask):
    return scipy.stats.pearsonr(
        agg_true_theta[possible_agg_mask],
        fmodel.agg_refit_theta[possible_agg_mask],
    )[0]

def _get_raw_pearson(fmodel, full_feat_generator, raw_true_theta, agg_true_theta, possible_agg_mask):
    return scipy.stats.pearsonr(
        raw_true_theta[~fmodel.model_masks.feats_to_remove_mask],
        fmodel.refit_theta,
    )[0]

def _get_agg_norm_diff(fmodel, full_feat_generator, raw_true_theta, agg_true_theta, possible_agg_mask):
    tot_elems = np.sum(possible_agg_mask)

    # Subtract the median
    possible_agg_true_theta = agg_true_theta[possible_agg_mask] - np.median(agg_true_theta[possible_agg_mask])
    possible_agg_refit_theta = fmodel.agg_refit_theta[possible_agg_mask] - np.median(fmodel.agg_refit_theta[possible_agg_mask])

    return np.linalg.norm(possible_agg_refit_theta - possible_agg_true_theta)/np.sqrt(tot_elems)

def _get_agg_coverage(fmodel, full_feat_generator, raw_true_theta, agg_true_theta, possible_agg_mask):
    hier_feat_gen = HierarchicalMotifFeatureGenerator(
        motif_lens=fmodel.motif_lens,
        feats_to_remove=fmodel.model_masks.feats_to_remove,
        left_motif_flank_len_list=fmodel.positions_mutating,
    )

    # calculate coverage of groups of theta values
    agg_coverage = []
    for col_idx in range(agg_true_theta.shape[1]):
        print "col_idx", col_idx
        agg_fitted_theta, agg_fitted_lower, agg_fitted_upper = combine_thetas_and_get_conf_int(
            hier_feat_gen,
            full_feat_generator,
            fmodel.refit_theta,
            # fmodel.refit_theta[:,col_idx:col_idx + 1] - np.median(fmodel.refit_theta[:,col_idx:col_idx + 1]),
            covariance_est=fmodel.variance_est,
            col_idx=col_idx + 1 if agg_true_theta.shape[1] == NUM_NUCLEOTIDES else 0,
            zstat=1.96,
        )
        #agg_fitted_lower -= med_theta
        #agg_fitted_upper -= med_theta
        comparison_mask = np.abs(agg_fitted_lower - agg_fitted_upper) > 1e-5 # only look at things with confidence intervals
        print np.sum(comparison_mask)
        # comparison_mask = np.ones(agg_fitted_lower.shape, dtype=bool)
        agg_fitted_lower_small = agg_fitted_lower[comparison_mask]
        agg_fitted_upper_small = agg_fitted_upper[comparison_mask]

        agg_true_theta_col = agg_true_theta[:,col_idx] #- np.median(agg_true_theta[:,col_idx])
        agg_true_theta_small = agg_true_theta_col[comparison_mask]
    # print np.vstack([agg_fitted_lower_small, agg_true_theta_small, agg_fitted_upper_small])
        coverage = np.mean((agg_fitted_lower_small - 1e-5 <= agg_true_theta_small) & (agg_fitted_upper_small + 1e-5 >= agg_true_theta_small))
        agg_coverage.append(coverage)
    print "agg_cover", agg_coverage
    return np.mean(agg_coverage)

def _get_raw_coverage(fmodel, full_feat_generator, raw_true_theta, true_theta, possible_agg_mask):
    conf_int = ConfidenceIntervalMaker.create_confidence_intervals(
        fmodel.refit_theta - np.median(fmodel.refit_theta),
        np.sqrt(np.diag(fmodel.variance_est)),
        fmodel.refit_possible_theta_mask,
        fmodel.model_masks.zero_theta_mask_refit,
        z=1.96,
    )

    raw_true_theta -= np.median(raw_true_theta)
    true_theta_trunc = raw_true_theta[~fmodel.model_masks.feats_to_remove_mask]
    true_theta_trunc = true_theta_trunc.reshape((true_theta_trunc.size,), order="F")
    return np.mean((conf_int[:,0] <= true_theta_trunc) & (true_theta_trunc <= conf_int[:,2]))


def _load_true_model(file_name):
    with open(file_name, "r") as f:
        true_model_agg, true_model = pickle.load(f)

    # Find the most parsimonious version of this theta
    true_theta_lasso = true_model
    if true_model_agg.shape[1] == NUM_NUCLEOTIDES + 1:
        true_model_agg = true_model_agg[:,0:1] + true_model_agg[:,1:]
    return true_model_agg, true_theta_lasso

def main(args=sys.argv[1:]):
    args = parse_args()

    true_thetas = [_load_true_model(tmodel) for tmodel in args.true_models]

    fitted_models = [
        [load_fitted_model(file_name, args.agg_motif_len, args.agg_pos_mutating) for file_name in fnames]
        for fnames in args.fitted_models
    ]

    num_cols = fitted_models[0][0].refit_theta.shape[1]
    per_target = num_cols == NUM_NUCLEOTIDES + 1

    for i in range(len(fitted_models)):
        statistics = _collect_statistics(fitted_models[i], args, true_thetas[i][1], true_thetas[i][0], args.stat_func)
        # print "statistics", statistics
        print "MEAN", np.mean(statistics), "(%f)" % np.sqrt(np.var(statistics))

if __name__ == "__main__":
    main(sys.argv[1:])
