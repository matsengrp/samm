import sys
import argparse
import pickle
import numpy as np
import scipy.stats

from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from confidence_interval_maker import ConfidenceIntervalMaker
from make_model_sparse import SparseModelMaker
from common import *
import matplotlib as plt

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--fitted-models',
        type=str,
        help='fitted model pickle, comma separated, colon separated')
    parser.add_argument('--model-labels',
        type=str,
        help='model labels, colon separated',
        default="40,120,360")
    parser.add_argument('--shmulate-models',
        type=str,
        help='fitted model shmulate pickle, comma separated, colon separated')
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
        choices=("norm", "raw_norm", "coverage", "raw_coverage", "raw_pearson", "pearson", "support"),
    )
    parser.add_argument('--z',
        type=float,
        help="z statistic",
        default=1.96)

    args = parser.parse_args()
    args.fitted_models = args.fitted_models.split(":")
    for i, fmodels in enumerate(args.fitted_models):
        args.fitted_models[i] = fmodels.split(",")
    args.model_labels = args.model_labels.split(":")
    args.true_models = args.true_models.split(":")

    if args.stat == "norm":
        args.stat_func = _get_agg_norm_diff
    elif args.stat == "raw_norm":
        args.stat_func = _get_raw_norm_diff
    elif args.stat == "raw_coverage":
        args.stat_func = _get_raw_coverage
    elif args.stat == "coverage":
        args.stat_func = _get_agg_coverage
    elif args.stat == "raw_pearson":
        args.stat_func = _get_raw_pearson
    elif args.stat == "pearson":
        args.stat_func = _get_agg_pearson
    elif args.stat == "support":
        args.stat_func = _get_support

    return args

def load_fitted_model(file_name, agg_motif_len, agg_pos_mutating):
    with open(file_name, "r") as f:
        fitted_models = pickle.load(f)
        best_model = pick_best_model(fitted_models)

    hier_feat_gen = HierarchicalMotifFeatureGenerator(
        motif_lens=best_model.motif_lens,
        feats_to_remove=best_model.model_masks.feats_to_remove,
        left_motif_flank_len_list=best_model.positions_mutating,
    )
    agg_feat_gen = HierarchicalMotifFeatureGenerator(
        motif_lens=[agg_motif_len],
        left_motif_flank_len_list=[[agg_pos_mutating]],
    )
    best_model.agg_refit_theta = create_aggregate_theta(
        hier_feat_gen,
        agg_feat_gen,
        best_model.refit_theta,
        best_model.model_masks.zero_theta_mask_refit,
        best_model.refit_possible_theta_mask,
        keep_col0=False,
    )

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

def _get_support(fmodel, full_feat_generator, raw_true_theta, agg_true_theta, possible_agg_mask):
    raw_refit_theta = np.zeros(raw_true_theta.shape)
    raw_refit_theta[~fmodel.model_masks.feats_to_remove_mask, :] = fmodel.refit_theta
    support_refit_theta = (raw_refit_theta != -np.inf) & (raw_refit_theta != 0)
    true_support = (raw_true_theta != -np.inf) & (raw_true_theta != 0)
    return float(np.sum(support_refit_theta & true_support))/np.sum(support_refit_theta)

def _get_raw_pearson(fmodel, full_feat_generator, raw_true_theta, agg_true_theta, possible_agg_mask):
    raw_refit_theta = np.zeros(raw_true_theta.shape)
    raw_refit_theta[~fmodel.model_masks.feats_to_remove_mask, :] = fmodel.refit_theta
    possible_raw_mask = raw_true_theta != -np.inf
    return scipy.stats.pearsonr(raw_true_theta[possible_raw_mask], raw_refit_theta[possible_raw_mask])[0]

def _get_agg_pearson(fmodel, full_feat_generator, raw_true_theta, agg_true_theta, possible_agg_mask):
    return scipy.stats.pearsonr(
        agg_true_theta[possible_agg_mask],
        fmodel.agg_refit_theta[possible_agg_mask],
    )[0]

def _get_raw_norm_diff(fmodel, full_feat_generator, raw_true_theta, agg_true_theta, possible_agg_mask):
    raw_refit_theta = np.zeros(raw_true_theta.shape)
    raw_refit_theta[~fmodel.model_masks.feats_to_remove_mask, :] = fmodel.refit_theta
    possible_raw_mask = raw_true_theta != -np.inf
    tot_elems = np.sum(possible_raw_mask)
    return np.linalg.norm(raw_true_theta[possible_raw_mask] - raw_refit_theta[possible_raw_mask])/np.sqrt(tot_elems)

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
    tot_covered = 0
    tot_considered = 0
    agg_fitted_thetas = []
    for col_idx in range(agg_true_theta.shape[1]):
        agg_fitted_theta, agg_fitted_lower, agg_fitted_upper = combine_thetas_and_get_conf_int(
            hier_feat_gen,
            full_feat_generator,
            fmodel.refit_theta,
            fmodel.model_masks.zero_theta_mask_refit,
            fmodel.refit_possible_theta_mask,
            fmodel.variance_est,
            col_idx=col_idx + 1 if agg_true_theta.shape[1] == NUM_NUCLEOTIDES else 0,
            zstat=1.96,
        )
        agg_fitted_thetas.append((agg_fitted_theta, agg_fitted_lower, agg_fitted_upper))

    all_theta = np.vstack([t[0] for t in agg_fitted_thetas])
    med_theta = np.median(all_theta[all_theta != -np.inf])

    for col_idx, (agg_fitted_theta, agg_fitted_lower, agg_fitted_upper) in enumerate(agg_fitted_thetas):
        agg_fitted_lower -= med_theta
        agg_fitted_upper -= med_theta
        comparison_mask = np.abs(agg_fitted_lower - agg_fitted_upper) > 1e-5 # only look at things with confidence intervals

        agg_fitted_lower_small = agg_fitted_lower[comparison_mask]
        agg_fitted_upper_small = agg_fitted_upper[comparison_mask]
        agg_true_theta_col = agg_true_theta[:,col_idx] - np.median(agg_true_theta[agg_true_theta != -np.inf])
        agg_true_theta_small = agg_true_theta_col[comparison_mask]

        num_covered = np.sum((agg_fitted_lower_small - 1e-5 <= agg_true_theta_small) & (agg_fitted_upper_small + 1e-5 >= agg_true_theta_small))
        tot_covered += num_covered
        num_considered = np.sum(comparison_mask)
        tot_considered += num_considered

    return tot_covered/float(tot_considered)

def _get_raw_coverage(fmodel, full_feat_generator, raw_true_theta, true_theta, possible_agg_mask):
    conf_int = ConfidenceIntervalMaker.create_confidence_intervals(
        fmodel.refit_theta,
        np.sqrt(np.diag(fmodel.variance_est)),
        fmodel.refit_possible_theta_mask,
        fmodel.model_masks.zero_theta_mask_refit,
        z=ZSCORE_95,
    )

    theta_mask = fmodel.refit_possible_theta_mask & ~fmodel.model_masks.zero_theta_mask_refit
    theta_mask_flat = theta_mask.reshape((theta_mask.size,), order="F")

    true_theta_trunc = np.array(raw_true_theta[~fmodel.model_masks.feats_to_remove_mask, :])
    true_theta_trunc = true_theta_trunc.reshape((true_theta_trunc.size,), order="F")
    true_theta_trunc = true_theta_trunc[theta_mask_flat]
    return np.mean((conf_int[:,0] <= true_theta_trunc) & (true_theta_trunc <= conf_int[:,2]))

def _load_true_model(file_name, agg_motif_len, agg_pos_mutating, hier_motif_lens, hier_positions_mutating):
    with open(file_name, "r") as f:
        true_model_agg, true_model = pickle.load(f)

    dense_agg_feat_gen = HierarchicalMotifFeatureGenerator(
        motif_lens=[agg_motif_len],
        left_motif_flank_len_list=[[agg_pos_mutating]],
    )

    dense_hier_feat_gen = HierarchicalMotifFeatureGenerator(
        motif_lens=hier_motif_lens,
        left_motif_flank_len_list=hier_positions_mutating,
    )

    sparse_raw_theta = SparseModelMaker.solve(true_model_agg, dense_hier_feat_gen, dense_agg_feat_gen, raw_theta=true_model)
    possible_raw_theta_mask = get_possible_motifs_to_targets(dense_hier_feat_gen.motif_list, sparse_raw_theta.shape, dense_hier_feat_gen.mutating_pos_list)
    sparse_raw_theta[~possible_raw_theta_mask] = -np.inf
    return np.array(true_model_agg), np.array(sparse_raw_theta)

def main(args=sys.argv[1:]):
    args = parse_args()

    fitted_models = [
        [load_fitted_model(file_name, args.agg_motif_len, args.agg_pos_mutating) for file_name in fnames]
        for fnames in args.fitted_models
    ]
    example_model = fitted_models[0][0]

    true_thetas = [
        _load_true_model(
            tmodel,
            args.agg_motif_len,
            args.agg_pos_mutating,
            example_model.motif_lens,
            example_model.positions_mutating,
        ) for tmodel in args.true_models
    ]

    num_cols = fitted_models[0][0].refit_theta.shape[1]
    per_target = num_cols == NUM_NUCLEOTIDES + 1

    samm_means = []
    for i in range(len(fitted_models)):
        statistics = _collect_statistics(fitted_models[i], args, true_thetas[i][1], true_thetas[i][0], args.stat_func)
        mean = np.mean(statistics)
        se = np.sqrt(np.var(statistics)
        samm_means.append(mean)
        samm_se.append(se)
        print "MEAN", args.stat, result[0], "(%f)" % result[1]

    plt.errorbar(args.model_labels, samm_means, samm_se, linestyle='None')
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
