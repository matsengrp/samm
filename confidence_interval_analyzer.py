import sys
import argparse
import pickle
import numpy as np
import scipy.stats
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--fitted-model',
        type=str,
        help='fitted model pickle',
        default='_output/context_model.pkl')
    parser.add_argument('--true-model',
        type=str,
        help='true model pickle file',
        default='_output/true_model.pkl')
    parser.add_argument('--z',
        type=float,
        help="z statistic",
        default=1.96)

    args = parser.parse_args()
    return args

def main(args=sys.argv[1:]):
    args = parse_args()

    if args.true_model:
        with open(args.true_model, "r") as f:
            true_model = pickle.load(f)
        true_theta = true_model[0]
        print true_theta.shape

    with open(args.fitted_model, "r") as f:
        fitted_models = pickle.load(f)

    feat_gen = HierarchicalMotifFeatureGenerator(
        motif_lens=fitted_models[0].motif_lens,
        left_motif_flank_len_list=fitted_models[0].positions_mutating,
    )

    good_models = [f_model for f_model in fitted_models if f_model.has_refit_data and f_model.variance_est is not None]
    max_idx = np.argmax([f_model.num_not_crossing_zero for f_model in good_models]) # Take the one with the most nonzero and the largest penalty parameter
    for max_idx in range(len(good_models)):
        best_model = good_models[max_idx]
        refit_model = best_model.refit_theta
        motif_mask = ~best_model.model_masks.feats_to_remove_mask

        small_true_model = true_theta[motif_mask,:]
        small_penalized_model = best_model.penalized_theta[motif_mask,:]

        refit_model_flat = refit_model[best_model.refit_possible_theta_mask]
        small_true_model_flat = small_true_model[best_model.refit_possible_theta_mask]
        small_penalized_model_flat = small_penalized_model[best_model.refit_possible_theta_mask]
        se = np.sqrt(np.diag(best_model.variance_est))
        lower = refit_model_flat - args.z * se
        upper = refit_model_flat + args.z * se

        # calculate coverage of groups of theta values
        thres_coverages = []
        thresholds = np.percentile(true_theta, [25, 50, 75, 100])
        prev_thres = -np.inf
        for i, thres in enumerate(thresholds):
            thres_mask = (true_theta > prev_thres) & (true_theta <= thres)
            true_theta_thres = np.copy(true_theta)
            true_theta_thres[~thres_mask] = np.nan
            small_true_theta_thres = true_theta_thres[motif_mask, :]
            small_true_theta_thres_flat = small_true_theta_thres[best_model.refit_possible_theta_mask]
            small_thres_mask = ~np.isnan(small_true_theta_thres_flat)
            if np.sum(small_thres_mask) > 0:
                small_true_theta_thres_final = small_true_theta_thres_flat[small_thres_mask]
                thres_coverage = np.mean((lower[small_thres_mask] < small_true_theta_thres_final) & (small_true_theta_thres_final < upper[small_thres_mask]))
                thres_coverages.append(thres_coverage)
            else:
                thres_coverages.append(np.nan)
            prev_thres = thres


        print("num nonzero theta %d" % best_model.num_p)
        print("  num dont cross zero %d" % best_model.num_not_crossing_zero)
        print("  coverage %f" % np.mean((lower < small_true_model_flat) & (upper > small_true_model_flat)))
        print("  thres coverage %s" % thres_coverages)
        print("  support TP of fit %f" % (np.sum(motif_mask & (true_model[0].ravel() != 0))/float(np.sum(motif_mask))))
        print("  support TP of truth %f" % (np.sum(motif_mask & (true_model[0].ravel() != 0))/float(np.sum(true_model[0].ravel() != 0))))
        print("  refit: pearson %f, %f" % scipy.stats.pearsonr(refit_model_flat, small_true_model_flat))
        print("  refit: spearman %f, %f" % scipy.stats.spearmanr(refit_model_flat, small_true_model_flat))
        print("  penal: pearson %f, %f" % scipy.stats.pearsonr(small_penalized_model_flat, small_true_model_flat))
        print("  penal: spearman %f, %f" % scipy.stats.spearmanr(small_penalized_model_flat, small_true_model_flat))

if __name__ == "__main__":
    main(sys.argv[1:])
