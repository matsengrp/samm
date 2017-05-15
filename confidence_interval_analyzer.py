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

    with open(args.fitted_model, "r") as f:
        fitted_models = pickle.load(f)

    feat_gen = HierarchicalMotifFeatureGenerator(
        motif_lens=fitted_models[0].motif_lens,
        left_motif_flank_len_list=fitted_models[0].positions_mutating,
    )
    for f_model in fitted_models:
        if f_model.has_refit_data and f_model.variance_est is not None:
            feat_generator_stage2 = HierarchicalMotifFeatureGenerator(
                motif_lens=f_model.motif_lens,
                feats_to_remove=f_model.model_masks.feats_to_remove,
                left_motif_flank_len_list=f_model.positions_mutating,
            )
            refit_model = f_model.refit_theta
            motif_mask = ~f_model.model_masks.feats_to_remove_mask

            small_true_model = true_model[0][motif_mask,:]
            small_penalized_model = f_model.penalized_theta[motif_mask,:]

            refit_model_flat = refit_model[f_model.refit_possible_theta_mask]
            small_true_model_flat = small_true_model[f_model.refit_possible_theta_mask]
            small_penalized_model_flat = small_penalized_model[f_model.refit_possible_theta_mask]

            se = np.sqrt(np.diag(f_model.variance_est))
            lower = refit_model_flat - args.z * se
            upper = refit_model_flat + args.z * se

            print("num nonzero theta %d" % f_model.num_p)
            print("  num dont cross zero %d" % f_model.num_not_crossing_zero)
            print("  coverage %f" % np.mean((lower < small_true_model_flat) & (upper > small_true_model_flat)))
            print("  support TP %f" % (np.sum(motif_mask & (true_model[0].ravel() != 0))/float(np.sum(motif_mask))))
            print("  refit: pearson %f, %f" % scipy.stats.pearsonr(refit_model_flat, small_true_model_flat))
            print("  refit: spearman %f, %f" % scipy.stats.spearmanr(refit_model_flat, small_true_model_flat))
            print("  penal: pearson %f, %f" % scipy.stats.pearsonr(small_penalized_model_flat, small_true_model_flat))
            print("  penal: spearman %f, %f" % scipy.stats.spearmanr(small_penalized_model_flat, small_true_model_flat))

if __name__ == "__main__":
    main(sys.argv[1:])
