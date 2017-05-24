import sys
import argparse
import pickle
import numpy as np
import scipy.stats
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--fitted-models',
        type=str,
        help='fitted model pickle, comma separated',
        default='_output/context_model.pkl')
    parser.add_argument('--true-model',
        type=str,
        help='true model pickle file',
        default='_output/true_model.pkl')
    parser.add_argument('--agg-motif-len',
        type=int,
        default=5)
    parser.add_argument('--agg-pos-mutating',
        type=int,
        default=2)
    parser.add_argument('--z',
        type=float,
        help="z statistic",
        default=1.96)

    args = parser.parse_args()
    args.fitted_models = args.fitted_models.split(",")
    return args

def load_fitted_model(file_name, agg_motif_len, agg_pos_mutating):
    with open(file_name, "r") as f:
        fitted_models = pickle.load(f)

    good_models = [f_model for f_model in fitted_models if f_model.has_refit_data and f_model.variance_est is not None]
    max_idx = np.argmax([f_model.num_not_crossing_zero for f_model in good_models]) # Take the one with the most nonzero and the largest penalty parameter
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

def get_norm_diff(fmodel, agg_true_theta, possible_agg_mask):
    tot_elems = np.sum(possible_agg_mask)
    return np.linalg.norm(fmodel.agg_refit_theta[possible_agg_mask] - agg_true_theta[possible_agg_mask])/np.sqrt(tot_elems)

def main(args=sys.argv[1:]):
    args = parse_args()

    if args.true_model:
        with open(args.true_model, "r") as f:
            true_model = pickle.load(f)
        true_theta = true_model[1]

    fitted_models = [load_fitted_model(file_name, args.agg_motif_len, args.agg_pos_mutating) for file_name in args.fitted_models]
    num_cols = fitted_models[0].refit_theta.shape[1]
    per_target = num_cols == NUM_NUCLEOTIDES + 1

    dense_hier_feat_gen = HierarchicalMotifFeatureGenerator(
        motif_lens=fitted_models[0].motif_lens,
        left_motif_flank_len_list=fitted_models[0].positions_mutating,
    )
    dense_agg_feat_gen = HierarchicalMotifFeatureGenerator(
        motif_lens=[args.agg_motif_len],
        left_motif_flank_len_list=[[args.agg_pos_mutating]],
    )

    agg_true_theta = create_aggregate_theta(dense_hier_feat_gen, dense_agg_feat_gen, true_theta)
    if per_target:
        agg_true_theta = agg_true_theta[:,0:1] + agg_true_theta[:,1:]
    possible_agg_mask = get_possible_motifs_to_targets(
        dense_agg_feat_gen.motif_list,
        mask_shape=agg_true_theta.shape,
        mutating_pos_list=[args.agg_pos_mutating] * dense_agg_feat_gen.feature_vec_len,
    )

    norm_diffs = [get_norm_diff(fmodel, agg_true_theta, possible_agg_mask) for fmodel in fitted_models]
    print "norm_diffs", norm_diffs
    print "MEAN", np.mean(norm_diffs)
    print "SE", np.sqrt(np.var(norm_diffs))

if __name__ == "__main__":
    main(sys.argv[1:])
