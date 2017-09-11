import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import scipy.stats

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns

from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from confidence_interval_maker import ConfidenceIntervalMaker
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--fitted-models',
        type=str,
        default="simulation_section/_output/%s/nonzero%s/effect_size_%s/samples%s/0%d/samm/fitted.pkl",
        help='fitted model pickle, comma separated, colon separated, colon colon separated')
    parser.add_argument('--true-models',
        type=str,
        default="simulation_section/_output/%s/nonzero%s/effect_size_%s/true_model.pkl",
        help='true model pickle file, colon separated, colon colon separated')
    parser.add_argument('--model-types',
        type=str,
        default="3_targetFalse,3_targetTrue,2_3_targetFalse",
        help='model names')
    parser.add_argument('--effect-sizes',
        type=str,
        default="50,100,200")
    parser.add_argument('--sample-sizes',
        type=str,
        default="100,200,400")
    parser.add_argument('--sparsities',
        type=str,
        default="25,50,100")
    parser.add_argument('--reps',
        type=int,
        default=10)
    parser.add_argument('--agg-motif-len',
        type=int,
        default=3)
    parser.add_argument('--agg-pos-mutating',
        type=int,
        default=1)
    parser.add_argument('--stats',
        type=str,
        default="norm,kendall,coverage",
    )
    parser.add_argument('--outfile',
        type=str,
        default="_output/simulation3mer.pdf")

    args = parser.parse_args()
    args.model_types = args.model_types.split(",")
    args.stats = args.stats.split(",")
    args.sample_sizes = args.sample_sizes.split(",")
    args.effect_sizes = args.effect_sizes.split(",")
    args.sparsities = args.sparsities.split(",")
    return args

def load_fitted_model(file_name, agg_motif_len, agg_pos_mutating):
    with open(file_name, "r") as f:
        fitted_models = pickle.load(f)
        best_model = pick_best_model(fitted_models)

    if best_model is None:
        print "FAIL", file_name
        return None

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

def _collect_statistics(fitted_models, args, agg_true_theta, stat_func):
    dense_agg_feat_gen = HierarchicalMotifFeatureGenerator(
        motif_lens=[args.agg_motif_len],
        left_motif_flank_len_list=[[args.agg_pos_mutating]],
    )

    possible_agg_mask = get_possible_motifs_to_targets(
        dense_agg_feat_gen.motif_list,
        mask_shape=agg_true_theta.shape,
        mutating_pos_list=[args.agg_pos_mutating] * dense_agg_feat_gen.feature_vec_len,
    )
    statistics = []
    for fmodel in fitted_models:
        if fmodel is not None:
            try:
                s = stat_func(fmodel, dense_agg_feat_gen, agg_true_theta, possible_agg_mask)
                if s is not None:
                    statistics.append(s)
            except ValueError as e:
                print(e)
    return statistics

def _get_agg_pearson(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask):
    return scipy.stats.pearsonr(
        agg_true_theta[possible_agg_mask],
        fmodel.agg_refit_theta[possible_agg_mask],
    )[0]

def _get_agg_kendall(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask):
    return scipy.stats.kendalltau(
        agg_true_theta[possible_agg_mask],
        fmodel.agg_refit_theta[possible_agg_mask],
    )[0]

def _get_agg_norm_diff(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask):
    tot_elems = np.sum(possible_agg_mask)

    # Subtract the median
    possible_agg_true_theta = agg_true_theta[possible_agg_mask] - np.median(agg_true_theta[possible_agg_mask])
    possible_agg_refit_theta = fmodel.agg_refit_theta[possible_agg_mask] - np.median(fmodel.agg_refit_theta[possible_agg_mask])

    return np.linalg.norm(possible_agg_refit_theta - possible_agg_true_theta)/np.linalg.norm(possible_agg_true_theta)

def _get_agg_coverage_all(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask):
    return _get_agg_coverage(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask)

def _get_agg_coverage_negative(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask):
    compare_func = lambda x: x < 0
    return _get_agg_coverage(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, compare_func)

def _get_agg_coverage_positive(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask):
    compare_func = lambda x: x > 0 #np.percentile(x, 75)
    return _get_agg_coverage(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, compare_func)

def _get_agg_coverage_zero(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask):
    compare_func = lambda x: x == 0
    return _get_agg_coverage(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, compare_func)

def _get_agg_coverage(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, compare_func=None):
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
            np.linalg.pinv(fmodel.variance_est),
            col_idx=col_idx + 1 if agg_true_theta.shape[1] == NUM_NUCLEOTIDES else 0,
            zstat=1.96,
        )
        agg_fitted_thetas.append((agg_fitted_theta, agg_fitted_lower, agg_fitted_upper))

    all_theta = np.vstack([t[0] for t in agg_fitted_thetas])
    med_theta = np.median(all_theta[all_theta != -np.inf])

    agg_true_theta -= np.median(agg_true_theta[agg_true_theta != -np.inf])
    for col_idx, (agg_fitted_theta, agg_fitted_lower, agg_fitted_upper) in enumerate(agg_fitted_thetas):
        inf_mask = agg_fitted_theta != -np.inf
        agg_fitted_upper = agg_fitted_upper[inf_mask]
        agg_fitted_lower = agg_fitted_lower[inf_mask]
        agg_fitted_theta = agg_fitted_theta[inf_mask]
        agg_fitted_lower -= med_theta
        agg_fitted_upper -= med_theta
        agg_true_theta_col = agg_true_theta[inf_mask,col_idx]
        comparison_mask = np.abs(agg_fitted_lower - agg_fitted_upper) > 1e-5 # only look at things with confidence intervals
        if compare_func is not None:
            comparison_mask = np.ones(agg_fitted_lower.shape, dtype=bool)
            comparison_mask &= compare_func(agg_true_theta_col)

        agg_fitted_lower_small = agg_fitted_lower[comparison_mask]
        agg_fitted_upper_small = agg_fitted_upper[comparison_mask]
        agg_true_theta_small = agg_true_theta_col[comparison_mask]
        #print np.hstack([
        #    agg_fitted_lower_small.reshape((agg_fitted_lower_small.size, 1)),
        #    agg_true_theta_small.reshape((agg_true_theta_small.size, 1)),
        #    agg_fitted_upper_small.reshape((agg_fitted_upper_small.size, 1))])
        num_covered = np.sum((agg_fitted_lower_small - 1e-5 <= agg_true_theta_small) & (agg_fitted_upper_small + 1e-5 >= agg_true_theta_small))
        tot_covered += num_covered
        num_considered = np.sum(comparison_mask)
        tot_considered += num_considered

    return tot_covered/float(tot_considered) if tot_considered > 0 else None

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

    return np.array(true_model_agg)

def _get_stat_func(stat):
    STAT_FUNC_DICT = {
        "norm": _get_agg_norm_diff,
        "coverage": _get_agg_coverage_all,
        "coverage_pos": _get_agg_coverage_positive,
        "coverage_neg": _get_agg_coverage_negative,
        "coverage_zero": _get_agg_coverage_zero,
        "pearson": _get_agg_pearson,
        "kendall": _get_agg_kendall,
    }
    return STAT_FUNC_DICT[stat]

def main(args=sys.argv[1:]):
    args = parse_args()

    STAT_LABEL = {
        "coverage": "Coverage",
        "coverage_pos": "Coverage of Positive Theta",
        "coverage_neg": "Coverage of Negative Theta",
        "coverage_zero": "Coverage of Zero Theta",
        "pearson": "Pearson",
        "kendall": "Kendall's Tau",
        "norm": "Relative theta error",
    }
    MODEL_LABEL = {
        "3_targetTrue": "3-mer per-target",
        "3_targetFalse": "3-mer",
        "2_3_targetFalse": "2,3-mer",
    }
    LINE_STYLES = ["solid", "dashed", "dotted"]

    COLS = ["model_type","Percent effect size","Percent nonzeros","Number of samples","seed"] + [STAT_LABEL[s] for s in args.stats]
    all_df = pd.DataFrame(columns=COLS)
    for i, model_type in enumerate(args.model_types):
        for eff_size in args.effect_sizes:
            for sparsity in args.sparsities:
                for nsamples in args.sample_sizes:
                    eff_same = eff_size == args.effect_sizes[1]
                    sparse_same = sparsity == args.sparsities[1]
                    samples_same = nsamples == args.sample_sizes[1]
                    if eff_same + sparse_same + samples_same >= 2:
                        for seed in range(args.reps):
                            fitted_filename = args.fitted_models % (model_type, sparsity, eff_size, nsamples, seed)
                            fitted_model = load_fitted_model(fitted_filename, args.agg_motif_len, args.agg_pos_mutating)
                            tmodel_file = args.true_models % (model_type, sparsity, eff_size)
                            true_model = _load_true_model(
                                tmodel_file,
                                args.agg_motif_len,
                                args.agg_pos_mutating,
                                fitted_model.motif_lens,
                                fitted_model.positions_mutating,
                            )

                            tmp_df = pd.DataFrame(columns=COLS)
                            tmp_dat = {
                                "model_type": model_type,
                                "Percent effect size": int(eff_size),
                                "Percent nonzeros": int(sparsity),
                                "Number of samples": int(nsamples),
                                "seed":seed,
                            }
                            for stat in args.stats:
                                stat_func = _get_stat_func(stat)
                                samm_statistics = _collect_statistics(
                                    [fitted_model],
                                    args,
                                    true_model,
                                    stat_func,
                                )
                                if len(samm_statistics):
                                    tmp_dat[STAT_LABEL[stat]] = samm_statistics[0]
                            tmp_df = tmp_df.append(tmp_dat, ignore_index=True)
                            all_df = pd.concat((all_df, tmp_df))

    sns.set_context(context="paper", font_scale=1.4)
    sns_plot = sns.PairGrid(
        data=all_df,
        hue="model_type",
        x_vars=["Number of samples", "Percent nonzeros", "Percent effect size"],
        y_vars=[STAT_LABEL[s] for s in args.stats],
        hue_kws={"linestyles":["-","--",":"]},
        palette="Set2",
    )

    sns_plot.map(sns.pointplot, linestyles=["-","--",":"], markers=".", scale=1, errwidth=1, dodge=True, capsize=0.2)
    # majro hack cause seaborn is broken i think
    col_palette = sns.color_palette("Set2", 3)
    p1 = matplotlib.lines.Line2D([0], [0], linestyle='-', c=col_palette[0], label=args.model_types[0])
    if len(args.model_types) > 1:
        p2 = matplotlib.lines.Line2D([0], [0], linestyle='--', c=col_palette[1], label=args.model_types[1])
        p3 = matplotlib.lines.Line2D([0], [0], linestyle=':', c=col_palette[2],  label=args.model_types[2])
        proxies = [p1,p2,p3]
        descriptions = ["3-mer", "3-mer per-target", "2,3-mer"]
        plt.gca().legend(proxies, descriptions, numpoints=1, markerscale=2, bbox_to_anchor=(1.05,0.4))

    sns_plot.savefig(args.outfile)

if __name__ == "__main__":
    main(sys.argv[1:])
