import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import scipy.stats
import os.path
import copy

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from common import ZERO_THRES, NUM_NUCLEOTIDES, ZSCORE_95
from read_data import load_fitted_model

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--fitted-models',
        type=str,
        default="simulation_section/_output/%s/nonzero%s/effect_size_%s/samples%s/%02d/samm/fitted.pkl",
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
        default="norm,kendall,coverage,discovered",
    )
    parser.add_argument('--outplot',
        type=str,
        default="_output/simulation3mer.pdf")
    parser.add_argument('--outcsv',
        type=str,
        default="_output/simulation3mer.csv",
        help="csv file to dump all simulation output")
    parser.add_argument('--outtable',
        type=str,
        default="_output/simulation3mer.txt",
        help="txt file for latex table output")
    parser.add_argument('--rerun',
        action="store_true",
        help="recompute statistics")

    args = parser.parse_args()
    args.model_types = args.model_types.split(",")
    args.stats = args.stats.split(",")
    args.sample_sizes = args.sample_sizes.split(",")
    args.effect_sizes = args.effect_sizes.split(",")
    args.sparsities = args.sparsities.split(",")
    return args

STAT_LABEL = {
    "coverage": ["Coverage"],
    "coverage_pos": ["Coverage of Positive Theta"],
    "coverage_neg": ["Coverage of Negative Theta"],
    "coverage_zero": ["Coverage of Zero Theta"],
    "pearson": ["Pearson"],
    "kendall": ["Kendall's Tau"],
    "norm": ["Relative theta error"],
    "discovered": ["Num False Positive; Num Discovered"],
}
FIT_TYPES = [
    'refit',
    'penalized',
]

def _collect_statistics(fitted_models, args, true_thetas, stat, fit_type):
    statistics = []
    stat_func = _get_stat_func(stat)
    for fmodel in fitted_models:
        if fmodel is not None:
            if stat == 'discovered':
                # don't use aggregate fit for false discovery rate
                feat_gen = fmodel.refit_feature_generator
                feat_gen.update_feats_after_removing([])
                true_theta = true_thetas[1]
                possible_mask = feat_gen.get_possible_motifs_to_targets(
                    true_theta.shape,
                )
            else:
                feat_gen = HierarchicalMotifFeatureGenerator(
                    motif_lens=[args.agg_motif_len],
                    left_motif_flank_len_list=[[args.agg_pos_mutating]],
                )
                true_theta = true_thetas[0]
                possible_mask = feat_gen.get_possible_motifs_to_targets(
                    true_theta.shape,
                )
            try:
                s = stat_func(fmodel, feat_gen, true_theta, possible_mask, fit_type)
                if s is not None:
                    statistics.append(s)
            except ValueError as e:
                print(e)
    return statistics

def _get_discovered_corrected(fmodel, full_feat_generator, true_theta, possible_mask, fit_type):
    """
    Get a "corrected" number of false positive features, i.e., those whose CIs exclude zero but whose true theta was zero
    and number of discovered features, i.e., those whose CIs exclude zero
    """
    if fit_type == 'penalized':
        # No confidence intervals for penalized model
        return 0., 0.

    # calculate which nonzeros do not have CIs overlapping zero
    theta = fmodel.refit_theta

    cov_mat = fmodel.variance_est
    if np.any(np.diag(cov_mat) < 0):
        return np.nan, np.nan

    standard_errors = np.sqrt(np.diag(cov_mat))
    theta_mask = fmodel.refit_possible_theta_mask & ~fmodel.model_masks.zero_theta_mask_refit
    theta_mask_flat = theta_mask.reshape((theta_mask.size,), order="F")
    theta_flat = theta.reshape((theta.size,), order="F")
    theta_flat = theta_flat[theta_mask_flat]

    true_theta_mask = possible_mask & ~fmodel.model_masks.zeroed_thetas
    true_theta_mask_flat = true_theta_mask.reshape((true_theta_mask.size,), order="F")
    true_theta_flat = true_theta.reshape((true_theta.size,), order="F")
    true_theta_flat = true_theta_flat[true_theta_mask_flat]

    conf_int_low = theta_flat - ZSCORE_95 * standard_errors
    conf_int_upper = theta_flat + ZSCORE_95 * standard_errors

    true_zeros_fit_nonzero = np.abs(true_theta_flat) < ZERO_THRES
    fit_sig_nonzeros = np.logical_or(conf_int_low > ZERO_THRES, conf_int_upper < -ZERO_THRES)

    num_discovered = np.sum(fit_sig_nonzeros)
    num_false_positive = np.sum(np.logical_and(true_zeros_fit_nonzero, fit_sig_nonzeros))

    return float(num_false_positive), float(num_discovered)

def _get_agg_pearson(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, fit_type):
    if fit_type == 'refit':
        agg_theta = fmodel.agg_refit_theta[possible_agg_mask]
    else:
        agg_theta = fmodel.agg_penalized_theta[possible_agg_mask]
    return scipy.stats.pearsonr(
        agg_true_theta[possible_agg_mask],
        agg_theta,
    )[0]

def _get_agg_kendall(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, fit_type):
    if fit_type == 'refit':
        agg_theta = fmodel.agg_refit_theta[possible_agg_mask]
    else:
        agg_theta = fmodel.agg_penalized_theta[possible_agg_mask]
    return scipy.stats.kendalltau(
        agg_true_theta[possible_agg_mask],
        agg_theta,
    )[0]

def _get_agg_norm_diff(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, fit_type):
    if fit_type == 'refit':
        agg_theta = fmodel.agg_refit_theta[possible_agg_mask]
    else:
        agg_theta = fmodel.agg_penalized_theta[possible_agg_mask]

    # Subtract the median
    possible_agg_true_theta = agg_true_theta[possible_agg_mask] - np.median(agg_true_theta[possible_agg_mask])
    possible_agg_theta = agg_theta - np.median(agg_theta)

    return np.linalg.norm(possible_agg_theta - possible_agg_true_theta)/np.linalg.norm(possible_agg_true_theta)

def _get_agg_coverage_all(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, fit_type):
    return _get_agg_coverage(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, None, fit_type)

def _get_agg_coverage_negative(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, fit_type):
    compare_func = lambda x: x < 0
    return _get_agg_coverage(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, compare_func, fit_type)

def _get_agg_coverage_positive(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, fit_type):
    compare_func = lambda x: x > 0 #np.percentile(x, 75)
    return _get_agg_coverage(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, compare_func, fit_type)

def _get_agg_coverage_zero(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, fit_type):
    compare_func = lambda x: x == 0
    return _get_agg_coverage(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, compare_func, fit_type)

def _get_agg_coverage(fmodel, full_feat_generator, agg_true_theta, possible_agg_mask, compare_func=None, fit_type='refit'):
    if fit_type == 'penalized':
        # No confidence intervals for penalized model
        return 0.0

    hier_feat_gen = fmodel.refit_feature_generator

    # calculate coverage of groups of theta values
    agg_coverage = []
    tot_covered = 0
    tot_considered = 0
    agg_fitted_thetas = []
    for col_idx in range(agg_true_theta.shape[1]):
        agg_fitted_theta, agg_fitted_lower, agg_fitted_upper = hier_feat_gen.combine_thetas_and_get_conf_int(
            fmodel.refit_theta,
            fmodel.variance_est,
            col_idx=col_idx + 1 if agg_true_theta.shape[1] == NUM_NUCLEOTIDES else 0,
            zstat=ZSCORE_95,
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
        num_covered = np.sum((agg_fitted_lower_small - 1e-5 <= agg_true_theta_small) & (agg_fitted_upper_small + 1e-5 >= agg_true_theta_small))
        tot_covered += num_covered
        num_considered = np.sum(comparison_mask)
        tot_considered += num_considered

    return tot_covered/float(tot_considered) if tot_considered > 0 else None

def _load_true_model(file_name):
    with open(file_name, "r") as f:
        true_model_agg, true_model = pickle.load(f)

    return np.array(true_model_agg), np.array(true_model)

def _get_stat_func(stat):
    STAT_FUNC_DICT = {
        "norm": _get_agg_norm_diff,
        "coverage": _get_agg_coverage_all,
        "coverage_pos": _get_agg_coverage_positive,
        "coverage_neg": _get_agg_coverage_negative,
        "coverage_zero": _get_agg_coverage_zero,
        "pearson": _get_agg_pearson,
        "kendall": _get_agg_kendall,
        "discovered": _get_discovered_corrected,
    }
    return STAT_FUNC_DICT[stat]

def _build_dataframe(args):
    """
    Compute stats in dataframe if we haven't already... or if we ask it to
    """
    if os.path.exists(args.outcsv) and not args.rerun:
        with open(args.outcsv, 'r') as f:
            all_df = pd.read_csv(f)
    else:
        COLS = ["model_type","Percent effect size","Percent nonzeros","Number of samples","seed"] + [var for stat in args.stats for var in STAT_LABEL[stat]]
        all_df = pd.DataFrame(columns=COLS)
        for model_type in args.model_types:
            for eff_size in args.effect_sizes:
                for sparsity in args.sparsities:
                    tmodel_file = args.true_models % (model_type, sparsity, eff_size)
                    if not os.path.isfile(tmodel_file):
                        continue
                    # if file doesn't exist then continue
                    true_model_agg, true_model = _load_true_model(tmodel_file)
                    for nsamples in args.sample_sizes:
                        eff_same = eff_size == args.effect_sizes[1]
                        sparse_same = sparsity == args.sparsities[1]
                        samples_same = nsamples == args.sample_sizes[1]
                        for seed in range(args.reps):
                            fitted_filename = args.fitted_models % (model_type, sparsity, eff_size, nsamples, seed)
                            # if file doesn't exist then continue
                            if not os.path.isfile(fitted_filename):
                                continue
                            fitted_model = load_fitted_model(fitted_filename, keep_col0=False)

                            for fit_type in FIT_TYPES:
                                tmp_df = pd.DataFrame(columns=COLS)
                                tmp_dat = {
                                    "model_type": model_type,
                                    "Percent effect size": int(eff_size),
                                    "Percent nonzeros": int(sparsity),
                                    "Number of samples": int(nsamples),
                                    "seed":seed,
                                    "to_plot": eff_same + sparse_same + samples_same >= 2,
                                    "fit_type": fit_type
                                }
                                for stat in args.stats:
                                    samm_statistics = _collect_statistics(
                                        [fitted_model],
                                        args,
                                        [true_model_agg, true_model],
                                        stat,
                                        fit_type,
                                    )
                                    if len(samm_statistics):
                                        for var in STAT_LABEL[stat]:
                                            tmp_dat[var] = samm_statistics[0]
                                tmp_df = tmp_df.append(tmp_dat, ignore_index=True)
                                all_df = pd.concat((all_df, tmp_df))

        with open(args.outcsv, 'w') as f:
            all_df.to_csv(f)

    return all_df

def print_fcn(x):
    outx = x.apply(pd.Series)
    if outx.shape[1] > 1:
        return '%.1f; %.1f (%.1f; %.1f)' % (np.mean(outx[0]), np.mean(outx[1]), np.std(outx[0]), np.std(outx[1]))
    else:
        return '%.1f (%.1f)' % (np.mean(outx), np.std(outx))

def main(args=sys.argv[1:]):
    args = parse_args()

    all_df = _build_dataframe(args)

    MODEL_LABEL = {
        "3_targetTrue": "3-mer per-target",
        "3_targetFalse": "3-mer",
        "2_3_targetFalse": "2,3-mer",
    }
    SETTINGS = [
        'Percent effect size',
        'Percent nonzeros',
        'Number of samples',
    ]
    group_cols = ['model_type'] + SETTINGS
    all_df[SETTINGS] = all_df[SETTINGS].astype(int)
    all_df['model_type'] = [MODEL_LABEL[val] for val in all_df['model_type']]

    sns.set_context(context="paper", font_scale=1.4)
    sns_plot = sns.PairGrid(
        data=all_df[(all_df['fit_type']=='refit') & (all_df['to_plot'])],
        hue="model_type",
        x_vars=SETTINGS,
        y_vars=[var for stat in args.stats if stat not in ['discovered'] for var in STAT_LABEL[stat]],
        hue_kws={"linestyles":["-","--",":"]},
        palette="Set2",
    )

    all_df['Coverage'] *= 100
    out_str = ''
    for fit_type in FIT_TYPES:
        out_str += fit_type+'\n'
        print_df = all_df[all_df['fit_type']==fit_type].groupby(group_cols)[[var for stat in args.stats if stat not in ['coverage', 'discovered'] for var in STAT_LABEL[stat]]].agg(lambda x: '%.3f (%.3f)' % (np.mean(x), np.std(x)))
        print_df.reset_index(level='model_type', inplace=True)
        out_str += print_df.pivot(columns='model_type').to_latex()

    print_df = all_df[all_df['fit_type']=='refit'].groupby(group_cols)[[var for stat in args.stats if stat in ['coverage', 'discovered'] for var in STAT_LABEL[stat]]].agg(lambda x: print_fcn(x))
    print_df.reset_index(level='model_type', inplace=True)
    out_str += print_df.pivot(columns='model_type').to_latex()

    print_df = all_df[all_df['fit_type']=='refit'].groupby(group_cols)[[STAT_LABEL['coverage'][0]]].agg(lambda x: '%d, %d' % (np.sum(pd.isnull(x)), len(x)))
    print_df.reset_index(level='model_type', inplace=True)
    out_str += print_df.pivot(columns='model_type').to_latex()

    with open(args.outtable, 'w') as f:
        f.write(out_str)

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

    sns_plot.savefig(args.outplot)

if __name__ == "__main__":
    main(sys.argv[1:])
