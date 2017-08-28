"""
Assess how bad estimates are if we assume positions mutate at most once
"""

import numpy as np
import argparse

from common import *
from read_data import *
from plot_simulation_section import _collect_statistics, _get_agg_pearson, _get_agg_kendall, _get_agg_norm_diff

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samm-one',
        type=str,
        help='comma separated samm pkl files, fit for single mut')
    parser.add_argument('--samm-mult',
        type=str,
        help='comma separated samm pkl files, fit for multiple mut (assumes single mut)')
    parser.add_argument('--true-models',
        type=str,
        help='true model pkl')
    parser.add_argument('--agg-motif-len',
        type=int,
        default=3)
    parser.add_argument('--agg-pos-mutating',
        type=int,
        default=1)
    args = parser.parse_args()
    args.samm_one = args.samm_one.split(',')
    args.samm_mult = args.samm_mult.split(',')
    args.true_models = args.true_models.split(',')
    return args

def main(args=sys.argv[1:]):
    args = parse_args()

    samm_models_one = [
        load_fitted_model(
            samm_pkl,
            args.agg_motif_len,
            args.agg_pos_mutating,
            keep_col0=False,
            add_targets=True,
        ) for samm_pkl in args.samm_one
    ]
    samm_models_mult = [
        load_fitted_model(
            samm_pkl,
            args.agg_motif_len,
            args.agg_pos_mutating,
            keep_col0=False,
            add_targets=True,
        ) for samm_pkl in args.samm_mult
    ]
    example_model = samm_models_one[0]
    true_models = [
        load_true_model(tmodel_file) for tmodel_file in args.true_models
    ]

    stat_funcs = [_get_agg_norm_diff, _get_agg_kendall, _get_agg_pearson]
    stat_res = [{"one":[], "mult":[]} for i in stat_funcs]
    for true_m, samm_one, samm_mult in zip(true_models, samm_models_one, samm_models_mult):
        for stat_i, stat_f in enumerate(stat_funcs):
           stat_samm_one = _collect_statistics([samm_one], args, None, true_m, stat_f)
           stat_samm_mult = _collect_statistics([samm_mult], args, None, true_m, stat_f)
           stat_res[stat_i]["one"].append(stat_samm_one)
           stat_res[stat_i]["mult"].append(stat_samm_mult)

    for stat_r, stat_func in zip(stat_res, stat_funcs):
        print stat_func.__name__, "mean (se)"
        num_samples = len(stat_r["one"])
        print "single mut:", np.mean(stat_r["one"]), "(%f)" % np.sqrt(np.var(stat_r["one"])/num_samples)
        print "multiple mut:", np.mean(stat_r["mult"]), "(%f)" % np.sqrt(np.var(stat_r["mult"])/num_samples)

if __name__ == "__main__":
    main(sys.argv[1:])
