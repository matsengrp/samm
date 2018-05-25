import pickle
import numpy as np
import argparse

from common import *
from read_data import *
from fit_logistic_model import LogisticModel
from plot_simulation_section import _collect_statistics, _get_agg_pearson, _get_agg_kendall, _get_agg_norm_diff

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-shazam-mut',
        type=str,
        help='shazam mutability csv files prefix')
    parser.add_argument('--in-shazam-sub',
        type=str,
        help='shazam substitution csv files prefix',
        default=None)
    parser.add_argument('--in-logistic',
        type=str,
        help='logistic pkl files prefix')
    parser.add_argument('--in-samm',
        type=str,
        help='samm pkl files  prefix')
    parser.add_argument('--true-models',
        type=str,
        help='true model pkl  prefix',
        default=None)
    parser.add_argument('--seeds',
        type=str,
        help="comma separated seed strings")
    parser.add_argument('--agg-motif-len',
        type=int,
        default=5)
    parser.add_argument('--agg-pos-mutating',
        type=int,
        default=2)
    parser.add_argument('--wide-format',
        action='store_true',
        help='flag for different kinds of shazam output files',
    )
    args = parser.parse_args()
    args.seeds = args.seeds.split(",")
    args.shazam_mut_files = [args.in_shazam_mut % s for s in args.seeds]
    args.shazam_sub_files = [args.in_shazam_sub % s for s in args.seeds]
    args.in_logistic = [args.in_logistic % s for s in args.seeds]
    args.in_samm = [args.in_samm % s for s in args.seeds]
    args.true_models = [args.true_models % s for s in args.seeds]
    return args

class ShazamModel:
   def __init__(self, agg_motif_len, shazam_mut_csv, shazam_sub_csv, wide_format=False):
        """
        @param agg_motif_len: motif length
        @param shazam_mut_csv: csv of mutabilities
        @param shazam_sub_csv: csv of substitution probabilities
        @param wide_format: shazam can yield two types of files: wide format and tall format;
            wide format comes directly from createMutabilityMatrix, etc.
        """
        refit_theta = get_shazam_theta(shazam_mut_csv, shazam_sub_csv, wide_format=wide_format)
        if refit_theta.shape[1] > 1:
            self.agg_refit_theta = refit_theta[:,0:1] + refit_theta[:,1:]
        else:
            self.agg_refit_theta = refit_theta[:,0:1]
        print "SHAZAM contains nan in the estimates", np.any(np.isnan(self.agg_refit_theta))

def main(args=sys.argv[1:]):
    args = parse_args()

    shazam_models = [
        ShazamModel(
            args.agg_motif_len,
            shazam_mut_csv,
            shazam_sub_csv,
            args.wide_format,
        ) for shazam_mut_csv, shazam_sub_csv in
        zip(args.shazam_mut_files, args.shazam_sub_files)
    ]

    samm_models = [
        load_fitted_model(
            samm_pkl,
            keep_col0=False,
            add_targets=True,
        ) for samm_pkl in args.in_samm
    ]

    logistic_models = [load_logistic_model(logistic_pkl) for logistic_pkl in args.in_logistic]

    true_models = [
        load_true_model(tmodel_file) for tmodel_file in args.true_models
    ]

    stat_funcs = [_get_agg_norm_diff, _get_agg_kendall, _get_agg_pearson]
    stat_res = [{"shazam":[], "samm":[], "logistic": []} for i in stat_funcs]
    for true_m, samm_m, shazam_m, logistic_m in zip(true_models, samm_models, shazam_models, logistic_models):
        for stat_i, stat_f in enumerate(stat_funcs):
           try:
               stat_shazam = _collect_statistics([shazam_m], args, true_m, stat_f)
               if np.isfinite(stat_shazam):
                   stat_res[stat_i]["shazam"].append(stat_shazam)
               else:
                   raise ValueError("infinite value for statistic")
           except Exception as e:
               print "WARNING: Shazam has a bad estimate!"
           stat_samm = _collect_statistics([samm_m], args, true_m, stat_f)
           stat_res[stat_i]["samm"].append(stat_samm)
           stat_logistic = _collect_statistics([logistic_m], args, true_m, stat_f)
           stat_res[stat_i]["logistic"].append(stat_logistic)

    for stat_r, stat_func in zip(stat_res, stat_funcs):
        print "==================="
        print stat_func.__name__, "mean (se)", len(stat_r["shazam"])
        if len(stat_r["shazam"]):
            print "shazam", np.mean(stat_r["shazam"]), "(%f)" % np.sqrt(np.var(stat_r["shazam"])/len(stat_r["shazam"]))
        else:
            print "shazam", "all estimates are broken"
        print "samm", np.mean(stat_r["samm"]), "(%f)" % np.sqrt(np.var(stat_r["samm"])/len(stat_r["samm"]))
        if len(stat_r["shazam"]) != len(stat_r["samm"]):
            print "WARNING: Shazam had some bad estimates!"
        print stat_func.__name__, "mean (se)", len(stat_r["logistic"])
        print "logistic", np.mean(stat_r["logistic"]), "(%f)" % np.sqrt(np.var(stat_r["logistic"])/len(stat_r["logistic"]))

if __name__ == "__main__":
    main(sys.argv[1:])
