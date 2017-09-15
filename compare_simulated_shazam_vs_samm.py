import pickle
import numpy as np
import argparse

from common import *
from read_data import *
from plot_simulation_section import _collect_statistics, _get_agg_pearson, _get_agg_kendall, _get_agg_norm_diff

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-shazam-mut',
        type=str,
        help='comma separated shazam mutability csv files')
    parser.add_argument('--in-shazam-sub',
        type=str,
        help='comma separated shazam substitution csv files',
        default=None)
    parser.add_argument('--in-samm',
        type=str,
        help='comma separated samm pkl files')
    parser.add_argument('--true-models',
        type=str,
        help='true model pkl',
        default=None)
    parser.add_argument('--agg-motif-len',
        type=int,
        default=5)
    parser.add_argument('--agg-pos-mutating',
        type=int,
        default=2)
    args = parser.parse_args()
    args.shazam_mut_files = args.in_shazam_mut.split(',')
    args.shazam_sub_files = args.in_shazam_sub.split(',')
    args.in_samm = args.in_samm.split(',')
    args.true_models = args.true_models.split(',')
    return args

class ShazamModel:
   def __init__(self, agg_motif_len, shazam_mut_csv, shazam_sub_csv):
        refit_theta = get_shazam_theta(agg_motif_len, shazam_mut_csv, shazam_sub_csv)
        if refit_theta.shape[1] > 1:
            self.agg_refit_theta = refit_theta[:,0:1] + refit_theta[:,1:]
        else:
            self.agg_refit_theta = refit_theta[:,0:1]
        print "SHAZAM contains nan in the estimates", np.any(np.isnan(self.agg_refit_theta))
        #self.agg_refit_theta[np.isnan(self.agg_refit_theta)] = 0

def main(args=sys.argv[1:]):
    args = parse_args()

    shazam_models = [
        ShazamModel(
            args.agg_motif_len,
            shazam_mut_csv,
            shazam_sub_csv
        ) for shazam_mut_csv, shazam_sub_csv in
        zip(args.shazam_mut_files, args.shazam_sub_files)
    ]

    samm_models = [
        load_fitted_model(
            samm_pkl,
            args.agg_motif_len,
            args.agg_pos_mutating,
            keep_col0=False,
            add_targets=True,
        ) for samm_pkl in args.in_samm
    ]
    example_model = samm_models[0]
    true_models = [
        load_true_model(tmodel_file) for tmodel_file in args.true_models
    ]

    stat_funcs = [_get_agg_norm_diff, _get_agg_kendall, _get_agg_pearson]
    stat_res = [{"shazam":[], "samm":[]} for i in stat_funcs]
    for true_m, samm_m, shazam_m in zip(true_models, samm_models, shazam_models):
        for stat_i, stat_f in enumerate(stat_funcs):
           try:
               stat_shazam = _collect_statistics([shazam_m], args, true_m, stat_f)
               if np.isfinite(stat_shazam):
                   stat_res[stat_i]["shazam"].append(stat_shazam)
               else:
                   raise ValueError("infinite value for statistic")
           except Exception as e:
               print "WARNING: Shazam has a bad estimate!"
               continue
           stat_samm = _collect_statistics([samm_m], args, true_m, stat_f)
           stat_res[stat_i]["samm"].append(stat_samm)

    for stat_r, stat_func in zip(stat_res, stat_funcs):
        print stat_func.__name__, "mean (se)", len(stat_r["shazam"])
        if len(stat_r["shazam"]):
            print "shazam", np.mean(stat_r["shazam"]), "(%f)" % np.sqrt(np.var(stat_r["shazam"])/len(stat_r["shazam"]))
        else:
            print "shazam", "all estimates are broken"
        print "samm", np.mean(stat_r["samm"]), "(%f)" % np.sqrt(np.var(stat_r["samm"])/len(stat_r["samm"]))
        if len(stat_r["shazam"]) != len(stat_r["samm"]):
            print "WARNING: Shazam had some bad estimates!"

if __name__ == "__main__":
    main(sys.argv[1:])
