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
       refit_theta_5cols = get_shazam_theta(agg_motif_len, shazam_mut_csv, shazam_sub_csv)
       self.agg_refit_theta = refit_theta_5cols[:,0:1] + refit_theta_5cols[:,1:]

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
    for true_m, samm_m, shazam_m in zip(true_models, samm_models, shazam_models):
       print true_m - np.median(true_m)
       print samm_m.agg_refit_theta - np.median(samm_m.agg_refit_theta)
       print shazam_m.agg_refit_theta - np.median(shazam_m.agg_refit_theta)
       print "shazam", _collect_statistics([shazam_m], args, None, true_m, stat_funcs[2])
       print "samm", _collect_statistics([samm_m], args, None, true_m, stat_funcs[2])

if __name__ == "__main__":
    main(sys.argv[1:])


