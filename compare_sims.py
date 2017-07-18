import pickle
import numpy as np
import scipy.stats

from read_data import *
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator

MOTIF_LEN = 5
MUT_POS = 2
NSEEDS = 10

SAMM_FILE_STR = 'simulated_shazam_vs_samm/_output/3_5_target%s/0%d/motif-5-flank-2/%s/fitted.pkl'
SHAZAM_MUT_STR = 'simulated_shazam_vs_samm/_output/3_5_target%s/0%d/motif-5-flank-2/%s/fitted_shazam_mut.csv'
SHAZAM_TARGET_STR = 'simulated_shazam_vs_samm/_output/3_5_target%s/0%d/motif-5-flank-2/%s/fitted_shazam_target.csv'
TRUE_FILE_STR = 'simulated_shazam_vs_samm/_output/3_5_target%s/0%d/true_model.pkl'

def _get_agg_pearson(agg_fit_theta, full_feat_generator, agg_true_theta, possible_agg_mask):
    return scipy.stats.pearsonr(
        agg_true_theta[possible_agg_mask],
        agg_fit_theta[possible_agg_mask],
    )[0]

def _get_agg_norm_diff(agg_fit_theta, full_feat_generator, agg_true_theta, possible_agg_mask):
    tot_elems = np.sum(possible_agg_mask)

    # Subtract the median
    possible_agg_true_theta = agg_true_theta[possible_agg_mask] - np.median(agg_true_theta[possible_agg_mask])
    possible_agg_refit_theta = agg_fit_theta[possible_agg_mask] - np.median(agg_fit_theta[possible_agg_mask])

    return np.linalg.norm(possible_agg_refit_theta - possible_agg_true_theta)/np.linalg.norm(possible_agg_true_theta)

def _collect_statistics(fitted_models, agg_thetas, stat_func, agg_motif_len, agg_pos_mutating):
    dense_agg_feat_gen = HierarchicalMotifFeatureGenerator(
        motif_lens=[agg_motif_len],
        left_motif_flank_len_list=[[agg_pos_mutating]],
    )

    statistics = []
    for fmodel, agg_true_theta in zip(fitted_models, agg_thetas):
        possible_agg_mask = get_possible_motifs_to_targets(
            dense_agg_feat_gen.motif_list,
            mask_shape=agg_true_theta.shape,
            mutating_pos_list=[agg_pos_mutating] * dense_agg_feat_gen.feature_vec_len,
        )
        if fmodel is not None:
            try:
                s = stat_func(fmodel, dense_agg_feat_gen, agg_true_theta, possible_agg_mask)
                if s is not None:
                    statistics.append(s)
            except ValueError as e:
                print(e)
    return statistics

def _get_stat_func(stat):
    STAT_FUNC_DICT = {
        "norm": _get_agg_norm_diff,
        "pearson": _get_agg_pearson,
    }
    return STAT_FUNC_DICT[stat]

fitted_models = {}
true_models = {}
model_types = []
for per_target in ['False', 'True']:
    fitted_models['samm_%s' % per_target] = [
        load_fitted_model(
            SAMM_FILE_STR % (per_target, seed, per_target),
            MOTIF_LEN,
            MUT_POS,
            add_targets=True
        ).agg_refit_theta for seed in range(NSEEDS)
    ]
    
    if per_target == "False":
        fitted_models['shazam_%s' % per_target] = [
            get_shazam_theta(
                MOTIF_LEN,
                SHAZAM_MUT_STR % (per_target, seed, per_target),
            ) for seed in range(NSEEDS)
        ]
    else:
        fitted_models['shazam_%s' % per_target] = [
            get_shazam_target(
                MOTIF_LEN,
                SHAZAM_TARGET_STR % (per_target, seed, per_target),
            ) for seed in range(NSEEDS)
        ]
    
    true_models['%s' % per_target] = []
    for seed in range(NSEEDS):
        true_pkl = TRUE_FILE_STR % (per_target, seed)
        with open(true_pkl, 'r') as f:
            agg_true_theta, raw_true_theta = pickle.load(f)
            true_models['%s' % per_target].append(agg_true_theta)
    
    stats = ['norm', 'pearson']
    model_types += ['samm_%s' % per_target, 'shazam_%s' % per_target]

samm_means = {stat: {mtype : [] for mtype in set(model_types)} for stat in stats}
samm_se = {stat: {mtype : [] for mtype in set(model_types)} for stat in stats}
for stat in ['norm', 'pearson']:
    stat_func = _get_stat_func(stat)

    for mtype in model_types:
        true_mtype = mtype.split('_')[-1]
        samm_statistics = _collect_statistics(
            fitted_models[mtype],
            true_models[true_mtype],
            stat_func,
            MOTIF_LEN,
            MUT_POS
        )
        mean = np.mean(samm_statistics)
        se = 1.96 * np.sqrt(np.var(samm_statistics)/len(samm_statistics))
        samm_means[stat][mtype].append(mean)
        samm_se[stat][mtype].append(se)
        print "MEAN", stat, mtype, "%.3f" % mean, "(%.3f)" % se
        if se > 0.1:
            print samm_statistics

