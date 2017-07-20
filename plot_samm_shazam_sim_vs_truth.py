import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import matplotlib.lines as mlines

sns.set(style="white")

from common import *
from read_data import *
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator

# some constants---pass these through as variables?
NSEEDS = 10
MOTIF_LEN = 5
MUT_POS = 2
TRUE_MODEL_STR = 'simulated_shazam_vs_samm/_output/3_5_target%s/0%d/true_model.pkl'
SAMM_MODEL_STR = 'simulated_shazam_vs_samm/_output/3_5_target%s/0%d/motif-5-flank-2/%s/fitted.pkl'
SHAZAM_MUT_STR = 'simulated_shazam_vs_samm/_output/3_5_target%s/0%d/motif-5-flank-2/%s/fitted_shazam_mut.csv'
SHAZAM_SUB_STR = 'simulated_shazam_vs_samm/_output/3_5_target%s/0%d/motif-5-flank-2/%s/fitted_shazam_target.csv'
THETA_MIN = -4
THETA_MAX = 4
# what step size to use to divide true theta by effect size
STEP = 1.5

def raw_diff(agg_fit_theta, agg_true_theta):
    possible_agg_true_theta = agg_true_theta - np.median(agg_true_theta)
    possible_agg_refit_theta = agg_fit_theta - np.median(agg_fit_theta)

    return possible_agg_refit_theta - possible_agg_true_theta

def mean_raw_diff(agg_fit_theta, agg_true_theta):
    possible_agg_true_theta = agg_true_theta - np.median(agg_true_theta)
    possible_agg_refit_theta = agg_fit_theta - np.median(agg_fit_theta)

    return np.mean(possible_agg_refit_theta - possible_agg_true_theta)

def pearson(agg_fit_theta, agg_true_theta):
    return scipy.stats.pearsonr(
        agg_true_theta,
        agg_fit_theta,
    )[0]

def norm_diff(agg_fit_theta, agg_true_theta):
    possible_agg_true_theta = agg_true_theta - np.median(agg_true_theta)
    possible_agg_refit_theta = agg_fit_theta - np.median(agg_fit_theta)

    return np.linalg.norm(possible_agg_refit_theta - possible_agg_true_theta)/np.linalg.norm(possible_agg_true_theta)

def _plot_single_effect_size_overall(all_df, true_categories, per_target='False', fname=''):

    all_df = all_df[all_df['per_target'] == per_target]
    all_df = all_df[['samm_raw_diff', 'shazam_raw_diff', 'true theta size']]
    all_df.rename(index=str, columns={'samm_raw_diff': 'samm', 'shazam_raw_diff': 'SHazaM', 'true theta size': 'true theta size'}, inplace=True)
    melt_df = pd.melt(all_df, id_vars=['true theta size'])

    sns_plot = sns.factorplot(
        x='true theta size',
        y='value',
        hue='variable',
        data=melt_df,
        kind="box",
        palette="Set1",
        order=true_categories.categories,
        legend=False,
    )
    sns_plot.set(ylabel='fitted minus truth')
    sns_plot.set(xlabel="true theta size")
    x = sns_plot.axes[0,0].get_xlim()
    sns_plot.axes[0,0].plot(x, len(x) * [0], 'k--', alpha=.4)
    plt.legend(loc='upper right', title='model')

    sns_plot.savefig(fname)

def _plot_scatter(all_df, per_target="False", fname=''):
    all_df = all_df[all_df['per_target'] == per_target]
    all_df = all_df[['samm', 'shazam', 'theta']]
    all_df.rename(index=str, columns={'samm': 'samm', 'shazam': 'SHazaM', 'theta': 'theta'}, inplace=True)
    melt_df = pd.melt(all_df, id_vars=['theta'])
    
    sns_plot = sns.lmplot(
        x="theta",
        y="value",
        hue="variable",
        data=melt_df,
        scatter_kws={'alpha':0.2},
        legend=False,
        markers=["o", "x"],
        palette="Set1",
    )
    
    model_legend = plt.legend(loc='lower right', title='model')
    xy_line = mlines.Line2D([], [], color='black', marker='', label='y=x')
    xy_legend = plt.legend(loc='upper left', handles=[xy_line])
    plt.gca().add_artist(xy_legend)
    plt.gca().add_artist(model_legend)

    sns_plot.set(ylabel='fitted theta')
    sns_plot.set(xlabel="true theta")
    
    xmin, xmax = sns_plot.axes[0, 0].get_xlim()
    ymin, ymax = sns_plot.axes[0, 0].get_ylim()
    
    lims = [
        np.max([xmin, ymin]),
        np.min([xmax, ymax]),
    ]
    
    sns_plot.axes[0,0].plot(lims, lims, 'k-')
    sns_plot.savefig(fname)
    
dense_agg_feat_gen = HierarchicalMotifFeatureGenerator(
    motif_lens=[MOTIF_LEN],
    left_motif_flank_len_list=[[MUT_POS]],
)

stats_df = pd.DataFrame(columns=['theta', 'samm_fit', 'shazam_fit', 'per_target', 'seed'])
all_df = pd.DataFrame(columns=['theta', 'samm_fit', 'shazam_fit', 'per_target', 'seed'])
for per_target in ['False', 'True']:
    for seed in range(NSEEDS):
        with open(TRUE_MODEL_STR % (per_target, seed), 'r') as f:
            theta, _ = pickle.load(f)
        model_shape = theta.shape
        possible_agg_mask = get_possible_motifs_to_targets(
            dense_agg_feat_gen.motif_list,
            mask_shape=model_shape,
            mutating_pos_list=[MUT_POS] * dense_agg_feat_gen.feature_vec_len,
        )
        theta = theta[possible_agg_mask] - np.median(theta[possible_agg_mask])
        tmp_df = pd.DataFrame()
        tmp_df['theta'] = theta
        samm = load_fitted_model(
            SAMM_MODEL_STR % (per_target, seed, per_target),
            MOTIF_LEN,
            MUT_POS,
            add_targets=True
        ).agg_refit_theta
        samm = samm[possible_agg_mask] - np.median(samm[possible_agg_mask])
        tmp_df['samm'] = samm
        if per_target == "False":
            shazam = get_shazam_theta(
                MOTIF_LEN,
                SHAZAM_MUT_STR % (per_target, seed, per_target),
            )
        else:
            shazam = get_shazam_target(
                MOTIF_LEN,
                SHAZAM_SUB_STR % (per_target, seed, per_target),
            )
        shazam = shazam[possible_agg_mask] - np.median(shazam[possible_agg_mask])
        tmp_df['shazam'] = shazam
        tmp_df['samm_raw_diff'] = raw_diff(samm, theta)
        tmp_df['shazam_raw_diff'] = raw_diff(shazam, theta)
        true_categories = pd.cut(theta.ravel(), np.arange(THETA_MIN, THETA_MAX + .1*STEP, STEP))
        tmp_df['true theta size'] = true_categories
        tmp_df['per_target'] = per_target
        tmp_df['seed'] = seed
    
        all_df = pd.concat((all_df, tmp_df))

for idx, per_target in enumerate(['False', 'True']):
    _plot_single_effect_size_overall(all_df, true_categories=true_categories, per_target=per_target, fname='_output/box_%s.svg' % per_target)

    _plot_scatter(all_df, per_target=per_target, fname='_output/scatter_%s.svg' % per_target)

