import pandas as pd
import matplotlib
matplotlib.use('pdf')
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
SIM_METHODS = ['survival'] #, 'shmulate']
TRUE_MODEL_STR = "simulated_shazam_vs_samm/_output/%s/0%d/true_model.pkl"
SAMM_MODEL_STR = "simulated_shazam_vs_samm/_output/%s/0%d/fitted.pkl"
SHAZAM_MUT_STR = "simulated_shazam_vs_samm/_output/%s/0%d/fitted_shazam_mut.csv"
SHAZAM_SUB_STR = "simulated_shazam_vs_samm/_output/%s/0%d/fitted_shazam_sub.csv"
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

def _plot_single_effect_size_overall(all_df, true_categories, fname=''):
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

def _plot_scatter(all_df, fname=''):
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
    xy_legend = plt.legend([xy_line], loc='upper left')
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

all_df = pd.DataFrame(columns=['theta', 'samm_fit', 'shazam_fit', 'sim_method', 'seed'])
for sim_method in SIM_METHODS:
    for seed in range(NSEEDS):
        with open(TRUE_MODEL_STR % (sim_method, seed), 'r') as f:
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
        print SAMM_MODEL_STR % (sim_method, seed)
        samm = load_fitted_model(
            SAMM_MODEL_STR % (sim_method, seed),
            MOTIF_LEN,
            MUT_POS,
            add_targets=True
        ).agg_refit_theta
        samm = samm[possible_agg_mask] - np.median(samm[possible_agg_mask])
        tmp_df['samm'] = samm
        shazam_raw = get_shazam_theta(
            MOTIF_LEN,
            SHAZAM_MUT_STR % (sim_method, seed),
            SHAZAM_SUB_STR % (sim_method, seed),
        )
        shazam = shazam_raw[:,0:1] + shazam_raw[:,1:]
        shazam = shazam[possible_agg_mask] - np.median(shazam[possible_agg_mask])
        tmp_df['shazam'] = shazam
        tmp_df['samm_raw_diff'] = raw_diff(samm, theta)
        tmp_df['shazam_raw_diff'] = raw_diff(shazam, theta)
        true_categories = pd.cut(theta.ravel(), np.arange(THETA_MIN, THETA_MAX + .1*STEP, STEP))
        tmp_df['sim_method'] = sim_method
        tmp_df['true theta size'] = true_categories
        tmp_df['seed'] = seed

        all_df = pd.concat((all_df, tmp_df))

for idx, sim_method in enumerate(SIM_METHODS):
    sub_df = all_df[all_df['sim_method'] == sim_method]
    _plot_single_effect_size_overall(sub_df, true_categories=true_categories, fname='_output/box_%s.pdf' % sim_method)
    _plot_scatter(all_df, fname='_output/scatter_%s.pdf' % sim_method)
