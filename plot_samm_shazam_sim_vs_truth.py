import pandas as pd
import matplotlib
matplotlib.use('pdf')
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import numpy as np
import scipy.stats
import matplotlib.lines as mlines

sns.set(style="white")

from common import *
from read_data import *
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from fit_logistic_model import LogisticModel

# some constants---pass these through as variables?
NSEEDS = range(50)
MOTIF_LEN = 5
MUT_POS = 2
SIM_METHODS = ['final_revisions_shmulate_m3-5_s2000']
#SIM_METHODS = ['final_revisions_survival_m3-5_s2000']
TRUE_MODEL_STR = "simulated_shazam_vs_samm/_output/%s/%02d/True/true_model.pkl"
SAMM_MODEL_STR = "simulated_shazam_vs_samm/_output/%s/%02d/True/fitted.pkl"
LOGISTIC_MODEL_STR = "simulated_shazam_vs_samm/_output/%s/%02d/True/logistic_model.pkl"
SHAZAM_MUT_STR = "simulated_shazam_vs_samm/_output/%s/%02d/True/fitted_shazam_mut.csv"
SHAZAM_SUB_STR = "simulated_shazam_vs_samm/_output/%s/%02d/True/fitted_shazam_sub.csv"
THETA_MIN = -5
THETA_MAX = 5
# what step size to use to divide true theta by effect size
STEP = 2

def raw_diff(agg_fit_theta, agg_true_theta):
    possible_agg_true_theta = agg_true_theta - np.median(agg_true_theta)
    possible_agg_refit_theta = agg_fit_theta - np.median(agg_fit_theta)

    return (possible_agg_refit_theta - possible_agg_true_theta)

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
    all_df = all_df[['samm_raw_diff', 'shazam_raw_diff', 'logistic_raw_diff', 'true theta size']]
    all_df.rename(index=str,
            columns={
                'samm_raw_diff': 'samm',
                'shazam_raw_diff': 'SHazaM',
                'logistic_raw_diff': 'Logistic',
                'true theta size': 'true theta size'},
            inplace=True)
    melt_df = pd.melt(all_df, id_vars=['true theta size'])

    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14) 

    sns_plot = sns.factorplot(
        x='true theta size',
        y='value',
        hue='variable',
        data=melt_df,
        kind="box",
        palette="Set2",
        order=true_categories.categories,
        legend=False,
        size=7
    )
    hatches = ['//', '+', 'x', '\\', '*', 'o', 'O', '.']
    for i, bar in enumerate(sns_plot.axes[0,0].artists):
        hatch = hatches[i % 3]
        bar.set_hatch(hatch)
    for i, bar in enumerate(sns_plot.axes[0,0].patches):
        hatch = hatches[i % 3]
        bar.set_hatch(hatch)
    plt.ylabel('Diff. from true theta', fontsize=17)
    plt.xlabel("True theta size", fontsize=17)
    sns_plot.set(ylim=(-5, 5))
    x = sns_plot.axes[0,0].get_xlim()
    sns_plot.axes[0,0].plot(x, len(x) * [0], 'k--', alpha=.4)
    plt.legend(loc='upper right', fontsize='large')

    sns_plot.savefig(fname)

dense_agg_feat_gen = HierarchicalMotifFeatureGenerator(
    motif_lens=[MOTIF_LEN],
    left_motif_flank_len_list=[[MUT_POS]],
)

all_df = pd.DataFrame(columns=['theta', 'samm_fit', 'shazam_fit', 'sim_method', 'seed'])
for sim_method in SIM_METHODS:
    for seed in NSEEDS:
        with open(TRUE_MODEL_STR % (sim_method, seed), 'r') as f:
            theta, _ = pickle.load(f)
        model_shape = theta.shape
        possible_agg_mask = dense_agg_feat_gen.get_possible_motifs_to_targets(
            model_shape,
        )
        theta = theta[possible_agg_mask] - np.median(theta[possible_agg_mask])
        tmp_df = pd.DataFrame()
        tmp_df['theta'] = theta

        # Load samm
        print SAMM_MODEL_STR % (sim_method, seed)
        samm = load_fitted_model(
            SAMM_MODEL_STR % (sim_method, seed),
            add_targets=True
        ).agg_refit_theta
        samm = samm[possible_agg_mask] - np.median(samm[possible_agg_mask])
        tmp_df['samm'] = samm

        # Load logistic
        logistic_model = load_logistic_model(
            LOGISTIC_MODEL_STR % (sim_method, seed)
        ).agg_refit_theta
        logistic_model = logistic_model[possible_agg_mask] - np.median(logistic_model[possible_agg_mask])
        tmp_df['logistic'] = logistic_model

        # Load shazam
        shazam_raw = get_shazam_theta(
            SHAZAM_MUT_STR % (sim_method, seed),
            SHAZAM_SUB_STR % (sim_method, seed),
            wide_format=True
        )
        shazam = shazam_raw[:,0:1] + shazam_raw[:,1:]
        shazam = shazam[possible_agg_mask] - np.median(shazam[possible_agg_mask])
        tmp_df['shazam'] = shazam

        # Final processing
        tmp_df['samm_raw_diff'] = raw_diff(samm, theta)
        tmp_df['logistic_raw_diff'] = raw_diff(logistic_model, theta)
        tmp_df['shazam_raw_diff'] = raw_diff(shazam, theta)
        true_categories = pd.cut(theta.ravel(), np.arange(THETA_MIN, THETA_MAX + .1*STEP, STEP))
        tmp_df['sim_method'] = sim_method
        tmp_df['true theta size'] = true_categories
        tmp_df['seed'] = seed

        all_df = pd.concat((all_df, tmp_df))

for idx, sim_method in enumerate(SIM_METHODS):
    sub_df = all_df[all_df['sim_method'] == sim_method]
    print np.max(sub_df['samm']), np.min(sub_df['samm'])
    print np.max(sub_df['shazam']), np.min(sub_df['shazam'])
    print np.max(sub_df['theta']), np.min(sub_df['theta'])
    _plot_single_effect_size_overall(
            sub_df,
            true_categories=true_categories,
            fname='_output/revision_box_%s.pdf' % sim_method)
