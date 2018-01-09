import numpy as np
import itertools
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')

AXIS_INCREMENT = 5
THETA_MIN = -5
THETA_MAX = 5

def plot_martingale_residuals(sampler_results, fname, trim_proportion=None, plot_average=False):
    """
    Plot martingale residuals per nucleotide position

    @param sampler_results: output from running get_samples from a SamplerCollection object
    @param fname: file name for residuals plot
    @param trim_proportion: don't plot positions with trim_proportion or more missing values;
        if None then print all data points

    @return: numpy array of n_obs x seq_len martingale residuals
    """
    residual_list = [res.residuals for res in sampler_results]
    residuals = np.array(list(itertools.izip_longest(*residual_list, fillvalue=np.nan))).T
    n_obs, seq_len = residuals.shape
    if trim_proportion is not None:
        column_mask = np.tile(np.sum(np.isnan(residuals), axis=0) / float(n_obs) > trim_proportion, n_obs)
    else:
        column_mask = np.tile([False] * seq_len, n_obs)

    if plot_average:
        flat_residuals = np.nanmean(residuals, axis=0)
        flat_residuals[column_mask[:seq_len]] = np.nan
        xval = np.array(range(seq_len))
        alpha = 1.
    else:
        xval = np.array(range(seq_len) * n_obs)
        flat_residuals = residuals.flatten()
        flat_residuals[column_mask] = np.nan
        alpha = .3

    ax = sns.regplot(xval, flat_residuals, dropna=True, scatter_kws={'alpha': alpha}, lowess=True)
    ax.set(xlabel='nucleotide position', ylabel='residual')
    if seq_len > 100:
        increment = AXIS_INCREMENT * 2
    else:
        increment = AXIS_INCREMENT
    ax.set_xticks(np.arange(0, seq_len, increment))
    ax.set_xticklabels(np.arange(0, seq_len, increment))
    plt.savefig(fname)
    plt.clf()
    return residuals

def plot_model_scatter(all_df, fname='', hue_var=None, df_labels=['1', '2']):

    sns_plot = sns.lmplot(
        x="theta{}".format(df_labels[0]),
        y="theta{}".format(df_labels[1]),
        hue=hue_var,
        lowess=True,
        scatter=True,
        data=all_df,
        line_kws={'lw':3},
        scatter_kws={'alpha':.5},
        legend=False,
        palette="Set2",
    )
    sns_plot.axes[0][0].plot([THETA_MIN, THETA_MAX],[THETA_MIN,THETA_MAX], color="black", ls="--", label="y=x")
    model_legend = plt.legend(loc='lower right', labels=['other', 'nonoverlap nonzero'])

    sns_plot.set(xlabel="{} theta".format(df_labels[0]))
    sns_plot.set(ylabel="{} theta".format(df_labels[1]))
    sns_plot.set(ylim=(THETA_MIN, THETA_MAX))
    sns_plot.set(xlim=(THETA_MIN, THETA_MAX))

    plt.savefig(fname)
    plt.clf()

# TODO: other plots
#def trace_plot():
#def hedgehog_plot():
#def comparison_hedgehog_plot():

