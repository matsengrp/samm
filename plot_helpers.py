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

def _get_flat_residuals(sampler_results, trim_proportion=None, plot_average=False):
    """
    Get flattened residuals from Gibbs sampler results

    @param sampler_results: list of GibbsSamplerResults
    @param trim_proportion: don't plot positions with trim_proportion or more missing values;
        if None then print all data points
    @param plot_average: plot the average of the residuals per position instead of all residuals

    @return: x-values, i.e., nucleotide positions
    @return: numpy array of n_obs x seq_len martingale residuals
    @return: numpy array of masked, possibly averaged residuals
    """

    residual_list = [res.residuals for res in sampler_results]
    residuals = np.array(list(itertools.izip_longest(*residual_list, fillvalue=np.nan))).T
    n_obs, seq_len = residuals.shape
    # Construct mask of positions with too many NaNs
    if trim_proportion is not None:
        column_mask = np.tile(np.sum(np.isnan(residuals), axis=0) / float(n_obs) > trim_proportion, n_obs)
    else:
        column_mask = np.tile([False] * seq_len, n_obs)

    # Take column-wise average if we ask for it
    if plot_average:
        flat_residuals = np.nanmean(residuals, axis=0)
        flat_residuals[column_mask[:seq_len]] = np.nan
        xval = np.array(range(seq_len))
    else:
        xval = np.array(range(seq_len) * n_obs)
        flat_residuals = residuals.flatten()
        flat_residuals[column_mask] = np.nan

    return xval, residuals, flat_residuals

def plot_martingale_residuals_on_axis(sampler_results, ax, trim_proportion=None, plot_average=False, title='Residuals vs. Position', alpha=1., pointsize=5, linesize=1, fontsize=8):
    """
    Plot martingale residuals per nucleotide position

    @param sampler_results: list of GibbsSamplerResults
    @param ax: axis on which to plot current residuals; useful for plt.subplots calls
    @param trim_proportion: don't plot positions with trim_proportion or more missing values;
        if None then print all data points
    @param plot_average: plot the average of the residuals per position instead of all residuals
    @param title: title of plot
    @param alpha: alpha parameter for points
    @param pointsize: size of scatterplot points
    @param linesize: thickness of lowess line
    @param fontsize: size of font for axes/title

    @return: numpy array of n_obs x seq_len martingale residuals
    """

    xval, residuals, flat_residuals = _get_flat_residuals(sampler_results, trim_proportion, plot_average)
    _, seq_len = residuals.shape

    sns.regplot(
        xval,
        flat_residuals,
        dropna=True,
        scatter_kws={'alpha': alpha, 's': pointsize, 'color': 'black'},
        line_kws={'color': 'black', 'lw': linesize},
        lowess=True,
        ax=ax
    )
    ax.set_xlabel('nucleotide position', fontsize=fontsize)
    ax.set_ylabel('residual', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    increment = int(AXIS_INCREMENT * round(float(seq_len)/(10*AXIS_INCREMENT)))
    ax.set_xticks(np.arange(0, seq_len, increment))
    ax.set_xticklabels(np.arange(0, seq_len, increment), fontsize=fontsize/2)
    return residuals

def plot_martingale_residuals_to_file(sampler_results, fname, trim_proportion=None, plot_average=False, title='Average residuals vs. position', alpha=1.):
    """
    Plot martingale residuals per nucleotide position

    @param sampler_results: list of GibbsSamplerResults
    @param fname: file name for residuals plot
    @param trim_proportion: don't plot positions with trim_proportion or more missing values;
        if None then print all data points
    @param plot_average: plot the average of the residuals per position instead of all residuals
    @param title: title of plot

    @return: numpy array of n_obs x seq_len martingale residuals
    """

    xval, residuals, flat_residuals = _get_flat_residuals(sampler_results, trim_proportion, plot_average)
    _, seq_len = residuals.shape

    ax = sns.regplot(xval, flat_residuals, dropna=True, scatter_kws={'alpha': alpha}, lowess=True)
    ax = sns.regplot(
        xval,
        flat_residuals,
        dropna=True,
        scatter_kws={'alpha': alpha},
        lowess=True
    )
    ax.set(xlabel='nucleotide position', ylabel='residual', title=title)
    increment = int(AXIS_INCREMENT * round(float(seq_len)/(10*AXIS_INCREMENT)))
    ax.set_xticks(np.arange(0, seq_len, increment))
    ax.set_xticklabels(np.arange(0, seq_len, increment))
    plt.savefig(fname)
    plt.clf()
    return residuals

def plot_model_scatter(all_df, fname, hue_var=None, df_labels=['1', '2'], alpha=.5, linesize=3, legend_labels=None):
    """
    Scatterplot of two theta fits

    @param all_df: pandas dataframe containing information on theta fits
    @param fname: file name for residuals plot
    @param hue_var: variable on which to color points
    @param df_labels: labels of the different theta fits
    @param alpha: alpha parameter for points
    @param linesize: thickness of lowess line
    @param legend_labels: labels for hue_var; if None then use unique values of hue_var

    @return: None, just save and clear the plot
    """

    sns_plot = sns.lmplot(
        x="theta{}".format(df_labels[0]),
        y="theta{}".format(df_labels[1]),
        hue=hue_var,
        lowess=True,
        scatter=True,
        data=all_df,
        line_kws={'lw': linesize},
        scatter_kws={'alpha': alpha},
        legend=False,
        palette="Set2",
    )
    sns_plot.axes[0][0].plot([THETA_MIN, THETA_MAX],[THETA_MIN,THETA_MAX], color="black", ls="--", label="y=x")
    if legend_labels is None:
        legend_labels = all_df[hue_var].unique()
    model_legend = plt.legend(loc='lower right', labels=legend_labels)

    sns_plot.set(xlabel="{} theta".format(df_labels[0]))
    sns_plot.set(ylabel="{} theta".format(df_labels[1]))
    sns_plot.set(ylim=(THETA_MIN, THETA_MAX))
    sns_plot.set(xlim=(THETA_MIN, THETA_MAX))

    plt.savefig(fname)
    plt.clf()

