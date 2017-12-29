import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')

AXIS_INCREMENT = 5

def plot_martingale_residuals(sampler_results, fname, trim_proportion=None):
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
    xval = np.array(range(seq_len) * n_obs)
    flat_residuals = residuals.flatten()
    if trim_proportion is not None:
        column_mask = np.tile(np.sum(np.isnan(residuals), axis=0) / float(n_obs) > trim_proportion, n_obs)
        flat_residuals[column_mask] = np.nan
    ax = sns.regplot(xval, flat_residuals, dropna=True, scatter_kws={'alpha': .3})
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

# TODO: other plots
#def trace_plot():
#def hedgehog_plot():
#def comparison_hedgehog_plot():

