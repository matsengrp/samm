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

def _align(metadata, residual_list, gap_dict):
    """
    Align to IMGT gaps

    @param metadata: contains information on aligning and recentering
    @param residual_list: residuals obtained from sampler results
    @param gap_dict: dictionary of {'gene': list-of-IMGT-gaps} for aligning

    @return aligned residuals
    """
    aligned_residuals = []
    for meta, residual in zip(metadata, residual_list):
        gene = meta['v_gene'].split('+')[0]
        if gene not in gap_dict.keys():
            print "%s not found, skipping" % gene
            continue
        # if gap dict has an "n" then insert NaN and continue
        for idx in gap_dict[gene]:
            if idx <= len(residual):
                # if it's greater then we'll pad the end regardless
                residual = np.insert(residual, idx, np.nan)
        aligned_residuals.append(residual)
    return aligned_residuals

def _recenter(metadata, residual_list, center_col):
    """
    Recenter, e.g., DJ genes around the start of the J gene

    @param metadata: contains information on aligning and recentering
    @param residual_list: residuals obtained from sampler results
    @param center_col: column in metadata that has the centering position

    @return recentered residuals, offset value (the largest value of position start of J gene)
    """

    recenter_residuals = []

    # calculate max start position so we can align x values at the end
    offset = max([meta[center_col] for meta in metadata])

    for meta, residual in zip(metadata, residual_list):
        pad_begin = [np.nan] * (offset - meta[center_col])
        recenter_residuals.append(np.concatenate((pad_begin, residual)))
    return recenter_residuals, offset

def _get_flat_processed_residuals(residual_list, metadata=None, trim_proportion=None, plot_average=False, align=False, gap_dict=None, recenter=False, center_col='j_gene_start'):
    """
    Get flattened residuals from Gibbs sampler results

    @param sampler_results: list of GibbsSamplerResults
    @param metadata: contains information on aligning and recentering
    @param trim_proportion: don't plot positions with trim_proportion or more missing values;
        if None then print all data points
    @param plot_average: plot the average of the residuals per position instead of all residuals
    @param align: align to IMGT positions?
    @param gap_dict: dictionary of {'gene': list-of-IMGT-gaps} for aligning
    @param recenter: recenter plot around value at center_col
    @param center_col: column in metadata that has the centering position

    @return: x-values, i.e., nucleotide positions
    @return: numpy array of n_obs x seq_len martingale residuals
    @return: numpy array of masked, possibly averaged residuals
    """

    if (align or recenter) and metadata is None:
        raise ValueError()

    offset = 0

    if align:
        residual_list = _align(metadata, residual_list, gap_dict)

    if recenter:
        residual_list, offset = _recenter(metadata, residual_list, center_col)

    # pad ends of shorter residuals with NaNs
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
        xval = np.array(range(seq_len)) - offset
    else:
        xval = np.array(range(seq_len) * n_obs) - offset
        flat_residuals = residuals.flatten()
        flat_residuals[column_mask] = np.nan

    return xval, residuals, flat_residuals, offset

def plot_martingale_residuals_on_axis(residuals_list, ax, metadata=None, trim_proportion=None, plot_average=False, align=False, gap_dict=None, recenter=False, center_col='j_gene_start', region_bounds=[], title='Residuals vs. Position', xlabel='residual', alpha=1., pointsize=5, linesize=1, fontsize=8):
    """
    Plot martingale residuals per nucleotide position

    @param residuals_list: list of residuals; can also be any list of position-wise statistics
    @param ax: axis on which to plot current residuals; useful for plt.subplots calls
    @param metadata: contains information on aligning and recentering
    @param trim_proportion: don't plot positions with trim_proportion or more missing values;
        if None then print all data points
    @param plot_average: plot the average of the residuals per position instead of all residuals
    @param align: align to IMGT positions?
    @param gap_dict: dictionary of {'gene': list-of-IMGT-gaps} for aligning
    @param recenter: recenter plot around value at center_col
    @param center_col: column in metadata that has the centering position
    @param region_bounds: list of positions showing where to plot FW/CDR region boundaries
    @param title: title of plot
    @param xlabel: xlabel of plot
    @param alpha: alpha parameter for points
    @param pointsize: size of scatterplot points
    @param linesize: thickness of lowess line
    @param fontsize: size of font for axes/title

    @return: numpy array of n_obs x seq_len martingale residuals and plot
    """

    if align and gap_dict is None:
        raise ValueError()

    xval, residuals, flat_residuals, offset = _get_flat_processed_residuals(
        residuals_list,
        metadata,
        trim_proportion,
        plot_average,
        align,
        gap_dict,
        recenter,
        center_col,
    )

    _, seq_len = residuals.shape

    sns.regplot(
        xval,
        flat_residuals,
        dropna=True,
        scatter_kws={'alpha': alpha, 's': pointsize, 'color': 'black'},
        line_kws={'color': 'black', 'lw': linesize},
        #lowess=True,
        fit_reg=False,
        ax=ax
    )
    ax.axhline(y=0, color='black', linestyle='--', linewidth=linesize)
    ax.set_xlabel('nucleotide position', fontsize=fontsize)
    ax.set_ylabel(xlabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    increment = int(AXIS_INCREMENT * round(float(seq_len)/(10*AXIS_INCREMENT)))
    ax.set_xticks(np.arange(np.min(xval), np.max(xval), increment))
    ax.set_xticklabels(np.arange(np.min(xval), np.max(xval), increment), fontsize=fontsize/2)
    if recenter:
        ax.axvline(x=0, color='black', linestyle='--', linewidth=linesize)
    if align:
        for bound in region_bounds:
            if bound < seq_len:
                ax.axvline(x=bound-offset, color='black', linestyle=':', linewidth=1)
    return residuals

def plot_model_scatter(all_df, fname, hue_var=None, df_labels=['1', '2'], alpha=.5, linesize=3, legend_labels=None, title=''):
    """
    Scatterplot of two theta fits

    @param all_df: pandas dataframe containing information on theta fits
    @param fname: file name for residuals plot
    @param hue_var: variable on which to color points
    @param df_labels: labels of the different theta fits
    @param alpha: alpha parameter for points
    @param linesize: thickness of lowess line
    @param legend_labels: labels for hue_var; if None then use unique values of hue_var
    @param title: title of plot

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
    if legend_labels is None:
        legend_labels = all_df[hue_var].unique()
    model_legend = plt.legend(loc='lower right', labels=legend_labels)

    sns_plot.axes[0][0].plot([THETA_MIN, THETA_MAX],[THETA_MIN,THETA_MAX], color="black", ls="--", label="y=x")

    sns_plot.set(xlabel="{} theta".format(df_labels[0]))
    sns_plot.set(ylabel="{} theta".format(df_labels[1]))
    sns_plot.set(ylim=(THETA_MIN, THETA_MAX))
    sns_plot.set(xlim=(THETA_MIN, THETA_MAX))
    sns_plot.set(title=title)

    plt.savefig(fname)
    plt.clf()

def plot_model_scatter_on_axis(all_df, ax, df_labels=['1', '2'], alpha=.5, linesize=3, title=''):
    """
    Scatterplot of two theta fits

    @param all_df: pandas dataframe containing information on theta fits
    @param ax: file name for residuals plot
    @param hue_var: variable on which to color points
    @param df_labels: labels of the different theta fits
    @param alpha: alpha parameter for points
    @param linesize: thickness of lowess line
    @param title: title of plot

    @return: None, just plot
    """

    sns_plot = sns.regplot(
        x=all_df["theta{}".format(df_labels[0])],
        y=all_df["theta{}".format(df_labels[1])],
        lowess=True,
        line_kws={'lw': linesize},
        scatter_kws={'alpha': alpha},
        ax=ax,
    )

    sns_plot.axes.plot([THETA_MIN, THETA_MAX],[THETA_MIN,THETA_MAX], color="black", ls="--", label="y=x")

    sns_plot.set(xlabel="{} theta".format(df_labels[0]))
    sns_plot.set(ylabel="{} theta".format(df_labels[1]))
    sns_plot.set(ylim=(THETA_MIN, THETA_MAX))
    sns_plot.set(xlim=(THETA_MIN, THETA_MAX))
    sns_plot.set(title=title)

