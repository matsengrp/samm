import numpy as np
import itertools
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')

from common import is_re_match, compute_known_hot_and_cold, HOT_COLD_SPOT_REGS

AXIS_INCREMENT = 5
THETA_MIN = -5
THETA_MAX = 5

# Colors used in theta plots; RGB values plus an alpha value (courtesy Kleinstein group)
GRAY = (153. / 255, 153. / 255, 153. / 255, 1.)
GREEN = (77. / 255, 175. / 255, 74. / 255, 1.)
RED = (239. / 255, 26. / 255, 28. / 255, 1.)
BLUE = (9. / 255, 77. / 255, 133. / 255, 1.)

# additional colors
YELLOW = (255. / 255, 215. / 255, 0. / 255, 1.)
PURPLE = (128. / 255, 0. / 255, 255. / 255, 1.)
YELLOW_ORANGE = (252. / 255, 209. / 255, 22. / 255, 1.)

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

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

def plot_martingale_residuals_on_axis(residuals_list, ax, metadata=None, trim_proportion=None, plot_average=False, align=False, gap_dict=None, recenter=False, center_col='j_gene_start', region_bounds=[], region_labels=[], ylabel='residual', alpha=1., pointsize=5, linesize=1, fontsize=8):
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
    @param region_labels: list of labels showing what region is what
    @param ylabel: ylabel of plot
    @param alpha: alpha parameter for points
    @param pointsize: size of scatterplot points
    @param linesize: thickness of lowess line
    @param fontsize: size of font for axes

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
        line_kws={'color': 'black', 'lw': linesize},
        fit_reg=False,
        ax=ax
    )
    ax.axhline(y=0, color='black', linestyle='--', linewidth=linesize)
    ax.set_xlabel('nucleotide position', fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    increment = int(AXIS_INCREMENT * round(float(seq_len)/(10*AXIS_INCREMENT)))
    ax.set_xticks(np.arange(np.min(xval), np.max(xval), increment))
    ax.set_xticklabels(np.arange(np.min(xval), np.max(xval), increment), fontsize=fontsize/2)

    if recenter:
        ax.axvline(x=0, color='black', linestyle='--', linewidth=linesize)
    top_ticks = []
    top_labels = []
    if region_bounds:
        for idx, (bound1, bound2) in enumerate(pairwise(region_bounds)):
            top_ticks.append(.5 * (bound1 + bound2))
            if not (idx % 2):
                top_labels.append(region_labels[idx])
                continue
            top_labels.append(region_labels[idx])
            ax.axvspan(bound1, bound2, ymin=.01, ymax=.99, alpha=0.5, color='gray')

    ax.legend()

    ax2 = ax.twiny()
    ax2.set_xticks(top_ticks)
    ax2.set_xticklabels(top_labels)
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')
    ax2.set_xlim(ax.get_xlim())

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

def plot_linear_top_n(mutability_info, ax, n, title, ylabel):
    """
    Plot theta values linearly---plot the top n positive and top n negative values

    @param mutability_info: list of dictionaries that is the output of get_mutability_info
    @param ax: axis on which to plot current residuals; useful for plt.subplots calls
    @param n: number of highest and lowest theta values to plot---if the number of nonzeros is greater than 2*n then plot all variables
    @param n: number of highest and lowest theta values to plot---if the number of nonzeros is greater than 2*n then plot all variables
    @param title: title of plot
    @param ylabel: ylabel of plot

    @return: None, just plot
    """
    # first sort by mutability and only take top (and bottom) n of them
    num_nonzero = len(mutability_info)
    sorted_mutability_info = [info for info in sorted(mutability_info, key=lambda k: k['mutability'][1], reverse=True)]
    if n < num_nonzero / 2:
        mutability_info = sorted_mutability_info[:n] + sorted_mutability_info[-n:]
        title += '\n (top {} values)'.format(len(mutability_info)/2)
    else:
        mutability_info = sorted_mutability_info
        # these are all the theta values
        title += '\n (num nonzeros={})'.format(len(mutability_info))

    # Add dummy values to vectors
    if n < num_nonzero / 2:
        mutability_info.insert(
            n,
            {
                'mutability': [0., 0., 0.],
                'feature_type': None,
                'label': r'$\vdots$',
                'color': None,
                'mut_pos': None,
                'motif': None,
            }
        )

    # Color features; default color is light gray
    colors = np.array([GRAY] * len(mutability_info))
    y = []
    ylo = []
    yhi = []
    feat_labels = []
    for idx, info in enumerate(mutability_info):
        # Currently hard-coded to assume fwr/cdr with yellow-orange as the default for any position feature
        if info['feature_type'] == 'position':
            if 'fwr' in info['label']:
                colors[idx,:] = PURPLE
            else:
                colors[idx,:] = YELLOW_ORANGE
        if info['feature_type'] == 'motif':
            motif = info['motif']
            motif_len = len(info['motif'])
            mut_pos = info['mut_pos']
            if mut_pos not in range(motif_len):
                colors[idx,:] = YELLOW
            else:
                known_hot_cold = compute_known_hot_and_cold(HOT_COLD_SPOT_REGS, motif_len, mut_pos)
                str_name = ''
                for spot_name, spot_regex in known_hot_cold:
                    if is_re_match(spot_regex, motif):
                        if 'hot' in spot_name and motif[mut_pos] == 'a' or motif[mut_pos] == 't':
                            colors[idx,:] = GREEN
                        elif 'hot' in spot_name:
                            colors[idx,:] = RED
                        elif 'cold' in spot_name:
                            colors[idx,:] = BLUE
                        break
        # change alpha value of features that have CIs overlapping with zero
        if not (info['mutability'][2] < 0 or info['mutability'][0] > 0):
            colors[idx,3] = .2
        y.append(info['mutability'][1])
        ylo.append(info['mutability'][1]-info['mutability'][0])
        yhi.append(info['mutability'][2]-info['mutability'][1])
        feat_labels.append(info['label'])

    nobs = len(y)
    x = np.array(range(nobs))
    errors = np.array([ylo, yhi])
    errors = np.reshape(errors, (2, len(ylo)))

    ax.scatter(x, y, s=15, c=colors)
    ax.errorbar(x, y, yerr=errors, xerr=None, ls='none', elinewidth=1.7, c=colors)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(range(nobs))

    # change font sizes based on how many labels we have
    if nobs < 60:
        fontsize = 10
    elif nobs < 100:
        fontsize = 8
    else:
        fontsize = 6

    ax.set_xticklabels(feat_labels, rotation='vertical', fontsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=24)
    for xtick, color in zip(ax.get_xticklabels(), colors):
        xtick.set_color(color)

def get_mutability_info(method_res):
    """
    Compute list of dicts for input into plot_linear_top_n

    @param method_res: a MethodResults() object from a samm fit; must have refit data

    @return list of dictionaries for input into plot_linear_top_n
    """
    assert(method_res.has_refit_data)

    feat_gen = method_res.refit_feature_generator
    theta = method_res.refit_theta

    output_info = []
    for i, info in enumerate(feat_gen.feature_info_list):
        out_dict = {}
        if method_res.has_conf_ints:
            out_dict['mutability'] = [method_res.conf_ints[i, 0], theta[i, 0], method_res.conf_ints[i, 2]]
        else:
            out_dict['mutability'] = [theta[i, 0], theta[i, 0], theta[i, 0]]

        # currently a bit of a hack so already-fit models can be plotted
        # basically if the second element of feature info is a list it's a position feature
        if isinstance(info[1], list):
            # position feature
            out_dict['feature_type'] = 'position'
            out_dict['motif'] = None
            out_dict['mut_pos'] = None
            out_dict['label'] = r"%s" % info[0]
        else:
            # motif feature
            motif, pos = info
            out_dict['feature_type'] = 'motif'
            out_dict['motif'] = motif
            out_dict['mut_pos'] = -pos
            motif, pos = info
            mlen = len(motif)
            if pos <= 0 and -pos <= mlen - 1:
                # kmer
                out_dict['label'] = r'\texttt{%s\underline{%s}%s}' % (motif[:-pos], motif[-pos], motif[-pos+1:])
            else:
                # adjacent motif
                if pos < 0:
                    out_dict['label'] = r"$\texttt{%s}$-%d" % (motif, -pos)
                else:
                    out_dict['label'] = r"$\texttt{%s}$+%d" % (motif, pos)
        output_info.append(out_dict)
    return output_info
