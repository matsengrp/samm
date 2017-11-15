import pickle
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

RATIO_FILES = [
    '_output/shazam_vs_samm_intermediate/_output/cui/sample/igk/igk-motif-5-flank-2-False-loglik.pkl',
    '_output/shazam_vs_samm_intermediate/_output/cui/sample/igk/igk-motif-3-5-flank-1--2-False-loglik.pkl',
    '_output/shazam_vs_samm_intermediate/_output/cui/sample/igk/igk-motif-3-5-flank-0-1-2--2-False-loglik.pkl',
    '_output/shazam_vs_samm_intermediate/_output/cui/sample/igk/igk-motif-5-flank-2-True-pt-loglik.pkl',
    '_output/shazam_vs_samm_intermediate/_output/cui/sample/igk/igk-motif-3-5-flank-1--2-True-pt-loglik.pkl',
    '_output/shazam_vs_samm_intermediate/_output/cui/sample/igk/igk-motif-3-5-flank-0-1-2--2-True-pt-loglik.pkl',
]

def grouped_errorbar(df, fname):

    plt.clf()
    fig, axes = plt.subplots(2, sharex=True)
    for i, (per_target, ax) in enumerate(zip([False, True], axes)):
        plot_df = df[df['per_target'] == per_target]
        for idx, mtype in enumerate(set(df['motif_len'])):
            plot_df = df[((df['per_target'] == per_target) & (df['motif_len'] == mtype))]
            ax.errorbar(plot_df['validation mouse'] + .07 * idx,
                    plot_df['EM surrogate log ratio'],
                    yerr=[
                        plot_df['EM surrogate log ratio'] - plot_df['lower 95'],
                        -plot_df['EM surrogate log ratio'] + plot_df['upper 95']
                    ], fmt='.', marker=".", label=mtype)
        ax.set_xticks(plot_df['validation mouse'] + .07)
        ax.set_xticklabels(('1', '2', '3', '4'))
        if per_target:
            ax.set_ylabel('per-target')
        else:
            ax.set_ylabel('same target')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.73, box.height])
        x = ax.get_xlim()
        ax.plot(x, len(x) * [0], 'k--', alpha=.4)
        if i == 1:
            lgd = ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    
    fig.text(0.005, 0.5, 'EM surrogate difference', va='center', rotation='vertical')
    plt.xlabel('Validation mouse')
    plt.savefig(fname)

df = pd.DataFrame(columns=['validation mouse', 'EM surrogate log ratio'])
for ratio_file in RATIO_FILES:
    with open(ratio_file, 'r') as f:
        res = pickle.load(f)
        ratios = [output['shazam_ref'] for output in res]
        ratios = np.array(ratios).T
    
    tmp_df = pd.DataFrame(columns=['validation mouse', 'EM surrogate log ratio'])
    tmp_df['validation mouse'] = range(1, 5)
    
    tmp_df['EM surrogate log ratio'] = ratios[0]
    tmp_df['lower 95'] = ratios[1]
    tmp_df['upper 95'] = ratios[2]
    tmp_df['per_target'] = 'True' in ratio_file.split('-')
    if 'motif-5-' in ratio_file:
        tmp_df['motif_len'] = '5mer'
    elif 'motif-3-5-' in ratio_file:
        if 'flank-2-' in ratio_file:
            tmp_df['motif_len'] = '3,5mer'
        elif 'flank-0-1-2--2' in ratio_file:
            tmp_df['motif_len'] = '3,5mer offset'

    df = pd.concat((df, tmp_df))

grouped_errorbar(df, '_output/real_data_summary.svg')

