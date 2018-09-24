import pandas as pd
import sys
import scipy.stats
import numpy as np
import argparse

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input-naive',
        type=str,
        help='Input CSV file with naive sequences')
    parser.add_argument('--input-mutated',
        type=str,
        help='Input CSV file with naive sequences')
    parser.add_argument('--true-csv',
        type=str,
        help='csv with true theta values')
    parser.add_argument('--fitted-csv',
        type=str,
        help='csv with results from imputed_ancestors_comparison')
    parser.add_argument('--outtable',
        type=str,
        default="_output/stats.txt",
        help='file to print output table')

    args = parser.parse_args()
    return args

def print_fcn(x):
    output = {}
    x['theta'] -= np.median(x['theta'])
    cols = [
        r"Relative $\boldsymbol\theta$ error",
        r"Kendall's tau",
    ]
    output[cols[0]] = np.linalg.norm(x['theta'] - x['theta_truth']) / np.linalg.norm(x['theta_truth'])
    output[cols[1]] = scipy.stats.kendalltau(
        x['theta'],
        x['theta_truth'],
    )[0]
    return pd.Series(output, index=cols).round(3)

def main(args=sys.argv[1:]):
    args = parse_args()

    ### First get cluster size statistics
    out_str = ''
    genes = pd.read_csv(args.input_naive)
    seqs = pd.read_csv(args.input_mutated)

    full_data = pd.merge(genes, seqs, on='germline_name')

    cluster_sizes = []
    n_singletons = 0
    for _, cluster in full_data.groupby(['germline_name']):
        cluster_sizes.append(len(cluster))
        if len(cluster) == 1:
            n_singletons += 1
    out_str += "Cluster statistics"
    out_str += """
        Min cluster size: %d
        Max cluster size: %d
        Median cluster size: %d
        Pct singletons: %.0f\n
    """ % (min(cluster_sizes), max(cluster_sizes), np.median(cluster_sizes), 100 * np.mean([1. if csize == 1 else 0. for csize in cluster_sizes]))


    ### Now get Table 3 of the manuscript (error and correlation of each processing strategy)
    out_str += "Table 3\n"
    MODEL_LABEL = {
        "SHazaM": r"\texttt{SHazaM}",
        "samm": r"\texttt{samm}",
    }
    DATA_LABEL = {
        "all_data": r"All data",
        "imputed_ancestors": r"Imputation",
        "sample_random": r"Sampling",
    }
    true_df = pd.read_csv(args.true_csv)
    df = pd.read_csv(args.fitted_csv)
    all_df = pd.merge(df, true_df, on=['motif', 'col', 'rep'], suffixes=('', '_truth'))
    all_df['model'] = [MODEL_LABEL[val] for val in all_df['model']]
    all_df['data'] = [DATA_LABEL[val] for val in all_df['data']]
    all_df.rename(columns={'model': 'Model', 'data': 'Data processing'}, inplace=True)

    group_cols= [
        'Data processing',
        'Model',
    ]
    print_df = all_df.groupby(group_cols).apply(print_fcn)
    out_str += print_df.to_latex(escape=False)
    with open(args.outtable, 'w') as f:
        f.write(out_str)

if __name__ == "__main__":
    main(sys.argv[1:])
