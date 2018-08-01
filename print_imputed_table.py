import pandas as pd
import sys
import scipy.stats
import numpy as np
import argparse

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

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
    with open(args.outtable, 'w') as f:
        f.write(print_df.to_latex(escape=False))

if __name__ == "__main__":
    main(sys.argv[1:])
