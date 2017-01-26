import csv
import numpy as np
import pandas as pd

from models import ObservedSequenceMutations

NUCLEOTIDES = "atcg"
GERMLINE_PARAM_FILE = '/home/matsengrp/working/matsen/SRR1383326-annotations-imgt-v01.h5'
ZSCORE = 1.65
ZERO_THRES = 1e-6

def mutate_string(begin_str, mutate_pos, mutate_value):
    """
    Mutate a string
    """
    return "%s%s%s" % (begin_str[:mutate_pos], mutate_value, begin_str[mutate_pos + 1:])

def sample_multinomial(pvals):
    """
    Sample 1 item from multinomial and get the index of this sample
    will renormalize pvals if needed
    """
    norm_pvals = np.array(pvals)/np.sum(pvals)
    assert(np.sum(norm_pvals) > 1 - 1e-10)
    sample = np.random.multinomial(1, norm_pvals)
    return np.where(sample == 1)[0][0]

def get_random_dna_seq(seq_length, nucleotide_probs=[0.25, 0.25, 0.25, 0.25]):
    """
    Generate a random dna sequence
    """
    random_nucleotides = [
        NUCLEOTIDES[sample_multinomial(nucleotide_probs)] for i in range(seq_length)
    ]
    return "".join(random_nucleotides)

def check_unordered_equal(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)

def get_standard_error_ci_corrected(values, zscore):
    """
    @returns
        the standard error of the values correcting for auto-correlation between the values
        the lower bound of the mean of the values using the standard error and the given zscore
        the upper bound of the mean of the values using the standard error and the given zscore
    Calculate the autocorrelation, then the effective sample size, scale the standard error appropriately the effective sample size
    """
    mean = np.mean(values)
    var = np.var(values)

    # If the values are essentially constant, then the autocorrelation is zero.
    # (There are numerical stability issues if we go thru the usual calculations)
    if var < 1e-10:
        return 0, mean, mean

    # Calculate auto-correlation
    # Definition from p. 151 of Carlin/Louis:
    # \kappa = 1 + 2\sum_{k=1}^\infty \rho_k
    # So we don't take the self-correlation
    # TODO: do we worry about cutting off small values?
    # Glynn/Whitt say we could use batch estimation with batch sizes going to
    # infinity. Is this a viable option?
    result = np.correlate(values - mean, values - mean, mode='full')
    result = result[result.size/2:]
    result /= (var * np.arange(values.size, 0, -1))

    # truncate sum once the autocorrelation is negative
    neg_indices = np.where(result < 0)
    neg_idx = result.size
    if len(neg_indices) > 0 and neg_indices[0].size > 1:
        neg_idx = np.where(result < 0)[0][0]

    autocorr = 1 + 2*np.sum(result[1:neg_idx])

    # Effective sample size calculation
    ess = values.size/autocorr

    # Corrected standard error
    ase = np.sqrt(var/ess)

    return ase, mean - zscore * ase, mean + zscore * ase

def soft_threshold(theta, thres):
    """
    The soft thresholding function S is zero in the range [-thresh, thresh],
    theta+thresh when theta < -thresh and theta-thresh when theta > thresh.

    @param theta: a numpy vector
    @param thres: the amount to threshold theta by
    @return theta that is soft-thresholded with constant thres
    """
    return np.maximum(theta - thres, 0) + np.minimum(theta + thres, 0)

def read_bcr_hd5(path, remove_gap=True):
    """
    read hdf5 parameter file and process
    """

    sites = pd.read_hdf(path, 'sites')

    if remove_gap:
        return sites.query('base != "-"')
    else:
        return sites

def read_gene_seq_csv_data(gene_file_name, seq_file_name):
    gene_dict = {}
    with open(gene_file_name, "r") as gene_csv:
        gene_reader = csv.reader(gene_csv, delimiter=',')
        gene_reader.next()
        for row in gene_reader:
            gene_dict[row[0]] = row[1]

    obs_data = []
    with open(seq_file_name, "r") as seq_csv:
        seq_reader = csv.reader(seq_csv, delimiter=",")
        seq_reader.next()
        for row in seq_reader:
            start_seq = gene_dict[row[0]].lower()
            end_seq = row[2]
            obs_data.append(
                ObservedSequenceMutations(
                    start_seq=start_seq[:len(end_seq)],
                    end_seq=end_seq,
                )
            )
    return gene_dict, obs_data
