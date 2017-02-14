import csv
import numpy as np
import pandas as pd
import sys

PARTIS_PATH = './partis'
sys.path.insert(1, PARTIS_PATH + '/python')
import utils
import glutils

# needed to read partis files
csv.field_size_limit(sys.maxsize)

from models import ObservedSequenceMutations
NUM_NUCLEOTIDES = 4
NUCLEOTIDES = "atcg"
NUCLEOTIDE_SET = set(["a", "t", "c", "g"])
NUCLEOTIDE_DICT = {
    "a": 0,
    "t": 1,
    "c": 2,
    "g": 3,
}
GERMLINE_PARAM_FILE = '/home/matsengrp/working/matsen/SRR1383326-annotations-imgt-v01.h5'
SAMPLE_PARTIS_ANNOTATIONS = PARTIS_PATH + '/test/reference-results/partition-new-simu-cluster-annotations.csv'
ZSCORE = 1.65
ZERO_THRES = 1e-6
MAX_TRIALS = 10

def contains_degenerate_base(seq_str):
    for nucleotide in seq_str:
        if nucleotide not in NUCLEOTIDE_SET:
            return True
    return False

def get_randint():
    """
    @return a random integer from a large range
    """
    return np.random.randint(low=0, high=2**32 - 1)

def get_nonzero_theta_print_lines(theta, motif_list):
    """
    @return a string that summarizes the theta vector/matrix
    """
    lines = []
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if np.isfinite(theta[i,j]) and np.abs(theta[i,j]) > ZERO_THRES:
                if i == theta.shape[0] - 1:
                    lines.append("%d: %s (EDGES)" % (i, theta[i, :]))
                else:
                    lines.append("%d: %s (%s)" % (i, theta[i, :], motif_list[i]))
                break
    return "\n".join(lines)

def get_possible_motifs_to_targets(motif_list, mask_shape):
    """
    @param motif_list: list of motifs - assumes that the first few theta rows correspond to these motifs
    @param mask_shape: shape of the theta matrix

    @return a boolean matrix with possible mutations as True, impossible mutations as False
    """
    theta_mask = np.ones(mask_shape, dtype=bool)
    if mask_shape[1] == 1:
        # Estimating a single theta vector - then we should estimate all theta values
        return theta_mask

    # Estimating a different theta vector for different target nucleotides
    # We cannot have a motif mutate to the same center nucleotide
    center_motif_idx = len(motif_list[0])/2
    for i in range(len(motif_list)):
        center_nucleotide_idx = NUCLEOTIDE_DICT[motif_list[i][center_motif_idx]]
        theta_mask[i, center_nucleotide_idx] = False
    return theta_mask

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

def get_standard_error_ci_corrected(values, zscore, pen_val_diff):
    """
    @param values: the values that are correlated
    @param zscore: the zscore to form the confidence interval
    @param pen_val_diff: the total penalized value (so it should be the average of the values plus some penalty)

    @returns
        the standard error of the values correcting for auto-correlation between the values
        the lower bound of the mean of the total penalized value using the standard error and the given zscore
        the upper bound of the mean of the total penalized value using the standard error and the given zscore
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

    return ase, pen_val_diff - zscore * ase, pen_val_diff + zscore * ase

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

def read_partis_annotations(annotations_file_names, chain='h', use_v=True, species='human', use_np=True, inferred_gls=None):
    """
    Function to read partis annotations csv

    @param annotations_file_names: list of paths to annotations files
    @param chain: h for heavy, k or l for kappa or lambda light chain
    @param use_v: use just the V gene or use the whole sequence?
    @param species: 'human' or 'mouse'
    @param use_np: use nonproductive sequences only
    @param inferred_gls: list of paths to partis-inferred germlines

    TODO: do we want to output intermediate genes.csv/seqs.csv files?

    TODO: do we want support for including multiple annotations files?
    is this something people do, or is this done at the partis level?

    @return gene_dict, obs_data
    """

    if not isinstance(annotations_file_names, list):
        annotations_file_names = [annotations_file_names]

    # read default germline info
    if inferred_gls is not None:
        germlines = {}
        for germline_file in set(inferred_gls):
            germlines[germline_file] = glutils.read_glfo(germline_file, chain=chain)
    else:
        glfo = glutils.read_glfo(PARTIS_PATH + '/data/germlines/' + species, chain=chain)
        inferred_gls = [None] * len(annotations_file_names)

    gene_dict = {}
    obs_data = []

    seqs_col = 'v_qr_seqs' if use_v else 'seqs'
    gene_col = 'v_gl_seq' if use_v else 'naive_seq'

    if use_np:
        # return only nonproductive sequences
        # here "nonproductive" is defined as having a stop codon or being
        # out of frame or having a mutated conserved cysteine
        good_seq = lambda seqs: seqs['stops'] or not seqs['in_frames'] or seqs['mutated_invariants']
    else:
        # return all sequences
        good_seq = lambda seqs: [True for seq in seqs[seqs_col]]

    for annotations_file, germline_file in zip(annotations_file_names, inferred_gls):
        if germline_file is not None:
            glfo = germlines[germline_file]
        with open(annotations_file, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for idx, line in enumerate(reader):
                # add goodies from partis
                utils.process_input_line(line)
                utils.add_implicit_info(glfo, line)
                # for now just use V gene for ID
                key = 'clone{}-{}'.format(*[idx, line['v_gene']])
                gene_dict[key] = line[gene_col]
                start_seq = line[gene_col].lower()
                good_seqs = [seq for seq, cond in zip(line[seqs_col], good_seq(line)) if cond]
                for end_seq in good_seqs:
                    obs_data.append(
                        ObservedSequenceMutations(
                            start_seq=start_seq[:len(end_seq)],
                            end_seq=end_seq.lower(),
                        )
                    )
    return gene_dict, obs_data

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

def get_idx_differ_by_one_character(s1, s2):
    """
    Return the index at strings s1 and s2 which differ by one character. If the strings
    are the same or differ by more than one character, return None
    """
    count_diffs = 0
    idx_differ = None
    for i, (a, b) in enumerate(zip(s1, s2)):
        if a != b:
            if count_diffs:
                return None
            count_diffs += 1
            idx_differ = i
    return idx_differ

def get_theta_sum_mask(theta, feature_mask):
    return theta[feature_mask].sum()
