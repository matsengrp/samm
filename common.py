import numpy as np
import pandas as pd
import re
import random
import warnings

from Bio import SeqIO

DEBUG = False

NUM_NUCLEOTIDES = 4
NUCLEOTIDES = "acgt"
NUCLEOTIDE_SET = set(["a", "c", "g", "t"])
NUCLEOTIDE_DICT = {
    "a": 0,
    "c": 1,
    "g": 2,
    "t": 3,
}
ZSCORE = 1.65
ZSCORE_95 = 1.96
ZERO_THRES = 1e-6
MAX_TRIALS = 10

COMPLEMENT_DICT = {
    'A': 'T',
    'G': 'C',
    'C': 'G',
    'T': 'A',
    'Y': 'R',
    'R': 'Y',
    'S': 'S',
    'W': 'W',
    'M': 'K',
    'K': 'M',
    'B': 'V',
    'D': 'H',
    'H': 'D',
    'V': 'B',
    'N': 'N',
}

DEGENERATE_BASE_DICT = {
    'A': 'a',
    'G': 'g',
    'C': 'c',
    'T': 't',
    'Y': '[ct]',
    'R': '[ag]',
    'S': '[gc]',
    'W': '[at]',
    'M': '[ac]',
    'K': '[gt]',
    'B': '[cgt]',
    'D': '[agt]',
    'H': '[act]',
    'V': '[acg]',
    'N': '[agct]',
}

HOT_COLD_SPOT_REGS = [
        {'central': 'G', 'left_flank': 'R', 'right_flank': 'YW', 'hot_or_cold': 'hot'},
        {'central': 'A', 'left_flank': 'W', 'right_flank': '', 'hot_or_cold': 'hot'},
        {'central': 'C', 'left_flank': 'SY', 'right_flank': '', 'hot_or_cold': 'cold'},
        {'central': 'G', 'left_flank': '', 'right_flank': 'YW', 'hot_or_cold': 'hot'},
]
INT8_MAX = 127

FUSED_LASSO_PENALTY_RATIO = [1./4, 1./2, 1., 2., 4.]

def get_batched_list(my_list, num_batches):
    batch_size = max(len(my_list)/num_batches, 1)
    batched_list = []
    for i in range(num_batches + 1):
        additional_batch = my_list[i * batch_size: (i+1) * batch_size]
        if len(additional_batch):
            batched_list.append(additional_batch)
    return batched_list

def return_complement(kmer):
    return ''.join([COMPLEMENT_DICT[nuc] for nuc in kmer[::-1]])

def compute_known_hot_and_cold(hot_or_cold_dicts, motif_len=5, half_motif_len=2):
    """
    Known hot and cold spots were constructed on a 5mer model, so "N" pad
    longer motifs and subset shorter ones
    """
    kmer_list = []
    hot_or_cold_list = []
    hot_or_cold_complements = []
    for spot in hot_or_cold_dicts:
        hot_or_cold_complements.append({'central': return_complement(spot['central']),
                'left_flank': return_complement(spot['right_flank']),
                'right_flank': return_complement(spot['left_flank']),
                'hot_or_cold': spot['hot_or_cold']})

    for spot in hot_or_cold_dicts + hot_or_cold_complements:
        if len(spot['left_flank']) > half_motif_len or \
            len(spot['right_flank']) > motif_len - half_motif_len - 1:
                # this hot/cold spot is not a part of our motif size
                continue

        left_pad = spot['left_flank'].rjust(half_motif_len, 'N')
        right_pad = spot['right_flank'].ljust(motif_len - half_motif_len - 1, 'N')
        kmer_list.append(left_pad + spot['central'] + right_pad)
        hot_or_cold_list.append(spot['hot_or_cold'])

    hot_cold_regs = []
    for kmer, hot_or_cold in zip(kmer_list, hot_or_cold_list):
        hot_cold_regs.append([' - '.join([kmer.replace('N', ''), hot_or_cold]),
            ''.join([DEGENERATE_BASE_DICT[nuc] for nuc in kmer])])
    return hot_cold_regs

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

def get_num_nonzero(theta):
    nonzero_idx = np.logical_and(np.isfinite(theta), np.abs(theta) > ZERO_THRES)
    return np.sum(nonzero_idx)

def get_num_unique_theta(theta):
    zero_theta_mask = np.abs(theta) < ZERO_THRES
    nonzero_idx = np.logical_and(np.isfinite(theta), ~zero_theta_mask)
    unique_theta = set(theta[nonzero_idx].flatten().tolist())
    num_unique = len(unique_theta)
    if np.any(zero_theta_mask):
        num_unique += 1
    return num_unique

def is_re_match(regex, submotif):
    match_res = re.match(regex, submotif)
    return match_res is not None

def get_nonzero_theta_print_lines(theta, motif_list, mutating_pos_list, motif_len):
    """
    @return a string that summarizes the theta vector/matrix
    """
    lines = []
    mutating_pos_set = list(set(mutating_pos_list))
    known_hot_cold = [compute_known_hot_and_cold(HOT_COLD_SPOT_REGS, motif_len, half_motif_len) for half_motif_len in mutating_pos_set]
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            if np.isfinite(theta[i,j]) and np.abs(theta[i,j]) > ZERO_THRES:
                # print the whole line if any element in the theta is nonzero
                motif = motif_list[i]
                pos_idx = mutating_pos_set.index(mutating_pos_list[i])
                hot_cold_matches = ""
                for spot_name, spot_regex in known_hot_cold[pos_idx]:
                    if is_re_match(spot_regex, motif):
                        hot_cold_matches = " -- " + spot_name
                        break
                thetas = theta[i,]
                lines.append((
                    thetas[np.isfinite(thetas)].sum(),
                    "%s (%s%s) pos %s" % (thetas, motif_list[i], hot_cold_matches, mutating_pos_list[i]),
                ))
                break
    sorted_lines = sorted(lines, key=lambda s: s[0])
    return "\n".join([l[1] for l in sorted_lines])

def print_known_cold_hot_spot(motif, known_hot_cold_regexs):
    for spot_name, spot_regex in known_hot_cold_regexs:
        if is_re_match(spot_regex, motif):
            return spot_name
    return None

def get_zero_theta_mask(target_pairs_to_remove, feat_generator, theta_shape):
    """
    Combines `target_pairs_to_remove` from `read_zero_motif_csv` and with the feature generator

    @return a boolean matrix with fixed zero theta values True, others as False
    """
    zero_theta_mask = np.zeros(theta_shape, dtype=bool)
    if theta_shape[1] == 1:
        return zero_theta_mask

    for motif, target_dict in target_pairs_to_remove.iteritems():
        if motif in feat_generator.motif_dict:
            for mut_pos, targets in target_dict:
                for nuc in targets:
                    motif_idx = feat_generator.motif_dict[motif][mut_pos]
                    if nuc == "n":
                        zero_theta_mask[motif_idx, 0] = 1
                    else:
                        zero_theta_mask[motif_idx, NUCLEOTIDE_DICT[nuc] + 1] = 1
    return zero_theta_mask

def get_possible_motifs_to_targets(motif_list, mask_shape, mutating_pos_list):
    """
    @param motif_list: list of motifs - assumes that the first few theta rows correspond to these motifs
    @param mask_shape: shape of the theta matrix
    @param mutating_pos_list: list of mutating positions

    @return a boolean matrix with possible mutations as True, impossible mutations as False
    """
    theta_mask = np.ones(mask_shape, dtype=bool)
    if mask_shape[1] == 1:
        # Estimating a single theta vector - then we should estimate all theta values
        return theta_mask

    # Estimating a different theta vector for different target nucleotides
    # We cannot have a motif mutate to the same center nucleotide
    for i in range(len(motif_list)):
        center_motif_idx = mutating_pos_list[i]
        if mask_shape[1] == NUM_NUCLEOTIDES:
            center_nucleotide_idx = NUCLEOTIDE_DICT[motif_list[i][center_motif_idx]]
        else:
            center_nucleotide_idx = NUCLEOTIDE_DICT[motif_list[i][center_motif_idx]] + 1
        theta_mask[i, center_nucleotide_idx] = False
    return theta_mask

def mutate_string(begin_str, mutate_pos, mutate_value):
    """
    Mutate a string
    """
    return "%s%s%s" % (begin_str[:mutate_pos], mutate_value, begin_str[mutate_pos + 1:])

def unmutate_string(mutated_str, unmutate_pos, orig_nuc):
    return (
        mutated_str[:unmutate_pos]
        + orig_nuc
        + mutated_str[unmutate_pos + 1:]
    )

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
    @param pen_val_diff: difference of the total penalized values (the negative log likelihood plus some penalty)

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

    if var/ess < 0:
        return None, -np.inf, np.inf
    else:
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

def read_germline_file(fasta):
    """
    Read fasta file containing germlines

    @return dataframe with column "gene" for the name of the germline gene and
    "base" for the nucleotide content
    """

    with open(fasta) as fasta_file:
        genes = []
        bases = []
        for seq_record in SeqIO.parse(fasta_file, 'fasta'):
            genes.append(seq_record.id)
            bases.append(str(seq_record.seq))

    return pd.DataFrame({'base': bases}, index=genes)

def process_degenerates_and_impute_nucleotides(start_seq, end_seq, motif_len, threshold=0.1):
    """
    Process the degenerate characters in sequences:
    1. Replace unknown characters with "n"
    2. Remove padding "n"s at beginning and end of sequence
    3. Collapse runs of "n"s into one of motif_len/2
    4. Replace all interior "n"s with nonmutating random nucleotide

    @param start_seq: starting sequence
    @param end_seq: ending sequence
    @param motif_len: motif length; needed to determine length of collapsed "n" run
    @param threshold: if proportion of "n"s in a sequence is larger than this then
        throw a warning
    """

    assert(len(start_seq) == len(end_seq))

    # replace all unknowns with an "n"
    processed_start_seq = re.sub('[^agctn]', 'n', start_seq)
    processed_end_seq = re.sub('[^agctn]', 'n', end_seq)

    # conform unknowns and collapse "n"s
    repl = 'n' * (motif_len/2)
    pattern = repl + '+' if motif_len > 1 else 'n'
    if re.search('n', processed_end_seq) or re.search('n', processed_start_seq):
        # turn known bases in start_seq to "n"s and collapse degenerates
        start_list = list(processed_start_seq)
        end_list = list(processed_end_seq)
        for idx in re.finditer('n', processed_end_seq):
            start_list[idx.start()] = 'n'
        processed_start_seq = ''.join(start_list)
        for idx in re.finditer('n', processed_start_seq):
            end_list[idx.start()] = 'n'
        processed_end_seq = ''.join(end_list)

        # first remove beginning and trailing "n"s
        processed_start_seq = re.sub('^n+|n+$', '', processed_start_seq)
        processed_end_seq = re.sub('^n+|n+$', '', processed_end_seq)

        # ensure there are not too many internal "n"s
        num_ns = processed_end_seq.count('n')
        seq_len = len(processed_end_seq)
        if num_ns > threshold * seq_len:
            warnings.warn("Sequence of length {0} had {1} unknown bases".format(seq_len, num_ns))

        # now collapse interior "n"s
        processed_start_seq = re.sub(pattern, repl, processed_start_seq)
        processed_end_seq = re.sub(pattern, repl, processed_end_seq)

        # generate random nucleotide if an "n" occurs in the middle of a sequence
        for match in re.compile('n').finditer(processed_start_seq):
            random_nuc = random.choice(NUCLEOTIDES)
            processed_start_seq = mutate_string(processed_start_seq, match.start(), random_nuc)
            processed_end_seq = mutate_string(processed_end_seq, match.start(), random_nuc)

    return processed_start_seq, processed_end_seq

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

def is_central_kmer_shared(s1, s2, k=5):
    """
    Return if central k-mer matches
    @param k: must be odd
    """
    if k % 2 == 0:
        return False

    str_len = len(s1)
    if str_len - k <= 0:
        return False
    else:
        offset = (str_len - k)/2
        central1 = s1[offset:-offset]
        central2 = s2[offset:-offset]
        return central1 == central2

def get_target_col(sample, mutation_pos):
    """
    @param sample: ObservedSequenceMutations
    @returns the index of the column in the hazard rate matrix for the target nucleotide
    """
    return NUCLEOTIDE_DICT[sample.end_seq[mutation_pos]] + 1
