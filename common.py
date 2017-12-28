import numpy as np
import re
import random
import warnings
import itertools

from Bio import SeqIO

DEBUG = False

NUM_NUCLEOTIDES = 4
NUCLEOTIDES = "acgt"
DEGENERATE_NUCLEOTIDE = "n"
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

def process_mutating_positions(motif_len_vals, positions_mutating):
    max_motif_len = max(motif_len_vals)
    if positions_mutating is None:
        # default to central base mutating
        positions_mutating = [[m/2] for m in motif_len_vals]
        max_mut_pos = [[max_motif_len/2]]
    else:
        positions_mutating = [[int(m) for m in positions.split(',')] for positions in positions_mutating.split(':')]
        max_mut_pos = get_max_mut_pos(motif_len_vals, positions_mutating)
    return positions_mutating, max_mut_pos

def get_max_mut_pos(motif_len_vals, positions_mutating):
    max_motif_len = max(motif_len_vals)
    for motif_len, positions in zip(motif_len_vals, positions_mutating):
        for m in positions:
            assert(m in range(motif_len))
    max_mut_pos = [mut_pos for mut_pos, motif_len in zip(positions_mutating, motif_len_vals) if motif_len == max_motif_len]
    return max_mut_pos

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
    return np.random.randint(low=0, high=2**31)

def is_zero(num):
    return np.abs(num) < ZERO_THRES

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

def get_nonzero_theta_print_lines(theta, feat_gen):
    """
    @return a string that summarizes the theta vector/matrix
    """
    motif_list = feat_gen.motif_list
    motif_len = feat_gen.motif_len
    mutating_pos_list = feat_gen.mutating_pos_list

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
        for mut_pos, targets in target_dict:
            if motif in feat_generator.motif_dict:
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
    if mask_shape[1] == NUM_NUCLEOTIDES + 1:
        # Estimating a different theta vector for different target nucleotides
        # We cannot have a motif mutate to the same center nucleotide
        for i in range(len(motif_list)):
            center_motif_idx = mutating_pos_list[i]
            mutating_nucleotide = motif_list[i][center_motif_idx]
            center_nucleotide_idx = NUCLEOTIDE_DICT[mutating_nucleotide] + 1
            theta_mask[i, center_nucleotide_idx] = False
    elif mask_shape[1] == NUM_NUCLEOTIDES:
        for i in range(len(motif_list)):
            center_motif_idx = mutating_pos_list[i]
            mutating_nucleotide = motif_list[i][center_motif_idx]
            center_nucleotide_idx = NUCLEOTIDE_DICT[mutating_nucleotide]
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
        effective sample size (negative if the values are essentially constant)
    Calculate the autocorrelation, then the effective sample size, scale the standard error appropriately the effective sample size
    """
    mean = np.mean(values)
    var = np.var(values)

    # If the values are essentially constant, then the autocorrelation is zero.
    # (There are numerical stability issues if we go thru the usual calculations)
    if var < 1e-10:
        return 0, mean, mean, -1

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
        return None, -np.inf, np.inf, ess
    else:
        # Corrected standard error
        ase = np.sqrt(var/ess)
        return ase, pen_val_diff - zscore * ase, pen_val_diff + zscore * ase, ess

def soft_threshold(theta, thres):
    """
    The soft thresholding function S is zero in the range [-thresh, thresh],
    theta+thresh when theta < -thresh and theta-thresh when theta > thresh.

    @param theta: a numpy vector
    @param thres: the amount to threshold theta by
    @return theta that is soft-thresholded with constant thres
    """
    return np.maximum(theta - thres, 0) + np.minimum(theta + thres, 0)

def process_degenerates_and_impute_nucleotides(start_seq, end_seq, motif_len, threshold=0.1):
    """
    Process the degenerate characters in sequences:
    1. Replace unknown characters with "n"
    2. Collapse runs of "n"s into one of motif_len/2
    3. Replace all interior "n"s with nonmutating random nucleotide

    @param start_seq: starting sequence
    @param end_seq: ending sequence
    @param motif_len: motif length; needed to determine length of collapsed "n" run
    @param threshold: if proportion of "n"s in a sequence is larger than this then
        throw a warning

    @return processed_start_seq: starting sequence with interior "n"s collapsed and imputed
    @return processed_end_seq: ending sequence with same
    @return collapse_list: list of tuples of (index offset, start index of run of "n"s, end index of run of "n"s) for bookkeeping later
    """
    assert(len(start_seq) == len(end_seq))

    # replace all unknowns with an "n"
    processed_start_seq = re.sub('[^agctn]', 'n', start_seq)
    processed_end_seq = re.sub('[^agctn]', 'n', end_seq)

    # conform unknowns and collapse "n"s
    repl = 'n' * (motif_len/2)
    pattern = repl + '+' if motif_len > 1 else 'n'
    collapse_list = []
    if re.search('n', processed_end_seq) or re.search('n', processed_start_seq):
        # if one sequence has an "n" but the other doesn't, make them both have "n"s
        start_list = list(processed_start_seq)
        end_list = list(processed_end_seq)
        for idx in re.finditer('n', processed_end_seq):
            start_list[idx.start()] = 'n'
        processed_start_seq = ''.join(start_list)
        for idx in re.finditer('n', processed_start_seq):
            end_list[idx.start()] = 'n'
        processed_end_seq = ''.join(end_list)

        # ensure there are not too many internal "n"s
        seq_len = len(processed_end_seq)
        start_idx = re.search('[^n]', processed_end_seq).start()
        end_idx = seq_len - re.search('[^n]', processed_end_seq[::-1]).start()
        interior_end_seq = processed_end_seq[start_idx:end_idx]
        num_ns = interior_end_seq.count('n')
        seq_len = len(interior_end_seq)
        if num_ns > threshold * seq_len:
            warnings.warn("Sequence of length {0} had {1} unknown bases".format(seq_len, num_ns))

        # now collapse interior "n"s
        for match in re.finditer(pattern, interior_end_seq):
            # num "n"s removed
            # starting position of "n"s removed
            collapse_list.append((start_idx + motif_len/2, match.regs[0][0], match.regs[0][1]))

        processed_start_seq = re.sub(pattern, repl, processed_start_seq)
        processed_end_seq = re.sub(pattern, repl, processed_end_seq)

        # generate random nucleotide if an "n" occurs in the middle of a sequence
        for match in re.compile('n').finditer(processed_start_seq):
            random_nuc = random.choice(NUCLEOTIDES)
            processed_start_seq = mutate_string(processed_start_seq, match.start(), random_nuc)
            processed_end_seq = mutate_string(processed_end_seq, match.start(), random_nuc)

    return processed_start_seq, processed_end_seq, collapse_list

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

def make_zero_theta_refit_mask(theta, feat_generator):
    """
    @param theta: a fitted theta from which we determine the theta support
    @param feat_generator: the feature generator

    Figure out what the theta support is from the fitted theta
    """
    zeroed_thetas = np.array(np.abs(theta) < ZERO_THRES, dtype=bool)
    zeroed_or_inf_thetas = zeroed_thetas | (~np.isfinite(theta))
    motifs_to_remove_mask = np.sum(zeroed_or_inf_thetas, axis=1) == theta.shape[1]
    motifs_to_remove = [feat_generator.motif_list[i] for i in np.where(motifs_to_remove_mask)[0].tolist()]

    zero_theta_mask_refit = zeroed_thetas[~motifs_to_remove_mask,:]
    return zero_theta_mask_refit, motifs_to_remove, motifs_to_remove_mask

def initialize_theta(theta_shape, possible_theta_mask, zero_theta_mask):
    """
    Initialize theta
    @param possible_theta_mask: set the negative of this mask to negative infinity theta values
    @param zero_theta_mask: set the negative of this mask to negative infinity theta values
    """
    theta = np.random.randn(theta_shape[0], theta_shape[1]) * 1e-3
    # Set the impossible thetas to -inf
    theta[~possible_theta_mask] = -np.inf
    # Set particular thetas to zero upon request
    theta[zero_theta_mask] = 0
    return theta

def split_train_val(num_obs, metadata, tuning_sample_ratio, validation_column=None, val_column_idx=None):
    """
    @param num_obs: number of observations
    @param feat_generator: submotif feature generator
    @param metadata: metadata to include variables to perform validation on
    @param tuning_sample_ratio: ratio of data to place in validation set
    @param validation_column: variable to perform validation on (if None then sample randomly)
    @param val_column_idx: which index to pick for K-fold validation

    @return training and validation indices
    """
    if validation_column is None:
        # For no validation column just sample data randomly
        val_size = int(tuning_sample_ratio * num_obs)
        if tuning_sample_ratio > 0:
            val_size = max(val_size, 1)
        permuted_idx = np.random.permutation(num_obs)
        train_idx = permuted_idx[:num_obs - val_size]
        val_idx = permuted_idx[num_obs - val_size:]
    else:
        # For a validation column, sample the categories randomly based on
        # tuning_sample_ratio
        categories = set([elt[validation_column] for elt in metadata])
        num_categories = len(categories)
        val_size = int(tuning_sample_ratio * num_categories) + 1
        if tuning_sample_ratio > 0:
            val_size = max(val_size, 1)

        if val_column_idx is None:
            # sample random categories from our validation variable
            val_categories_idx = np.random.choice(len(categories), size=val_size, replace=False)
            val_categories = set([list(categories)[j] for j in val_categories_idx])
        else:
            # choose val_column_idx as validation item
            val_categories = set([list(categories)[val_column_idx]])

        train_categories = categories - val_categories
        print "val cate", val_categories
        print "train cate", train_categories
        train_idx = [idx for idx, elt in enumerate(metadata) if elt[validation_column] in train_categories]
        val_idx = [idx for idx, elt in enumerate(metadata) if elt[validation_column] in val_categories]

    return train_idx, val_idx

def create_theta_idx_mask(zero_theta_mask_refit, possible_theta_mask):
    """
    From an aggregate theta, creates a matrix with the index of the hierarchical theta
    """
    theta_idx_counter = np.ones(possible_theta_mask.shape, dtype=int) * -1
    theta_mask = ~zero_theta_mask_refit & possible_theta_mask
    idx = 0
    for col in range(theta_mask.shape[1]):
        for row in range(theta_mask.shape[0]):
            if theta_mask[row, col]:
                theta_idx_counter[row, col] = idx
                idx += 1
    return theta_idx_counter

def combine_thetas_and_get_conf_int(feat_generator, full_feat_generator, theta, zero_theta_mask, possible_theta_mask, sample_obs_info=None, col_idx=0, zstat=ZSCORE_95, add_targets=True):
    """
    Combine hierarchical and offset theta values
    """
    full_theta_size = full_feat_generator.feature_vec_len
    theta_idx_counter = create_theta_idx_mask(zero_theta_mask, possible_theta_mask)
    # stores which hierarchical theta values were used to construct the full theta
    # important for calculating covariance
    theta_index_matches = {i:[] for i in range(full_theta_size)}

    full_theta = np.zeros(full_theta_size)
    theta_lower = np.zeros(full_theta_size)
    theta_upper = np.zeros(full_theta_size)

    full_feat_gen = full_feat_generator.feat_gens[0]
    for i, feat_gen in enumerate(feat_generator.feat_gens):
        for m_idx, m in enumerate(feat_gen.motif_list):
            raw_theta_idx = feat_generator.feat_offsets[i] + m_idx

            if col_idx != 0 and add_targets:
                m_theta = theta[raw_theta_idx, 0] + theta[raw_theta_idx, col_idx]
            else:
                m_theta = theta[raw_theta_idx, col_idx]

            if feat_gen.motif_len == full_feat_generator.motif_len:
                assert(full_feat_gen.left_motif_flank_len == feat_gen.left_motif_flank_len)
                # Already at maximum motif length, so nothing to combine
                full_m_idx = full_feat_generator.motif_dict[m][full_feat_gen.left_motif_flank_len]
                full_theta[full_m_idx] += m_theta

                if theta_idx_counter[raw_theta_idx, 0] != -1:
                    theta_index_matches[full_m_idx].append(theta_idx_counter[raw_theta_idx, 0])
                if col_idx != 0 and theta_idx_counter[raw_theta_idx, col_idx] != -1:
                    theta_index_matches[full_m_idx].append(theta_idx_counter[raw_theta_idx, col_idx])
            else:
                # Combine hierarchical feat_gens for given left_motif_len
                flanks = itertools.product(["a", "c", "g", "t"], repeat=full_feat_gen.motif_len - feat_gen.motif_len)
                for f in flanks:
                    full_m = "".join(f[:feat_gen.hier_offset]) + m + "".join(f[feat_gen.hier_offset:])
                    full_m_idx = full_feat_generator.motif_dict[full_m][full_feat_gen.left_motif_flank_len]
                    full_theta[full_m_idx] += m_theta

                    if theta_idx_counter[raw_theta_idx, 0] != -1:
                        theta_index_matches[full_m_idx].append(theta_idx_counter[raw_theta_idx, 0])
                    if col_idx != 0 and theta_idx_counter[raw_theta_idx, col_idx] != -1:
                        theta_index_matches[full_m_idx].append(theta_idx_counter[raw_theta_idx, col_idx])

    if sample_obs_info is not None:
        # Make the aggregation matrix
        agg_matrix = np.zeros((full_theta.size, np.max(theta_idx_counter) + 1))
        for full_theta_idx, matches in theta_index_matches.iteritems():
            agg_matrix[full_theta_idx, matches] = 1

        # Try two estimates of the obsersed information matrix
        tts = [0.5 * (sample_obs_info + sample_obs_info.T), sample_obs_info]
        for tt in tts:
            cov_mat_full = np.dot(np.dot(agg_matrix, np.linalg.pinv(tt)), agg_matrix.T)
            if not np.any(np.diag(cov_mat_full) < 0):
                break
        if np.any(np.diag(cov_mat_full) < 0):
            raise ValueError("Some variance estimates were negative: %d neg var" % np.sum(np.diag(cov_mat_full) < 0))

        full_std_err = np.sqrt(np.diag(cov_mat_full))
        theta_lower = full_theta - zstat * full_std_err
        theta_upper = full_theta + zstat * full_std_err

    return full_theta, theta_lower, theta_upper

def create_aggregate_theta(hier_feat_generator, agg_feat_generator, theta, zero_theta_mask, possible_theta_mask, keep_col0=True, add_targets=True):
    def _combine_thetas(col_idx):
        theta_col, _, _ = combine_thetas_and_get_conf_int(
            hier_feat_generator,
            agg_feat_generator,
            theta,
            zero_theta_mask,
            possible_theta_mask,
            sample_obs_info=None,
            col_idx=col_idx,
            add_targets=add_targets,
        )
        return theta_col.reshape((theta_col.size, 1))

    if theta.shape[1] == 1:
        theta_cols = [_combine_thetas(col_idx) for col_idx in range(1)]
    else:
        start_idx = 0 if keep_col0 else 1
        theta_cols = [_combine_thetas(col_idx) for col_idx in range(start_idx, theta.shape[1])]
    agg_theta = np.hstack(theta_cols)
    return agg_theta

def pick_best_model(fitted_models):
    """
    Select the one with largest (pseudo) log lik ratio
    """
    if isinstance(fitted_models[0], list):
        good_models = [f_model for f_model_list in fitted_models for f_model in f_model_list if f_model.has_refit_data]
    else:
        good_models = [f_model for f_model in fitted_models if f_model.has_refit_data]
    if len(good_models) == 0:
        return None

    for max_idx in reversed(range(len(good_models))):
        if good_models[max_idx].log_lik_ratio_lower_bound > 0:
            break
    best_model = good_models[max_idx]
    return best_model
