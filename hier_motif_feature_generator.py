import itertools
import numpy as np

from common import NUCLEOTIDE_SET, get_max_mut_pos, get_zero_theta_mask, create_theta_idx_mask, ZSCORE_95, NUM_NUCLEOTIDES, NUCLEOTIDE_DICT
from combined_feature_generator import CombinedFeatureGenerator
from feature_generator import MultiFeatureMutationStep
from motif_feature_generator import MotifFeatureGenerator
from scipy.sparse import hstack

class HierarchicalMotifFeatureGenerator(CombinedFeatureGenerator):
    """
    A hierarchical motif model is a special case of a CombinedFeatureGenerator.
    Included in this class is everything that will be needed to combined MotifFeatureGenerators into a Hierarchical generator, and the
    inputs are included as in the standard motif definitions, i.e., we pass in the left flank length instead of the distance to the motif
    start (which is what MotifFeatureGenerator takes).

    All code that previously uses HierarchicalMotifFeatureGenerator should still work as-is.
    """
    def __init__(self, motif_lens, feats_to_remove=[], left_motif_flank_len_list=None):
        """
        @param motif_lens: list of odd-numbered motif lengths
        @param feats_to_remove: list of feature info tuples to remove
        @param left_motif_flank_len_list: list of lengths of left motif flank; 0 will mutate the leftmost position, 1 the next to left, etc.
        """

        self.motif_lens = motif_lens

        if left_motif_flank_len_list is None:
            # default to central base mutating
            left_motif_flank_len_list = []
            for motif_len in motif_lens:
                left_motif_flank_len_list.append([motif_len/2])
        else:
            # make sure we're actually making a hierarchical model
            for left_motif_lens, motif_len in zip(left_motif_flank_len_list, motif_lens):
                for left_motif_len in left_motif_lens:
                    assert(left_motif_len in range(motif_len))

        self.max_motif_len = max(motif_lens)
        self.motif_len = self.max_motif_len
        self.left_motif_flank_len = get_max_mut_pos(motif_lens, left_motif_flank_len_list)

        # Find the maximum left and right motif flank lengths to pass to MotifFeatureGenerator
        # in order to update all the relevant features
        all_right_flanks = [m - flank_len - 1 \
                for m, flank_len_list in zip(motif_lens, left_motif_flank_len_list) \
                for flank_len in flank_len_list]
        self.max_left_motif_flank_len = max(sum(left_motif_flank_len_list, []))
        self.max_right_motif_flank_len = max(all_right_flanks)

        self.left_update_region = self.max_left_motif_flank_len
        self.right_update_region = self.max_right_motif_flank_len

        # Create list of feature generators for different motif lengths and different flank lengths
        self.feat_gens = []
        for motif_len, left_motif_flank_lens in zip(motif_lens, left_motif_flank_len_list):
            for left_motif_flank_len in left_motif_flank_lens:
                self.feat_gens.append(
                        MotifFeatureGenerator(
                            motif_len=motif_len,
                            distance_to_start_of_motif=-left_motif_flank_len,
                            flank_len_offset=self.max_left_motif_flank_len - left_motif_flank_len,
                        )
                    )

        self.feats_to_remove = feats_to_remove
        self.update_feats_after_removing(feats_to_remove)

    def update_feats_after_removing(self, feats_to_remove):
        """
        Updates feature generator properties after removing features.
        This feature generator also has motif_list and mutating_pos_list that must be updated.

        @param feats_to_remove: list of feature info elements to remove
        """
        # Create list of feature generators for different motif lengths and different flank lengths
        old_feat_gens = self.feat_gens
        self.feat_gens = []
        self.feature_info_list = []
        for feat_gen in old_feat_gens:
            feat_gen.update_feats_after_removing(feats_to_remove)
            self.feat_gens.append(feat_gen)
            self.feature_info_list += feat_gen.feature_info_list

        feat_offsets = [feat_gen.feature_vec_len for feat_gen in self.feat_gens]
        self.feat_offsets = np.cumsum([0] + feat_offsets)[:-1]
        self.num_feat_gens = len(self.feat_gens)
        self.feature_vec_len = np.sum(feat_offsets)

        # construct motif dictionary and lists of parameters
        self.motif_list = []
        self.mutating_pos_list = []
        for f in self.feat_gens:
            self.motif_list += f.motif_list
            self.mutating_pos_list += [-f.distance_to_start_of_motif] * len(f.motif_list)

    def get_possible_motifs_to_targets(self, mask_shape):
        """
        @return a boolean matrix with possible mutations as True, impossible mutations as False
        """
        # Estimating a different theta vector for different target nucleotides
        # We cannot have a motif mutate to the same center nucleotide
        theta_mask = np.ones(mask_shape, dtype=bool)
        if mask_shape[1] > 1:
            for i, (motif, mut_pos) in enumerate(zip(self.motif_list, self.mutating_pos_list)):
                mutating_nucleotide = motif[mut_pos]
                center_nucleotide_idx = NUCLEOTIDE_DICT[mutating_nucleotide]
                if mask_shape[1] == NUM_NUCLEOTIDES + 1:
                    center_nucleotide_idx += 1
                theta_mask[i, center_nucleotide_idx] = False
        return theta_mask

    def combine_thetas_and_get_conf_int(self, theta, sample_obs_info=None, col_idx=0, zstat=ZSCORE_95, add_targets=True):
        """
        Combine hierarchical and offset theta values
        """
        full_feat_generator = MotifFeatureGenerator(
            motif_len=self.motif_len,
            distance_to_start_of_motif=-self.max_left_motif_flank_len,
        )
        full_theta_size = full_feat_generator.feature_vec_len
        zero_theta_mask = get_zero_theta_mask(theta)
        possible_theta_mask = self.get_possible_motifs_to_targets(theta.shape)
        theta_idx_counter = create_theta_idx_mask(zero_theta_mask, possible_theta_mask)
        # stores which hierarchical theta values were used to construct the full theta
        # important for calculating covariance
        theta_index_matches = {i:[] for i in range(full_theta_size)}

        full_theta = np.zeros(full_theta_size)
        theta_lower = np.zeros(full_theta_size)
        theta_upper = np.zeros(full_theta_size)

        for i, feat_gen in enumerate(self.feat_gens):
            for m_idx, m in enumerate(feat_gen.motif_list):
                raw_theta_idx = self.feat_offsets[i] + m_idx

                if col_idx != 0 and add_targets:
                    m_theta = theta[raw_theta_idx, 0] + theta[raw_theta_idx, col_idx]
                else:
                    m_theta = theta[raw_theta_idx, col_idx]

                if feat_gen.motif_len == full_feat_generator.motif_len:
                    assert(self.max_left_motif_flank_len == -feat_gen.distance_to_start_of_motif)
                    # Already at maximum motif length, so nothing to combine
                    full_m_idx = full_feat_generator.motif_dict[m]
                    full_theta[full_m_idx] += m_theta

                    if theta_idx_counter[raw_theta_idx, 0] != -1:
                        theta_index_matches[full_m_idx].append(theta_idx_counter[raw_theta_idx, 0])
                    if col_idx != 0 and theta_idx_counter[raw_theta_idx, col_idx] != -1:
                        theta_index_matches[full_m_idx].append(theta_idx_counter[raw_theta_idx, col_idx])
                else:
                    # Combine hierarchical feat_gens for given left_motif_len
                    flanks = itertools.product(NUCLEOTIDE_SET, repeat=full_feat_generator.motif_len - feat_gen.motif_len)
                    for f in flanks:
                        full_m = "".join(f[:feat_gen.flank_len_offset]) + m + "".join(f[feat_gen.flank_len_offset:])
                        full_m_idx = full_feat_generator.motif_dict[full_m]
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

    def create_aggregate_theta(self, theta, keep_col0=True, add_targets=True):
        def _combine_thetas(col_idx):
            theta_col, _, _ = self.combine_thetas_and_get_conf_int(
                theta,
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
