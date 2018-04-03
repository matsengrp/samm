import itertools
import numpy as np

from common import NUCLEOTIDE_SET, get_max_mut_pos, get_zero_theta_mask, create_theta_idx_mask, ZSCORE_95, NUM_NUCLEOTIDES
from combined_feature_generator import CombinedFeatureGenerator
from feature_generator import MultiFeatureMutationStep
from submotif_feature_generator import SubmotifFeatureGenerator
from scipy.sparse import hstack

class HierarchicalMotifFeatureGenerator(CombinedFeatureGenerator):
    def __init__(self, motif_lens, model_masks=None, left_motif_flank_len_list=None):
        """
        @param motif_lens: list of odd-numbered motif lengths
        @param model_masks: dictionary whose keys are left_flank_len values and whose values are lists of motifs (strings) where the motif and left_flank_len have been zeroed out (completely - all targets are zeroed out)
        @param left_motif_flank_len_list: list of lengths of left motif flank; 0 will mutate the leftmost position, 1 the next to left, etc.
        """

        self.motif_lens = motif_lens

        if left_motif_flank_len_list is None:
            # default to central base mutating
            left_motif_flank_len_list = []
            for motif_len in motif_lens:
                left_motif_flank_len_list.append([motif_len/2])

        self.max_motif_len = max(motif_lens)
        self.motif_len = self.max_motif_len
        self.left_motif_flank_len = get_max_mut_pos(motif_lens, left_motif_flank_len_list)

        # Find the maximum left and right motif flank lengths to pass to SubmotifFeatureGenerator
        # in order to update all the relevant features
        all_right_flanks = [m - flank_len - 1 \
                for m, flank_len_list in zip(motif_lens, left_motif_flank_len_list) \
                for flank_len in flank_len_list]
        self.max_left_motif_flank_len = max(sum(left_motif_flank_len_list, []))
        self.max_right_motif_flank_len = max(all_right_flanks)

        if model_masks is None:
            # default to not removing any features
            feats_to_remove = []
        else:
            feats_to_remove = model_masks.feats_to_remove()

        # Create list of feature generators for different motif lengths and different flank lengths
        self.feat_gens = []
        for motif_len, left_motif_flank_lens in zip(motif_lens, left_motif_flank_len_list):
            for left_motif_flank_len in left_motif_flank_lens:
                curr_feats_to_remove = [label for label in feats_to_remove if int(label.split(' ')[2]) == left_motif_flank_len]
                self.feat_gens.append(
                        SubmotifFeatureGenerator(
                            motif_len=motif_len,
                            feats_to_remove=curr_feats_to_remove,
                            left_motif_flank_len=left_motif_flank_len,
                            hier_offset=self.max_left_motif_flank_len - left_motif_flank_len,
                            left_update_region=self.max_left_motif_flank_len,
                            right_update_region=self.max_right_motif_flank_len,
                        )
                    )

        feat_offsets = [feat_gen.feature_vec_len for feat_gen in self.feat_gens]
        self.feat_offsets = np.cumsum([0] + feat_offsets)[:-1]
        self.num_feat_gens = len(self.feat_gens)
        self.feature_vec_len = np.sum(feat_offsets)

        # construct motif dictionary and lists of parameters
        self.motif_list = []
        self.mutating_pos_list = []
        self.feature_label_list = []
        for f in self.feat_gens:
            self.motif_list += f.motif_list
            self.mutating_pos_list += [f.left_motif_flank_len] * len(f.motif_list)
            self.feature_label_list += f.feature_label_list

    def _update_feature_generator_after_removing(self, model_masks):
        """
        so we don't have to create a whole new feature vector
        """
        # Create list of feature generators for different motif lengths and different flank lengths
        feats_to_remove = model_masks.set_feats_to_remove()
        old_feat_gens = self.feat_gens
        self.feat_gens = []
        for feat_gen in old_feat_gens:
            curr_feats_to_remove = [label for label in feats_to_remove if int(label.split(' ')[2]) == feat_gen.left_motif_flank_len]
            feat_gen._update_motifs_after_removing(curr_feats_to_remove)
            self.feat_gens.append(feat_gen)

        feat_offsets = [feat_gen.feature_vec_len for feat_gen in self.feat_gens]
        self.feat_offsets = np.cumsum([0] + feat_offsets)[:-1]
        self.feature_vec_len = np.sum(feat_offsets)

        # construct motif dictionary and lists of parameters
        self.motif_list = []
        self.mutating_pos_list = []
        self.feature_label_list = []
        for i, f in enumerate(self.feat_gens):
            self.motif_list += f.motif_list
            self.mutating_pos_list += [f.left_motif_flank_len] * len(f.motif_list)
            self.feature_label_list += f.feature_label_list

    def get_possible_motifs_to_targets(self, mask_shape):
        """
        @return a boolean matrix with possible mutations as True, impossible mutations as False
        """
        theta_mask = np.ones(mask_shape, dtype=bool)
        if mask_shape[1] == NUM_NUCLEOTIDES + 1:
            # Estimating a different theta vector for different target nucleotides
            # We cannot have a motif mutate to the same center nucleotide
            for i in range(len(self.motif_list)):
                center_motif_idx = self.mutating_pos_list[i]
                mutating_nucleotide = self.motif_list[i][center_motif_idx]
                center_nucleotide_idx = NUCLEOTIDE_DICT[mutating_nucleotide] + 1
                theta_mask[i, center_nucleotide_idx] = False
        elif mask_shape[1] == NUM_NUCLEOTIDES:
            for i in range(len(self.motif_list)):
                center_motif_idx = self.mutating_pos_list[i]
                mutating_nucleotide = self.motif_list[i][center_motif_idx]
                center_nucleotide_idx = NUCLEOTIDE_DICT[mutating_nucleotide]
                theta_mask[i, center_nucleotide_idx] = False

        return theta_mask

    # why keep this column specific?
    def combine_thetas_and_get_conf_int(self, theta, sample_obs_info=None, col_idx=0, zstat=ZSCORE_95, add_targets=True):
        """
        Combine hierarchical and offset theta values
        """
        full_feat_generator = SubmotifFeatureGenerator(
            motif_len=self.motif_len,
            left_motif_flank_len=self.left_motif_flank_len,
        )
        full_theta_size = full_feat_generator.feature_vec_len
        zero_theta_mask = get_zero_theta_mask(theta)
        theta_idx_counter = create_theta_idx_mask(zero_theta_mask, possible_theta_mask)
        # stores which hierarchical theta values were used to construct the full theta
        # important for calculating covariance
        theta_index_matches = {i:[] for i in range(full_theta_size)}

        full_theta = np.zeros(full_theta_size)
        theta_lower = np.zeros(full_theta_size)
        theta_upper = np.zeros(full_theta_size)

        for i, feat_gen in enumerate(self.feat_gens):
            for m_idx, m in enumerate(feat_gen.motif_list):
                raw_theta_idx = feat_generator.feat_offsets[i] + m_idx

                if col_idx != 0 and add_targets:
                    m_theta = theta[raw_theta_idx, 0] + theta[raw_theta_idx, col_idx]
                else:
                    m_theta = theta[raw_theta_idx, col_idx]

                if feat_gen.motif_len == full_feat_generator.motif_len:
                    assert(full_feat_generator.left_motif_flank_len == feat_gen.left_motif_flank_len)
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
                        full_m = "".join(f[:feat_gen.hier_offset]) + m + "".join(f[feat_gen.hier_offset:])
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
