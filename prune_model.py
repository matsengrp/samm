from common import *

class ModelPruner:
    def __init__(self, feat_generator, full_feat_generator):
        self.feat_generator = feat_generator
        self.full_feat_generator = full_feat_generator
        self.agg_motif_list = self.full_feat_generator.motif_list
        self.motif_list = self.feat_generator.motif_list
        self.mutating_pos_list = self.feat_generator.mutating_pos_list

    def _create_matrix_agg(self, theta):
        is_per_target = theta.shape[1] == NUM_NUCLEOTIDES + 1
        num_full_cols = NUM_NUCLEOTIDES if is_per_target else 1

        # Construct the design matrix M, which satisifies M * theta_raw = theta_agg
        M = np.zeros((
            self.full_feat_generator.feature_vec_len * num_full_cols,
            theta.size,
        ))

        def _fill_in_M(raw_theta_idx, full_m_idx):
            if not is_per_target:
                if not is_zero(theta[raw_theta_idx, 0]):
                    M[full_m_idx, raw_theta_idx] = 1
            else:
                if not is_zero(theta[raw_theta_idx, 0]):
                    for nuc_idx in range(NUM_NUCLEOTIDES):
                        M[full_m_idx + nuc_idx * self.full_feat_generator.feature_vec_len, raw_theta_idx] = 1
                for nuc_idx in range(NUM_NUCLEOTIDES):
                    if not is_zero(theta[raw_theta_idx, nuc_idx + 1]):
                        per_target_full_idx = full_m_idx + nuc_idx * self.full_feat_generator.feature_vec_len
                        per_target_raw_idx = raw_theta_idx + (nuc_idx + 1) * self.full_feat_generator.feature_vec_len
                        M[per_target_full_idx, per_target_raw_idx] = 1

        for i, feat_gen in enumerate(self.feat_generator.feat_gens):
            for m_idx, m in enumerate(feat_gen.motif_list):
                raw_theta_idx = self.feat_generator.feat_offsets[i] + m_idx

                if feat_gen.motif_len == self.full_feat_generator.motif_len:
                    # Already at maximum motif length, so nothing to combine
                    full_m_idx = self.full_feat_generator.motif_dict[m][feat_gen.left_motif_flank_len]

                    _fill_in_M(raw_theta_idx, full_m_idx)

                    # if is_per_target:
                    #     if not is_zero(theta[raw_theta_idx, 0]):
                    #         M[full_m_idx, raw_theta_idx] = 1
                    # else:
                    #     if not is_zero(theta[raw_theta_idx, 0]):
                    #         for nuc_idx in range(NUM_NUCLEOTIDES):
                    #             M[full_m_idx + nuc_idx * self.full_feat_generator.feature_vec_len, raw_theta_idx] = 1
                    #     for nuc_idx in range(NUM_NUCLEOTIDES):
                    #         if not is_zero(theta[raw_theta_idx, nuc_idx + 1]):
                    #             per_target_full_idx = full_m_idx + nuc_idx * self.full_feat_generator.feature_vec_len
                    #             per_target_raw_idx = raw_theta_idx + (nuc_idx + 1) * self.full_feat_generator.feature_vec_len
                    #             M[per_target_full_idx, per_target_raw_idx] = 1
                else:
                    # Combine hierarchical feat_gens for given left_motif_len
                    for full_feat_gen in self.full_feat_generator.feat_gens:
                        flanks = itertools.product(["a", "c", "g", "t"], repeat=full_feat_gen.motif_len - feat_gen.motif_len)
                        for f in flanks:
                            full_m = "".join(f[:feat_gen.hier_offset]) + m + "".join(f[feat_gen.hier_offset:])
                            full_m_idx = self.full_feat_generator.motif_dict[full_m][full_feat_gen.left_motif_flank_len]

                            _fill_in_M(raw_theta_idx, full_m_idx)
        return M

    def prune(self, theta):
        M = self._create_matrix_agg(theta)

        # Throw away covariates that are redundant
        # We will do this using a really stupid way: add columns of M
        # that increase rank and throw away columns that do not increase rank.
        # This can be done more efficiently.
        last_rank = 0
        good_cols = []
        bad_cols= []
        for idx in reversed(range(M.shape[1])):
            if np.sum(M[:,idx]) > 0:
                new_rank = np.linalg.matrix_rank(M[:,good_cols + [idx]])
                if last_rank != new_rank:
                    last_rank = new_rank
                    good_cols.append(idx)
                else:
                    bad_cols.append(idx)

        prune_mask = np.zeros(theta.size, dtype=bool)
        prune_mask[bad_cols] = True
        prune_mask = prune_mask.reshape(theta.shape, order="F")
        return prune_mask
