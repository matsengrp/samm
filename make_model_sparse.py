import time
import numpy as np
from cvxpy import *
from common import *

class SparseModelMaker:
    """
    Objective function: make a minimum L1 version of the inputted model

    min ||raw_theta||_1
    s.t. agg_matrix * raw_theta = agg_theta_cp
    """

    @staticmethod
    def _create_agg_matrices(feat_generator, full_feat_generator):
        """
        For a same-target model, agg_matrix_hier * raw_theta = agg_theta_cp
        For a per-target model, agg_matrix_hier * raw_theta * agg_matrix_per_target = agg_theta_cp
        """
        full_feat_gen = full_feat_generator.feat_gens[0]

        agg_matrix_hier = np.zeros((full_feat_generator.feature_vec_len, feat_generator.feature_vec_len))
        for i, feat_gen in enumerate(feat_generator.feat_gens):
            for m_idx, m in enumerate(feat_gen.motif_list):
                raw_theta_idx = feat_generator.feat_offsets[i] + m_idx
                if feat_gen.motif_len == full_feat_generator.motif_len:
                    # Already at maximum motif length, so nothing to combine
                    full_m_idx = full_feat_generator.motif_dict[m][full_feat_gen.left_motif_flank_len]
                    agg_matrix_hier[full_m_idx, raw_theta_idx] = 1
                else:
                    # Combine hierarchical feat_gens for given left_motif_len
                    flanks = itertools.product(["a", "c", "g", "t"], repeat=full_feat_gen.motif_len - feat_gen.motif_len)
                    for f in flanks:
                        full_m = "".join(f[:feat_gen.hier_offset]) + m + "".join(f[feat_gen.hier_offset:])
                        full_m_idx = full_feat_generator.motif_dict[full_m][full_feat_gen.left_motif_flank_len]
                        agg_matrix_hier[full_m_idx, raw_theta_idx] = 1

        agg_matrix_per_target = np.vstack((np.ones((1, NUM_NUCLEOTIDES)), np.eye(NUM_NUCLEOTIDES)))
        return np.matrix(agg_matrix_hier), np.matrix(agg_matrix_per_target)

    @staticmethod
    def solve(agg_theta, feat_generator, full_feat_generator, raw_theta=None):
        """
        @param agg_theta: please return the sparse version of this aggregate theta
        """
        agg_theta_cp = np.copy(agg_theta)

        per_target = agg_theta_cp.shape[1] == NUM_NUCLEOTIDES
        num_theta_col = NUM_NUCLEOTIDES + 1 if per_target else 1
        theta_var = Variable(feat_generator.feature_vec_len, num_theta_col)

        agg_matrix_hier, agg_matrix_per_target = SparseModelMaker._create_agg_matrices(feat_generator, full_feat_generator)

        agg_possible_mask = agg_theta_cp != -np.inf
        if per_target:
            agg_theta_cp[agg_theta_cp == -np.inf] = 0
            constraints = [
                agg_matrix_hier * theta_var * agg_matrix_per_target == agg_theta_cp
            ]
        else:
            constraints = [
                agg_matrix_hier * theta_var == agg_theta_cp
            ]

        if raw_theta is not None:
            raw_theta_cp = np.copy(raw_theta)
            raw_theta_cp[raw_theta_cp == -np.inf] = 0
            if per_target:
                agg_formula = agg_matrix_hier * np.matrix(raw_theta_cp) * agg_matrix_per_target
            else:
                agg_formula = agg_matrix_hier * np.matrix(raw_theta_cp)
            assert(np.linalg.norm(agg_formula[agg_possible_mask] - agg_theta_cp[agg_possible_mask]) < ZERO_THRES)

        problem = Problem(Minimize(norm(theta_var, 1)), constraints)
        problem.solve()
        assert(problem.status == OPTIMAL)

        theta_soln = theta_var.value
        theta_soln[np.abs(theta_soln) < ZERO_THRES] = 0
        return theta_soln
