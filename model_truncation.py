import numpy as np
from common import ZERO_THRES
from prune_model import ModelPruner
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator

class ModelTruncation:
    def __init__(self, theta, feat_generator):
        """
        @param theta: a fitted theta from which we determine the theta support
        @param feat_generator: the feature generator
        """
        # First we must figure out a nonsingular support
        full_feat_gen = HierarchicalMotifFeatureGenerator(
            motif_lens=[max(feat_generator.motif_lens)],
            # HACK: THIS SHOULD TAKE IN THE REAL POSITION MUTATING
        )
        pruner = ModelPruner(feat_generator, full_feat_gen, theta)
        self.removal_mask = pruner.prune()

        zeroed_thetas = (np.abs(theta) < ZERO_THRES)  | self.removal_mask
        zeroed_or_inf_thetas = zeroed_thetas | (~np.isfinite(theta))
        self.feats_to_remove_mask = np.sum(zeroed_or_inf_thetas, axis=1) == theta.shape[1]

        self.feats_to_remove = {mut_pos: [] for mut_pos in feat_generator.all_mut_pos}
        for i in np.where(self.feats_to_remove_mask)[0].tolist():
            motif = feat_generator.motif_list[i]
            mut_pos = feat_generator.mutating_pos_list[i]
            self.feats_to_remove[mut_pos].append(motif)

        self.zero_theta_mask_refit = zeroed_thetas[~self.feats_to_remove_mask,:]

