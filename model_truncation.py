import numpy as np
from common import ZERO_THRES

class ModelTruncation:
    def __init__(self, theta, feat_generator):
        """
        @param theta: a fitted theta from which we determine the theta support
        @param feat_generator: the feature generator

        @return
        """

        zeroed_thetas = np.abs(theta) < ZERO_THRES
        zeroed_or_inf_thetas = zeroed_thetas | (~np.isfinite(theta))
        self.feats_to_remove_mask = np.sum(zeroed_or_inf_thetas, axis=1) == theta.shape[1]

        self.feats_to_remove = dict()
        for i in np.where(feats_to_remove_mask)[0].tolist():
            motif = feat_generator.motif_list[i]
            mut_pos = feat_generator.mutating_pos_list[i]
            if mut_pos not in self.feats_to_remove.keys():
                self.feats_to_remove[mut_pos] = []
            self.feats_to_remove[mut_pos].append(motif)

        self.zero_theta_mask_refit = zeroed_thetas[~feats_to_remove_mask,:]

