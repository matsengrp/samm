import numpy as np
from common import ZERO_THRES

class ModelTruncation:
    def __init__(self, theta, feat_generator):
        """
        @param theta: a fitted theta from which we determine the theta support
        @param feat_generator: the feature generator
        """
        self.feat_generator = feat_generator

        zeroed_thetas = (np.abs(theta) < ZERO_THRES)
        zeroed_or_inf_thetas = zeroed_thetas | (~np.isfinite(theta))
        self.feats_to_remove_mask = np.sum(zeroed_or_inf_thetas, axis=1) == theta.shape[1]
        self.zero_theta_mask_refit = zeroed_thetas[~self.feats_to_remove_mask,:]

        self.feats_to_remove = self.set_feats_to_remove()

    def set_feats_to_remove(self):
        """
        use the generic feature_label_list to remove features
        """
        feats_to_remove = []
        for i in np.where(self.feats_to_remove_mask)[0].tolist():
            feats_to_remove.append(self.feat_generator.feature_label_list[i])

        return feats_to_remove
