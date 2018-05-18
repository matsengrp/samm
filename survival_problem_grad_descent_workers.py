import threading
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, dok_matrix

from common import get_target_col, NUM_NUCLEOTIDES

class ThreadWorker(threading.Thread):
    """
    Stores many CalculatorWorkers and batches running them in this same Thread object
    """
    def __init__(self, workers):
        """
        @param workers: List[CalculatorWorker]
        """
        threading.Thread.__init__(self)
        self.workers = workers

    def run(self):
        for w in self.workers:
            w.run()

class CalculatorWorker:
    """
    The actual objects that calculate things in the Thread object.
    Each of these workers are assigned an index in a list that it is allowed to modify.
    That's how we get the output from these Thread objects (cause Thread objects dont return things)
    """
    def __init__(self, list_to_modify, index_to_modify):
        """
        @param list_to_modify: the list object that this worker object will get to modify
        @param index_to_modify: the index of the list that this worker object gets to modify
        """
        self.list_to_modify = list_to_modify
        self.index_to_modify

    def run(self):
        """
        Modifies the list in place
        """
        self.list_to_modify[self.index_to_modify] = self.calculate()

    def calculate(self):
        raise NotImplementedError()

class SamplePrecalcData:
    """
    Stores data for gradient calculations
    """
    def __init__(self, features_per_step_matrices, features_sign_updates, init_grad_vector, mutating_pos_feat_vals_rows, mutating_pos_feat_vals_cols, obs_seq_mutation, feat_mut_steps):
        self.features_per_step_matrices = features_per_step_matrices
        self.features_per_step_matricesT = [m.transpose() for m in features_per_step_matrices]
        self.features_sign_updates = features_sign_updates

        self.init_grad_vector = init_grad_vector
        self.mutating_pos_feat_vals_rows = mutating_pos_feat_vals_rows
        self.mutating_pos_feat_vals_cols = mutating_pos_feat_vals_cols
        self.obs_seq_mutation = obs_seq_mutation
        self.feat_mut_steps = feat_mut_steps

class PrecalcDataWorker(CalculatorWorker):
    """
    Stores the information for calculating gradient
    """
    def __init__(self, list_to_modify, index_to_modify, sample, feat_mut_steps, num_features, per_target_model):
        """
        @param exp_theta: theta is where to take the gradient of the total log likelihood, exp_theta is exp(theta)
        @param sample: ImputedSequenceMutations
        @param feat_mut_steps: list of FeatureMutationStep
        @param num_features: total number of features
        @param per_target_model: True if estimating different hazards for different target nucleotides
        """
        self.list_to_modify = list_to_modify
        self.index_to_modify = index_to_modify
        self.sample = sample
        self.feat_mut_steps = feat_mut_steps
        self.num_features = num_features
        self.per_target_model = per_target_model

    def calculate(self):
        """
        Calculate the components in the gradient at the beginning of gradient descent
        Then the gradient can be calculated using element-wise matrix multiplication
        This is much faster than a for loop!

        We pre-calculate:
            1. `features_per_step_matrix`: the number of times each feature showed up at the mutation step
            2. `base_grad`: the gradient of the sum of the exp * psi terms
            3. `mutating_pos_feat_vals_rows`: the feature row idxs for which a mutation occured
            4. `mutating_pos_feat_vals_cols`: the feature column idxs for which a mutation occured
        """
        mutating_pos_feat_vals_rows = np.array([])
        mutating_pos_feat_vals_cols = np.array([])
        num_targets = NUM_NUCLEOTIDES + 1 if self.per_target_model else 1
        base_grad = np.zeros((self.num_features, num_targets))
        # get the grad component from grad of psi * theta
        for i, feat_mut_step in enumerate(self.feat_mut_steps):
            base_grad[feat_mut_step.mutating_pos_feats, 0] += 1
            if self.per_target_model:
                col_idx = get_target_col(self.sample.obs_seq_mutation, self.sample.mutation_order[i])
                base_grad[feat_mut_step.mutating_pos_feats, col_idx] += 1
            mutating_pos_feat_vals_rows = np.append(mutating_pos_feat_vals_rows, feat_mut_step.mutating_pos_feats)
            if self.per_target_model:
                mutating_pos_feat_vals_cols = np.append(mutating_pos_feat_vals_cols, [col_idx] * len(feat_mut_step.mutating_pos_feats))

        # Get the grad component from grad of log(sum(exp(psi * theta)))
        features_per_step_matrices = []
        features_sign_updates = []
        prev_feat_mut_step = self.feat_mut_steps[0]
        for i, feat_mut_step in enumerate(self.feat_mut_steps[1:]):
            num_old = len(feat_mut_step.neighbors_feat_old)
            num_new = len(feat_mut_step.neighbors_feat_new)
            # First row of this matrix is the mutating pos. Next set are the positions with their old feature idxs. Then all the positions with their new feature idxs.
            # plus one because one position mutates and must be removed
            pos_feat_matrix = dok_matrix((
                num_old + num_new + 1,
                self.num_features,
            ), dtype=np.int8)

            # Remove feature corresponding to position that mutated already
            pos_feat_matrix[0, prev_feat_mut_step.mutating_pos_feats] = 1

            # Need to update the terms for positions near the previous mutation
            # Remove old feature values
            old_feat_idxs = feat_mut_step.neighbors_feat_old.values()
            for f_idx, f_list in enumerate(old_feat_idxs):
                pos_feat_matrix[f_idx + 1, f_list] = 1

            # Add new feature values
            new_feat_idxs = feat_mut_step.neighbors_feat_new.values()
            for f_idx, f_list in enumerate(new_feat_idxs):
                pos_feat_matrix[f_idx + 1 + num_old, f_list] = 1

            features_per_step_matrices.append(csr_matrix(pos_feat_matrix, dtype=np.int8))
            features_sign_updates.append(
                np.reshape(np.concatenate([-1 * np.ones(num_old + 1), np.ones(num_new)]), (num_old + num_new + 1, 1))
            )

            prev_feat_mut_step = feat_mut_step

        return SamplePrecalcData(
            features_per_step_matrices,
            features_sign_updates,
            base_grad,
            np.array(mutating_pos_feat_vals_rows, dtype=np.int16),
            np.array(mutating_pos_feat_vals_cols, dtype=np.int16),
            self.sample.obs_seq_mutation,
            self.feat_mut_steps,
        )

class GradientWorker(CalculatorWorker):
    """
    Stores the information for calculating gradient
    """
    def __init__(self, list_to_modify, index_to_modify, sample_data, per_target_model, theta):
        """
        @param sample_data: class SamplePrecalcData
        """
        self.list_to_modify = list_to_modify
        self.index_to_modify = index_to_modify
        self.sample_data = sample_data
        self.per_target_model = per_target_model
        self.theta = theta

    def calculate(self):
        """
        Calculate the gradient of the log likelihood of this sample
        All the gradients for each step are the gradient of psi * theta - log(sum(exp(theta * psi)))
        Calculate the gradient from each step one at a time

        @param theta: the theta to evaluate the gradient at
        @param sample_data: SamplePrecalcData
        """
        merged_thetas = self.theta[:,0, None]
        if self.per_target_model:
            merged_thetas = merged_thetas + self.theta[:,1:]
        pos_exp_theta = np.exp(self.sample_data.obs_seq_mutation.feat_matrix_start.dot(merged_thetas))
        prev_denom = pos_exp_theta.sum()

        prev_risk_group_grad = self.sample_data.obs_seq_mutation.feat_matrix_start.transpose().dot(pos_exp_theta)

        risk_group_grad_tot = prev_risk_group_grad/prev_denom
        for pos_feat_matrix, pos_feat_matrixT, features_sign_update in zip(self.sample_data.features_per_step_matrices, self.sample_data.features_per_step_matricesT, self.sample_data.features_sign_updates):
            exp_thetas = np.exp(pos_feat_matrix.dot(merged_thetas))
            signed_exp_thetas = np.multiply(exp_thetas, features_sign_update)

            prev_risk_group_grad += pos_feat_matrixT.dot(signed_exp_thetas)

            prev_denom += signed_exp_thetas.sum()
            prev_denom_inv = 1.0/prev_denom
            risk_group_grad_tot += prev_risk_group_grad * prev_denom_inv
        if self.per_target_model:
            risk_group_grad_tot = np.hstack([np.sum(risk_group_grad_tot, axis=1, keepdims=True), risk_group_grad_tot])
        return np.array(
                self.sample_data.init_grad_vector - risk_group_grad_tot,
                dtype=float)

    def __str__(self):
        return "GradientWorker %s" % self.sample_data.obs_seq_mutation

class ObjectiveValueWorker(CalculatorWorker):
    """
    Stores the information for calculating objective function value
    """
    def __init__(self, list_to_modify, index_to_modify, sample_data, per_target_model, theta):
        """
        @param sample: SamplePrecalcData
        """
        self.list_to_modify = list_to_modify
        self.index_to_modify = index_to_modify
        self.sample_data = sample_data
        self.per_target_model = per_target_model
        self.theta = theta

    def calculate(self):
        """
        Calculate the log likelihood of this sample
        """
        merged_thetas = self.theta[:,0, None]
        if self.per_target_model:
            merged_thetas = merged_thetas + self.theta[:,1:]
        prev_denom = (np.exp(self.sample_data.obs_seq_mutation.feat_matrix_start.dot(merged_thetas))).sum()
        denominators = [prev_denom]
        for pos_feat_matrix, features_sign_update in zip(self.sample_data.features_per_step_matrices, self.sample_data.features_sign_updates):
            exp_thetas = np.exp(pos_feat_matrix.dot(merged_thetas))
            signed_exp_thetas = np.multiply(exp_thetas, features_sign_update)
            new_denom = prev_denom + signed_exp_thetas.sum()
            denominators.append(new_denom)
            prev_denom = new_denom

        numerators = self.theta[self.sample_data.mutating_pos_feat_vals_rows, 0]
        if self.per_target_model:
            numerators = numerators + self.theta[self.sample_data.mutating_pos_feat_vals_rows, self.sample_data.mutating_pos_feat_vals_cols]

        log_lik = numerators.sum() - np.log(denominators).sum()
        return log_lik

    def __str__(self):
        return "ObjectiveValueWorker %s" % self.sample_data.obs_seq_mutation

class HessianWorker(CalculatorWorker):
    """
    Stores the information for calculating gradient
    """
    def __init__(self, list_to_modify, index_to_modify, sample_datas, per_target_model, theta):
        """
        @param sample_data: class SamplePrecalcData
        """
        self.list_to_modify = list_to_modify
        self.index_to_modify = index_to_modify
        self.sample_datas = sample_datas
        self.per_target_model = per_target_model
        self.theta = theta

    def calculate(self):
        """
        @return the sum of the second derivatives of the log likelihood for the complete data
        """
        tot_hessian = 0
        for s in self.sample_datas:
            h = self._get_hessian_per_sample(s)
            tot_hessian += h
        return tot_hessian

    def _get_hessian_per_sample(self, sample_data):
        """
        Calculates the second derivative of the log likelihood for the complete data
        """
        merged_thetas = self.theta[:,0, None]
        if self.per_target_model:
            merged_thetas = merged_thetas + self.theta[:,1:]

        # Create base dd matrix.
        dd_matrices = [np.zeros((self.theta.shape[0], self.theta.shape[0])) for i in range(self.theta.shape[1])]
        for pos in range(sample_data.obs_seq_mutation.seq_len):
            exp_theta_psi = np.exp(np.array(sample_data.obs_seq_mutation.feat_matrix_start[pos,:] * merged_thetas).flatten())
            features = sample_data.obs_seq_mutation.feat_matrix_start[pos,:].nonzero()[1]
            for f1 in features:
                for f2 in features:
                    # The first column in a per-target model appears in all other columns. Hence we need a sum of all the exp_thetas
                    dd_matrices[0][f1, f2] += exp_theta_psi.sum()
                    for j in range(1, self.theta.shape[1]):
                        # The rest of the theta values for the per-target model only appear once in the exp_theta vector
                        dd_matrices[j][f1, f2] += exp_theta_psi[j - 1]

        pos_exp_theta = np.exp(sample_data.obs_seq_mutation.feat_matrix_start.dot(merged_thetas))
        prev_denom = pos_exp_theta.sum()

        prev_risk_group_grad = sample_data.obs_seq_mutation.feat_matrix_start.transpose().dot(pos_exp_theta)
        if self.per_target_model:
            # Deal with the fact that the first column is special in a per-target model
            aug_prev_risk_group_grad = np.hstack([np.sum(prev_risk_group_grad, axis=1, keepdims=True), prev_risk_group_grad])
            aug_prev_risk_group_grad = aug_prev_risk_group_grad.reshape((aug_prev_risk_group_grad.size, 1), order="F")
        else:
            aug_prev_risk_group_grad = prev_risk_group_grad.reshape((prev_risk_group_grad.size, 1), order="F")

        block_diag_dd = sp.linalg.block_diag(*dd_matrices)
        for i in range(self.theta.shape[1] - 1):
            # Recall that the first column in a per-target model appears in all the exp_theta expressions. So we need to add in these values
            # to the off-diagonal blocks of the second-derivative matrix
            block_diag_dd[(i + 1) * self.theta.shape[0]:(i + 2) * self.theta.shape[0], 0:self.theta.shape[0]] = dd_matrices[i + 1]
            block_diag_dd[0:self.theta.shape[0], (i + 1) * self.theta.shape[0]:(i + 2) * self.theta.shape[0]] = dd_matrices[i + 1]

        risk_group_hessian = aug_prev_risk_group_grad * aug_prev_risk_group_grad.T * np.power(prev_denom, -2) - np.power(prev_denom, -1) * block_diag_dd
        for pos_feat_matrix, pos_feat_matrixT, features_sign_update in zip(sample_data.features_per_step_matrices, sample_data.features_per_step_matricesT, sample_data.features_sign_updates):
            exp_thetas = np.exp(pos_feat_matrix.dot(merged_thetas))
            signed_exp_thetas = np.multiply(exp_thetas, features_sign_update)

            prev_risk_group_grad += pos_feat_matrixT.dot(signed_exp_thetas)

            prev_denom += signed_exp_thetas.sum()

            # Now update the dd_matrix after the previous mutation step
            for i in range(pos_feat_matrix.shape[0]):
                feature_vals = pos_feat_matrix[i,:].nonzero()[1]
                for f1 in feature_vals:
                    for f2 in feature_vals:
                        dd_matrices[0][f1, f2] += signed_exp_thetas[i,:].sum()
                        for j in range(1, self.theta.shape[1]):
                            dd_matrices[j][f1, f2] += signed_exp_thetas[i,j - 1]

            if self.per_target_model:
                aug_prev_risk_group_grad = np.hstack([np.sum(prev_risk_group_grad, axis=1, keepdims=True), prev_risk_group_grad])
                aug_prev_risk_group_grad = aug_prev_risk_group_grad.reshape((aug_prev_risk_group_grad.size, 1), order="F")
            else:
                aug_prev_risk_group_grad = prev_risk_group_grad.reshape((prev_risk_group_grad.size, 1), order="F")

            block_diag_dd = sp.linalg.block_diag(*dd_matrices)
            for i in range(self.theta.shape[1] - 1):
                block_diag_dd[(i + 1) * self.theta.shape[0]:(i + 2) * self.theta.shape[0], 0:self.theta.shape[0]] = dd_matrices[i + 1]
                block_diag_dd[0:self.theta.shape[0], (i + 1) * self.theta.shape[0]:(i + 2) * self.theta.shape[0]] = dd_matrices[i + 1]

            risk_group_hessian += aug_prev_risk_group_grad * aug_prev_risk_group_grad.T * np.power(prev_denom, -2) - np.power(prev_denom, -1) * block_diag_dd
        return risk_group_hessian

class ScoreScoreWorker(CalculatorWorker):
    """
    Calculate the product of scores
    """
    def __init__(self, list_to_modify, index_to_modify, grad_log_liks):
        """
        @param grad_log_liks: the grad_log_liks to calculate the product of scores
        """
        self.list_to_modify = list_to_modify
        self.index_to_modify = index_to_modify
        self.grad_log_liks = grad_log_liks

    def calculate(self):
        """
        @return the sum of the product of scores
        """
        ss = 0
        for g in self.grad_log_liks:
            g = g.reshape((g.size, 1), order="F")
            ss += g * g.T
        return ss

class ExpectedScoreScoreWorker(CalculatorWorker):
    """
    Calculate the product of expected scores between itself (indexed by labels)
    """
    def __init__(self, list_to_modify, index_to_modify, label_list, expected_scores):
        """
        @param label_list: a list of labels to process
        """
        self.list_to_modify = list_to_modify
        self.index_to_modify = index_to_modify
        self.label_list = label_list
        self.expected_scores = expected_scores

    def calculate(self):
        """
        @param expected_scores: the expected scores
        @return the sum of the product of expected scores for the given label_list
        """
        ss = 0
        for label in self.label_list:
            ss += self.expected_scores[label] * self.expected_scores[label].T
        return ss
