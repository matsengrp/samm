import numpy as np
import scipy.misc
import logging as log

from models import ImputedSequenceMutations
from common import *

from sampler_collection import Sampler
from profile_support import profile

class GibbsSamplerResult:
    def __init__(self, samples, trace, residuals=None):
        """
        class returned by Sampler after a run

        @param samples: list of ImputedSequenceMutations
        @param trace: list of things to plot for trace diagnostics
        @param residuals: list of lists of Martingale residuals
        """
        self.samples = samples
        self.trace = trace
        self.residuals = residuals

class GibbsStepInfo:
    """
    Store the state of each gibbs sample and intermediate computations
    """
    def __init__(self, order, log_numerators, denominators, sampled_risks=None):
        """
        @param order: a list with positions in the order the mutations happened
        @param log_numerators: the log of the exp(theta * psi) term in the numerator of the likelihood
                            at each mutation step
        @param denominators: the sum of the exp(theta * psi) terms in the denominator of the likelihood
                            at each mutation step
        @param sampled_risks: list of risk values for this Gibbs step (summands in denominator)
        """
        self.order = order
        self.log_numerators = log_numerators
        self.denominators = denominators
        self.sampled_risks = sampled_risks

class MutationOrderGibbsSampler(Sampler):
    """
    A class that will do the heavy lifting of Gibbs sampling.
    Returns orders and log probability vector (for tracing)
    """
    def run(self, init_order, burn_in, num_samples, sampling_rate, conditional_partial_order=[]):
        """
        @param init_order: a mutation order to initialize the sampler (list of integers)
        @param burn_in: number of iterations for burn in
        @param num_samples: number of samples needed
        @param sampling_rate: non-neg int, get 1 sample per K gibbs sweeps. if zero, then get all samples in a sweep too.
        @param conditional_partial_order: list of position where the partial ordering is fixed. so if non-empty,
                                        we are drawing samples conditioned on this partial ordering
        """
        # Determine which positions we are performing gibbs on
        positions_to_sample = list(set(self.mutated_positions) - set(conditional_partial_order))

        traces = []
        residuals = []
        if self.num_mutations < 2 and not self.get_residuals:
            # If there are zero or one mutations then the same initial order will be returned for
            # every sample. We can still get residuals in this case, though
            samples = [init_order] * (num_samples)
        else:
            samples = []
            log.info("Gibbs: num mutations %d, seq len %d" % (self.num_mutations, self.obs_seq_mutation.seq_len))

            curr_gibbs_step_info = None
            curr_order = init_order
            num_iters = num_samples * sampling_rate if sampling_rate > 0 else num_samples
            for i in range(burn_in + num_iters):
                gibbs_orders, curr_gibbs_step_info, trace, curr_residuals = self._do_gibbs_sweep(positions_to_sample, curr_order, curr_gibbs_step_info)
                curr_order = gibbs_orders[-1]
                if i >= burn_in:
                    if sampling_rate == 0:
                        samples += gibbs_orders
                        residuals += curr_residuals
                    elif sampling_rate > 0 and i % sampling_rate == 0:
                        samples += [gibbs_orders[-1]]
                        residuals += curr_residuals
                traces += trace
        if self.get_residuals:
            residuals = np.nanmean(residuals, axis=0)

        return GibbsSamplerResult(
            [ImputedSequenceMutations(self.obs_seq_mutation, order) for order in samples],
            traces,
            residuals,
        )

    def _do_gibbs_sweep(self, positions_to_sample, curr_order, gibbs_step_info=None):
        """
        One gibbs sweep is a gibbs sampling step for all the positions, conditional on conditional_partial_order
        Returns all the sampled orders from the gibbs sweep and trace

        @param positions_to_sample: the positions we are performing gibbs on
        @param curr_order: current order in sampling step
        @param gibbs_step_info: GibbsStepInfo with the information from the most recent step.
                                used to minimize recomputation

        @return gibbs_step_orders: list of orders from each Gibbs step
        @return gibbs_step_info: an object of class GibbStepInfo
        @return trace: list of log likelihoods for trace plots
        @return residuals: list of lists of Martingale residuals
        """
        gibbs_step_orders = []
        trace = []
        residuals = []
        # Perform gibbs sweep in a random order
        for position in np.random.permutation(positions_to_sample):
            # Take out the position we are going to sample order for and get the partial ordering under consideration
            pos_order_idx = curr_order.index(position)
            partial_order = curr_order[0:pos_order_idx] + curr_order[pos_order_idx + 1:]
            gibbs_step_info, log_lik, _ = self._do_gibbs_step(partial_order, position, gibbs_step_info, pos_order_idx)
            curr_order = gibbs_step_info.order
            # Output probabilities for trace
            trace.append(log_lik)
            if gibbs_step_info.sampled_risks is not None:
                residuals.append(
                    self._calculate_unoffset_residuals(np.nansum(gibbs_step_info.sampled_risks, axis=0).ravel())
                )
            gibbs_step_orders.append(list(curr_order))
        return gibbs_step_orders, gibbs_step_info, trace, residuals

    def _do_gibbs_step(self, partial_order, position, gibbs_step_info=None, pos_order_idx=None):
        """
        Performs a single gibbs step
        @param partial_order: the partial order under consideration
        @param position: the position we are trying to determine the full ordering for
        @param gibbs_step_info: if None, compute all features from scratch. otherwise use info from a previous gibbs step
                                to minimize calculations in this step
        @param pos_order_idx: the position's mutation step from the order in `gibbs_step_info`
        """
        # A holder for all the log probs - we need to normalize these later to get our sampled order
        all_log_probs = []
        # Compute probabilities for the orderings under consideration
        # First consider the full ordering with position under consideration mutating last
        order_last = partial_order + [position]
        if gibbs_step_info is None:
            feat_mutation_steps, log_numerators, denominators, all_risks = self._compute_log_probs_from_scratch(
                order_last,
            )
        else:
            feat_mutation_steps, log_numerators, denominators, all_risks = self._compute_log_probs_with_reference(
                order_last,
                gibbs_step_info,
                update_step_start=pos_order_idx,
            )
        full_ordering_log_prob = np.sum(log_numerators) - (np.log(denominators)).sum()

        # Add the log probability of the position mutating last
        all_log_probs.append(full_ordering_log_prob)

        # These lists store the history of the log numerators, the denominators and the risk values
        # The numerators have a possibility of changing three times during order shuffling
        # The denominator have a possibility of changing two times during ordering shuffling
        # We can reconstruct the numerators and the denominators for a particular sampled mutation order
        # from these lists
        log_numerator_hist = [[log_n] for log_n in log_numerators]
        log_numerator_hist[-1].append(log_numerators[-1])
        denominator_hist = [[d] for d in denominators]
        if self.get_residuals:
            risk_hist = [[r] for r in all_risks]
        else:
            risk_hist = None

        # Now unmutate the string by one mutation step so that we can figure out the features at the positions
        flanked_seq = unmutate_string(
            self.obs_seq_mutation.end_seq_with_flanks,
            unmutate_pos=self.obs_seq_mutation.left_flank_len + position,
            orig_nuc=self.obs_seq_mutation.start_seq[position]
        )
        already_mutated_pos_set = set(partial_order)
        # iterate through the remaining possible full mutation orders consistent with this partial order
        for idx, i in enumerate(reversed(range(self.num_mutations - 1))):
            possible_full_order = partial_order[:i] + [position] + partial_order[i:]

            shuffled_position = partial_order[i]
            already_mutated_pos_set.remove(shuffled_position)
            # Now unmutate the string so that we can figure out the features at the positions
            # right before the i-th mutation step occured
            flanked_seq = unmutate_string(
                flanked_seq,
                unmutate_pos=self.obs_seq_mutation.left_flank_len + shuffled_position,
                orig_nuc=self.obs_seq_mutation.start_seq[shuffled_position]
            )
            # Now get the features - we only need the feature of the mutating position at the ith step
            # And the feature updates at the time of the `i+1`-th step
            seq_mut_order = ImputedSequenceMutations(
                self.obs_seq_mutation,
                possible_full_order
            )
            first_mutation_feats, second_feat_mut_step = self.feature_generator.get_shuffled_mutation_steps_delta(
                seq_mut_order,
                update_step=i,
                flanked_seq=flanked_seq,
                already_mutated_pos=already_mutated_pos_set,
            )

            # correct the full ordering probability by taking away the old terms
            full_ordering_log_prob += -log_numerators[i] - log_numerators[i + 1] + np.log(denominators[i + 1])

            log_numerators[i] = self.theta[first_mutation_feats, 0].sum()
            log_numerators[i + 1] = self.theta[second_feat_mut_step.mutating_pos_feats, 0].sum()
            if self.per_target_model:
                col_idx_earlier = get_target_col(self.obs_seq_mutation, position)
                log_numerators[i] += self.theta[first_mutation_feats, col_idx_earlier].sum()

                col_idx_later = get_target_col(self.obs_seq_mutation, shuffled_position)
                log_numerators[i + 1] += self.theta[second_feat_mut_step.mutating_pos_feats, col_idx_later].sum()
            denominators[i + 1] = self._get_denom_update(denominators[i], first_mutation_feats, second_feat_mut_step)
            if self.get_residuals:
                first_mutation_pos = [seq_mut_order.mutation_order[i]]
                all_risks[i + 1] = self._get_risk_update(
                    all_risks[i],
                    first_mutation_pos,
                    second_feat_mut_step,
                    denominators[i],
                    denominators[i+1],
                )

            # correct the full ordering probability by adding back the new terms
            full_ordering_log_prob += log_numerators[i] + log_numerators[i + 1] - np.log(denominators[i + 1])

            all_log_probs.append(full_ordering_log_prob)

            # Track the numerator, denominator and risk history
            log_numerator_hist[i].append(log_numerators[i])
            log_numerator_hist[i + 1].append(log_numerators[i + 1])
            denominator_hist[i+1].append(denominators[i + 1])
            if self.get_residuals:
                risk_hist[i+1].append(all_risks[i + 1])

        # Now sample and reconstruct our decision from the numerator/denominator/risk histories
        gibbs_step_info, log_lik = self._sample_order(
            all_log_probs,
            partial_order,
            position,
            denominator_hist,
            log_numerator_hist,
            risk_hist,
        )

        return gibbs_step_info, log_lik, all_log_probs

    def _sample_order(self, all_log_probs, partial_order, position, denominator_hist, log_numerator_hist, risk_hist=None):
        """
        Sample mutation order from all the possible full mutation orders
        Reconstruct the intermediate calculations for this sampled order for faster future computations

        @param partial_order: the partial ordering we were originally considering
        @param position: the position we need to determine the order for
        @param denominator_hist: the history of all the denominators for reconstructing the intermediate
                                calculations for the sampled mutation order
        @param log_numerator_hist: the history of all the (log) numerators for reconstructing the intermediate
                                calculations for the sampled mutation order
        @param risk_hist: the history of all the risks for reconstructing the fitted risk values; defaults to None if we are not calculating residuals

        @return tuple of GibbsStepInfo and log likelihood of the sampled mutation order
        """

        # for tiny probabilities shift by subtracting negative
        all_probs = np.exp(all_log_probs - min(all_log_probs))
        sampled_idx = sample_multinomial(all_probs)

        # Now reconstruct our decision
        idx = self.num_mutations - sampled_idx - 1
        sampled_order = partial_order[:idx] + [position] + partial_order[idx:]
        sampled_denominators = (
            [denominator_hist[0][0]]
            + [denominator_hist[i][0] for i in range(1, self.num_mutations - sampled_idx)]
            + [denominator_hist[i][1] for i in range(self.num_mutations - sampled_idx, self.num_mutations)]
        )
        sampled_log_numerators = (
            [log_numerator_hist[i][0] for i in range(self.num_mutations - sampled_idx - 1)]
            + [log_numerator_hist[self.num_mutations - sampled_idx - 1][1]]
            + [log_numerator_hist[i][2] for i in range(self.num_mutations - sampled_idx, self.num_mutations)]
        )
        if risk_hist is not None:
            sampled_risks = (
                [risk_hist[0][0]]
                + [risk_hist[i][0] for i in range(1, self.num_mutations - sampled_idx)]
                + [risk_hist[i][1] for i in range(self.num_mutations - sampled_idx, self.num_mutations)]
            )
        else:
            sampled_risks = None

        gibbs_step_sample = GibbsStepInfo(
            sampled_order,
            sampled_log_numerators,
            sampled_denominators,
            sampled_risks,
        )

        return gibbs_step_sample, all_log_probs[sampled_idx]

    def get_log_probs(self, order):
        """
        Compute the log likelihood for this full mutation ordering
        (Not conditional on the ending sequence)
        """
        _, log_numerators, denominators, _ = self._compute_log_probs_from_scratch(order)
        return np.sum(log_numerators) - (np.log(denominators)).sum()

    def _compute_log_probs_from_scratch(self, curr_order):
        """
        Compute the log likelihood for this full mutation ordering
        Calculates the log likelihood of each mutation step

        @params curr_order: the full mutation ordering under consideration
        @return tuple of
            1. feature mutation steps
            2. the log numerators in the log likelihood of each mutation step
            3. the denominator in the log likelihood of each mutation step
            4. all_risks, i.e., summands in the denominators; None if get_residuals is False
        """
        feat_mutation_steps = self.feature_generator.create_for_mutation_steps(
            ImputedSequenceMutations(
                self.obs_seq_mutation,
                curr_order,
            )
        )

        # Get the components -- numerators and the denominators
        log_numerators = []
        for i, mut_step in enumerate(feat_mutation_steps):
            theta_sum = self.theta[mut_step.mutating_pos_feats, 0].sum()
            if self.per_target_model:
                col_idx = get_target_col(self.obs_seq_mutation, curr_order[i])
                theta_sum += self.theta[mut_step.mutating_pos_feats, col_idx].sum()

            log_numerators.append(theta_sum)

        merged_thetas = self.theta[:,0,None]
        if self.per_target_model:
            merged_thetas = merged_thetas + self.theta[:,1:]

        risk_vec = (np.exp(self.obs_seq_mutation.feat_matrix_start * merged_thetas))
        risk_vec_sum = risk_vec.sum()
        all_risk_vecs = [risk_vec / risk_vec_sum]
        denominators = [risk_vec_sum]
        prev_feat_mut_step = feat_mutation_steps[0]
        for i, feat_mut_step in enumerate(feat_mutation_steps[1:]):
            new_denom = self._get_denom_update(denominators[i], prev_feat_mut_step.mutating_pos_feats, feat_mut_step)
            if self.get_residuals:
                new_risk_vec = self._get_risk_update(
                    all_risk_vecs[i],
                    prev_feat_mut_step.mutating_pos,
                    feat_mut_step,
                    denominators[i],
                    new_denom,
                )
                all_risk_vecs.append(new_risk_vec)
            prev_feat_mut_step = feat_mut_step
            denominators.append(new_denom)
        return feat_mutation_steps, log_numerators, denominators, all_risk_vecs

    def _compute_log_probs_with_reference(self, curr_order, gibbs_step_base, update_step_start=0):
        """
        Compute probabilities that takes advantage of precomputed information from `gibbs_step_base`

        @params curr_order: the full mutation ordering under consideration
        @param gibbs_step_base: GibbsStepInfo, the reference mutation order and its intermediate computations
        @param update_step_start: which mutations step to start recalculating the log liklihood terms from

        @return similar to compute_log_probs_from_scratch
        """
        # Update if we have previous computations
        feat_mutation_steps = self.feature_generator.create_remaining_mutation_steps(
            ImputedSequenceMutations(
                self.obs_seq_mutation,
                curr_order,
            ),
            update_step_start=update_step_start,
        )
        # Use gibbs_step_base to update denoms and numerators
        log_numerators = gibbs_step_base.log_numerators[:update_step_start]
        for i, mut_step in enumerate(feat_mutation_steps):
            theta_sum = self.theta[mut_step.mutating_pos_feats, 0].sum()
            if self.per_target_model:
                col_idx = get_target_col(self.obs_seq_mutation, curr_order[update_step_start + i])
                theta_sum += self.theta[mut_step.mutating_pos_feats, col_idx].sum()
            log_numerators.append(theta_sum)

        all_risk_vecs = gibbs_step_base.sampled_risks[:update_step_start + 1] if self.get_residuals else None
        denominators = gibbs_step_base.denominators[:update_step_start + 1]
        prev_feat_mut_step = feat_mutation_steps[0]
        for i in range(update_step_start, self.num_mutations - 1):
            feat_mut_step = feat_mutation_steps[i - update_step_start + 1]
            new_denom = self._get_denom_update(denominators[i], prev_feat_mut_step.mutating_pos_feats, feat_mut_step)
            if self.get_residuals:
                new_risk_vec = self._get_risk_update(
                    all_risk_vecs[i],
                    prev_feat_mut_step.mutating_pos,
                    feat_mut_step,
                    denominators[i],
                    new_denom,
                )
                all_risk_vecs.append(new_risk_vec)
            prev_feat_mut_step = feat_mut_step
            denominators.append(new_denom)

        return feat_mutation_steps, log_numerators, denominators, all_risk_vecs

    def _get_denom_update(self, old_denominator, prev_feat_idxs, feat_mut_step):
        """
        Calculate the denominator of the next mutation step quickly by reusing past computations
        and incorporating the deltas appropriately

        @param old_denominator: the denominator from the previous mutation step
        @param prev_feat_idxs: previously mutated feature indices that won't contribute to the updated risk group
        @param feat_mut_step: the features that differed for this next mutation step
        """
        if self.feature_generator.num_feat_gens > 1:
            if not self.per_target_model:
                old_feat_theta_sums = [self.theta[feat_idxs].sum() for feat_idxs in feat_mut_step.neighbors_feat_old.values()]
                new_feat_theta_sums = [self.theta[feat_idxs].sum() for feat_idxs in feat_mut_step.neighbors_feat_new.values()]
                new_denom = old_denominator - np.exp(self.theta[prev_feat_idxs].sum()).sum() - np.exp(old_feat_theta_sums).sum() + np.exp(new_feat_theta_sums).sum()
            else:
                old_feat_theta_sums = [self.theta[feat_idxs,0].sum() + self.theta[feat_idxs, 1:].sum(axis=0) for feat_idxs in feat_mut_step.neighbors_feat_old.values()]
                new_feat_theta_sums = [self.theta[feat_idxs,0].sum() + self.theta[feat_idxs, 1:].sum(axis=0) for feat_idxs in feat_mut_step.neighbors_feat_new.values()]
                prev_theta_sum = self.theta[prev_feat_idxs,0].sum() + self.theta[prev_feat_idxs,1:].sum(axis=0)
                new_denom = old_denominator - np.exp(prev_theta_sum).sum() - np.exp(old_feat_theta_sums).sum() + np.exp(new_feat_theta_sums).sum()
        else:
            old_feat_exp_theta_sums = [self.exp_theta_sum[feat_idx] if feat_idx.size else self.exp_theta_num_cols for feat_idx in feat_mut_step.neighbors_feat_old.values()]
            new_feat_exp_theta_sums = [self.exp_theta_sum[feat_idx] if feat_idx.size else self.exp_theta_num_cols for feat_idx in feat_mut_step.neighbors_feat_new.values()]
            prev_exp_theta_sum = self.exp_theta_sum[prev_feat_idxs] if len(prev_feat_idxs) else self.exp_theta_num_cols
            new_denom = old_denominator - prev_exp_theta_sum - np.sum(old_feat_exp_theta_sums) + np.sum(new_feat_exp_theta_sums)
        return float(new_denom)

    def _get_risk_update(self, old_risk_vec, prev_mut_pos, feat_mut_step, old_denom, new_denom):
        """
        Similar to _get_denom_update but calculate the updated risk vector to use as residuals

        @param old_risk_vec: the vector of summands from denominator from the previous mutation step
        @param prev_mut_pos: previously mutated position that won't contribute to the updated risk group
        @param feat_mut_step: the features that differed for this next mutation step
        @param old_denom: previous mutation step denominator
        @param new_denom: new denominator
        """
        # reweight old risk vector with new denominator
        base_risk_vec = old_risk_vec * old_denom / new_denom

        # update risk values for nearby positions
        for feat_pos, feat_idxs in feat_mut_step.neighbors_feat_new.iteritems():
            if self.feature_generator.num_feat_gens > 1:
                if not self.per_target_model:
                    base_risk_vec[feat_pos] = np.exp(self.theta[feat_idxs].sum()) / new_denom
                else:
                    base_risk_vec[feat_pos] = np.exp(self.theta[feat_idxs,0].sum() + self.theta[feat_idxs, 1:].sum(axis=0)) / new_denom
            elif feat_idxs.size:
                base_risk_vec[feat_pos] = self.exp_theta_sum[feat_idxs]/ new_denom
        if len(prev_mut_pos):
            # prev_mut_pos are now zero risk having mutated
            base_risk_vec[prev_mut_pos] = 0.

        # make sure there are no NaNs because we'll use that to pad later
        assert(not any([np.isnan(risk) for risk in base_risk_vec]))

        return base_risk_vec

    def _calculate_unoffset_residuals(self, acc_risks):
        """
        Get position-wise residuals

        These are martingale residuals padded with nans where we ignored positions due to
        mutations in the flanks.

        @param acc_risks: numpy array of accumulated risk values

        @return: indicator of whether a position mutated minus its accumulated risk; NaNs added to flanks and ambiguous positions
        """

        def _pad_vec_with_nan(vec):
            """
            Pad vector with NaN values both for the flanking positions and for the interior "n" values that were collapsed during processing
            """
            padded_vec = np.concatenate((
                [np.nan] * self.obs_seq_mutation.left_position_offset,
                vec,
                [np.nan] * self.obs_seq_mutation.right_position_offset
            ))
            # insert NaNs where we had collapsed in processing the data before
            for collapse_tuple in sorted(self.obs_seq_mutation.collapse_list, key=lambda val: val[1]):
                start_idx = collapse_tuple[0] + collapse_tuple[1]
                if start_idx <= len(padded_vec):
                    # if it's greater then we'll pad the end regardless
                    to_insert = [np.nan] * (collapse_tuple[2] - collapse_tuple[0] - collapse_tuple[1])
                    padded_vec = np.insert(padded_vec, start_idx, to_insert)
            return padded_vec

        # pad the indicator vector to take into account flanks and skipped bases
        padded_risks = _pad_vec_with_nan(acc_risks)
        padded_mutated_indicator = _pad_vec_with_nan(self.obs_seq_mutation.mutated_indicator)

        return [mutated - risk for mutated, risk in zip(padded_mutated_indicator, padded_risks)]
