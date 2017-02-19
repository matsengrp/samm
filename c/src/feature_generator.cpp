#include <assert.h>
#include <algorithm>

#include "feature_generator.hpp"
#include "common.hpp"

SubmotifFeatureGenerator::SubmotifFeatureGenerator(int m) {
  assert(m % 2 == 1);
  motif_len = m;
  motif_len_half = motif_len/2;
  feature_vec_len = 1 << (2 * motif_len);
}

shared_ptr<ObservedSample> SubmotifFeatureGenerator::CreateObservedSample(
  VectorNucleotide start_nuc,
  VectorNucleotide end_nuc,
  VectorNucleotide left_flank,
  VectorNucleotide right_flank
) {
  VectorFeature feature_vec;
  for (int i = 0; i < start_nuc.val.size(); i++) {
    feature_vec.val.push_back(GetFeatureIdxForPos(i, start_nuc, left_flank, right_flank));
  }
  shared_ptr<ObservedSample> obs_sample(new ObservedSample(
    start_nuc,
    end_nuc,
    left_flank,
    right_flank,
    feature_vec
  ));
  return obs_sample;
}

shared_ptr<OrderedMutationSteps> SubmotifFeatureGenerator::CreateForMutationSteps(
  const shared_ptr<ObservedSample> &obs_sample,
  VectorOrder mut_order,
  const pair<bool, vector<double> > &theta
) {
  shared_ptr<OrderedMutationSteps> ordered_mut_steps(new OrderedMutationSteps(mut_order));
  for (int i = 0; i < ordered_mut_steps->num_steps; i++) {
    shared_ptr<MutationStep> mutation_step;

    if (i == 0) {
      // The first step is special - we have precalculated a bunch of values already.
      VectorNucleotide intermediate_nucs = obs_sample->start_nucs;
      VectorFeature intermediate_feats = obs_sample->start_features;

      pair<bool, VectorThetaSums> theta_sum_option(theta.first, VectorThetaSums());
      if (theta.first) {
        // calculate theta sum from scratch
        for (int p = 0; p < obs_sample->num_pos; p++) {
          int feat_idx = intermediate_feats.val[p];
          if (feat_idx == MUTATED) {
            theta_sum_option.second.val.push_back(MUTATED);
          } else {
            theta_sum_option.second.val.push_back(theta.second[feat_idx]);
          }
        }
      }

      mutation_step = make_shared<MutationStep>(
        intermediate_nucs,
        intermediate_feats,
        theta_sum_option
      );
    } else {
      mutation_step = UpdateMutationStep(
        i,
        mut_order.val[i - 1],
        ordered_mut_steps,
        obs_sample,
        theta
      );
    }

    ordered_mut_steps->set(i, mutation_step);
  }

  return ordered_mut_steps;
}

shared_ptr<OrderedMutationSteps> SubmotifFeatureGenerator::UpdateForMutationSteps(
  const shared_ptr<ObservedSample> &obs_sample,
  VectorOrder mut_order,
  const vector<int> update_steps,
  const shared_ptr<OrderedMutationSteps> &base_mut_order,
  const pair<bool, vector<double> > &theta
) {
  shared_ptr<OrderedMutationSteps> ordered_mut_steps = make_shared<OrderedMutationSteps>(
    base_mut_order->num_steps,
    base_mut_order->order_vec,
    base_mut_order->mut_steps
  );
  ordered_mut_steps->order_vec = mut_order;

  for (int i: update_steps) {
    shared_ptr<MutationStep> mutation_step = UpdateMutationStep(
      i,
      mut_order.val[i - 1],
      ordered_mut_steps,
      obs_sample,
      theta
    );
    ordered_mut_steps->set(i, mutation_step);
  }

  return ordered_mut_steps;
}

int SubmotifFeatureGenerator::GetFeatureIdxForPos(
  int position,
  const VectorNucleotide &nuc_vec,
  const VectorNucleotide &left_flank,
  const VectorNucleotide &right_flank
) {
  // Initial testing shows that this is faster than stoi and map lookup
  int idx = 0;
  int base = 1;

  for (int i = 0; i < motif_len; i++) {
    int seq_pos = position + motif_len_half - i;

    int nuc;
    if (seq_pos < 0) {
      // Position is before the beginning! Use the left flank!
      nuc = left_flank.val[motif_len_half + seq_pos];
    } else if (seq_pos >= nuc_vec.val.size()) {
      // Position is after the end! Use the right flank!
      nuc = right_flank.val[seq_pos - nuc_vec.val.size()];
    } else {
      nuc = nuc_vec.val[seq_pos];
    }

    idx += nuc * base;
    base = base << 2;
  }
  return idx;
}

/*
* PRIVATE FUNCTIONS
*/
shared_ptr<MutationStep> SubmotifFeatureGenerator::UpdateMutationStep(
  int i,
  int mutated_pos,
  const shared_ptr<OrderedMutationSteps> &ordered_mut_steps,
  const shared_ptr<ObservedSample> &obs_sample,
  const pair<bool, vector<double> > &theta
) {
  VectorNucleotide intermediate_nucs = common::GetMutatedNucleotideVector(
    ordered_mut_steps->mut_steps[i - 1]->nuc_vec,
    mutated_pos,
    obs_sample->end_nucs.val[mutated_pos]
  );

  // copy over old feature indices and then update positions near the mutation
  VectorFeature intermediate_feats;
  intermediate_feats.val = ordered_mut_steps->mut_steps[i - 1]->feature_vec.val;
  int start_p = max(mutated_pos - motif_len_half, 0);
  int end_p = min(mutated_pos + motif_len_half + 1, obs_sample->num_pos);
  for (int p = mutated_pos - motif_len_half; p < end_p; p++) {
    int prev_feat_idx = ordered_mut_steps->mut_steps[i - 1]->feature_vec.val[p];
    if (p == mutated_pos || prev_feat_idx == MUTATED) {
      intermediate_feats.val[p] = MUTATED;
    } else {
    }
    intermediate_feats.val[p] = GetFeatureIdxForPos(p, intermediate_nucs, obs_sample->left_flank, obs_sample->right_flank);
  }

  pair<bool, VectorThetaSums> theta_sum_option(theta.first, VectorThetaSums());
  if (theta.first) {
    // copy over old theta sum values and then update positions near the mutation
    theta_sum_option.second.val = ordered_mut_steps->mut_steps[i - 1]->theta_sum_option.second.val;
    for (int p = start_p; p < end_p; p++) {
      int feat_idx = intermediate_feats.val[p];
      if (feat_idx == MUTATED) {
        theta_sum_option.second.val[p] = MUTATED;
      } else {
        theta_sum_option.second.val[p] = theta.second[feat_idx];
      }
    }
  }

  return make_shared<MutationStep>(
    intermediate_nucs,
    intermediate_feats,
    theta_sum_option
  );
}
