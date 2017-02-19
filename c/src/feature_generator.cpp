#include <assert.h>

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
  VectorNucleotide end_nuc
) {
  VectorFeature feature_vec;
  for (int i = 0; i < start_nuc.val.size(); i++) {
    feature_vec.val.push_back(GetFeatureIdxForPos(i, start_nuc));
  }
  shared_ptr<ObservedSample> obs_sample(new ObservedSample(
    start_nuc,
    end_nuc,
    feature_vec
  ));
  return obs_sample;
}

unique_ptr<OrderedMutationSteps> SubmotifFeatureGenerator::CreateForMutationSteps(
  const shared_ptr<ObservedSample> &obs_sample,
  VectorOrder mut_order,
  const pair<bool, vector<double>> &theta
) {
  unique_ptr<OrderedMutationSteps> ordered_mut_steps(new OrderedMutationSteps(mut_order));
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

int SubmotifFeatureGenerator::GetFeatureIdxForPos(int position, const VectorNucleotide &nuc_vec) {
  // Initial testing shows that this is faster than stoi and map lookup
  int idx = 0;
  int base = 1;
  for (int i = 0; i < motif_len; i++) {
    if (position + motif_len_half - i < 0 || position + motif_len_half - i > nuc_vec.val.size() - 1) {
      // TODO: replace me with reasonable things
      return 10000;
    }
    idx += nuc_vec.val[position + motif_len_half - i] * base;
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
  const unique_ptr<OrderedMutationSteps> &ordered_mut_steps,
  const shared_ptr<ObservedSample> &obs_sample,
  const pair<bool, vector<double>> &theta
) {
  VectorNucleotide intermediate_nucs;
  VectorFeature intermediate_feats;
  pair<bool, VectorThetaSums> theta_sum_option(theta.first, VectorThetaSums());

  intermediate_nucs = common::get_mutated_nucleotide_vector(
    ordered_mut_steps->mut_steps[i - 1]->nuc_vec,
    mutated_pos,
    obs_sample->end_nucs.val[mutated_pos]
  );

  // copy over old feature indices and then update positions near the mutation
  intermediate_feats.val = ordered_mut_steps->mut_steps[i - 1]->feature_vec.val;
  for (int p = mutated_pos - motif_len_half; p < mutated_pos + motif_len_half + 1; p++) {
    if (p < 0 || p >= obs_sample->num_pos) {
      intermediate_feats.val[p] = 10000;
    } else {
      int prev_feat_idx = ordered_mut_steps->mut_steps[i - 1]->feature_vec.val[p];
      if (p == mutated_pos || prev_feat_idx == MUTATED) {
        intermediate_feats.val[p] = MUTATED;
      } else {
        intermediate_feats.val[p] = GetFeatureIdxForPos(p, intermediate_nucs);
      }
    }
  }

  if (theta.first) {
    // copy over old theta sum values and then update positions near the mutation
    theta_sum_option.second.val = ordered_mut_steps->mut_steps[i - 1]->theta_sum_option.second.val;
    for (int p = mutated_pos - motif_len_half; p < mutated_pos + motif_len_half + 1; p++) {
      if (p < 0 || p >= obs_sample->num_pos) {
        // TODO: fix me
        theta_sum_option.second.val[p] = 10000;
      } else {
        int feat_idx = intermediate_feats.val[p];
        if (feat_idx == MUTATED) {
          theta_sum_option.second.val[p] = MUTATED;
        } else {
          theta_sum_option.second.val[p] = theta.second[feat_idx];
        }
      }
    }
  }

  return make_shared<MutationStep>(
    intermediate_nucs,
    intermediate_feats,
    theta_sum_option
  );
}
