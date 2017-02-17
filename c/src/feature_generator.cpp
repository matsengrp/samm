#include <assert.h>

#include "feature_generator.hpp"
#include "common.hpp"

SubmotifFeatureGenerator::SubmotifFeatureGenerator(int m) {
  assert(m % 2 == 1);
  motif_len = m;
  motif_len_half = motif_len/2;
  feature_vec_len = 1 << (2 * motif_len);
}

shared_ptr<ObservedSample> SubmotifFeatureGenerator::create_observed_sample(
  VectorNucleotide start_nuc,
  VectorNucleotide end_nuc
) {
  VectorNucleotide feature_vec;
  for (int i = 0; i < start_nuc.size(); i++) {
    feature_vec.push_back(get_feature_idx_for_pos(i, start_nuc));
  }
  shared_ptr<ObservedSample> obs_sample(new ObservedSample(
    start_nuc,
    end_nuc,
    feature_vec
  ));
  return obs_sample;
}

unique_ptr<OrderedMutationSteps> SubmotifFeatureGenerator::create_for_mutation_steps(
  const shared_ptr<ObservedSample> &obs_sample,
  VectorOrder mut_order,
  const pair<bool, vector<double>> &theta
) {
  unique_ptr<OrderedMutationSteps> ordered_mut_steps(new OrderedMutationSteps(mut_order));

  ordered_mut_steps->set(
    0,
    make_shared<MutationStep>(
      obs_sample->start_seq,
      obs_sample->start_seq_features,
      make_pair(false, ThetaSums())
    )
  );

  for (int i = 1; i < ordered_mut_steps->num_steps; i++) {
    int mutated_pos = mut_order[i - 1];
    VectorNucleotide intermediate_seq = common::get_mutated_string(
      ordered_mut_steps->mut_steps[i - 1]->nuc_vec,
      mutated_pos,
      obs_sample->end_seq[mutated_pos]
    );

    // Do something really stupid right now. Just make a new one.
    // TODO: be smart
    VectorFeature intermediate_feats;
    for (int p = 0; p < obs_sample->num_pos; p++) {
      int prev_feat_idx = ordered_mut_steps->mut_steps[i - 1]->feature_vec[p];
      if (p == mutated_pos || prev_feat_idx == MUTATED) {
        intermediate_feats.push_back(MUTATED);
      } else {
        intermediate_feats.push_back(
          get_feature_idx_for_pos(p, intermediate_seq)
        );
      }
    }
    ordered_mut_steps->set(
      i,
      make_shared<MutationStep>(
        intermediate_seq,
        intermediate_feats,
        make_pair(false, ThetaSums())
      )
    );
  }

  return ordered_mut_steps;
}

int SubmotifFeatureGenerator::get_feature_idx_for_pos(int position, const VectorNucleotide &nuc_seq) {
  // Initial testing shows that this is faster than stoi and map lookup
  int idx = 0;
  int base = 1;
  for (int i = 0; i < motif_len; i++) {
    if (position + motif_len_half - i < 0 || position + motif_len_half - i > nuc_seq.size() - 1) {
      // TODO: replace me with reasonable things
      return 10000;
    }
    idx += nuc_seq[position + motif_len_half - i] * base;
    base = base << 2;
  }
  return idx;
}
