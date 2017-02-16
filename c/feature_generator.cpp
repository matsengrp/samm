#include "feature_generator.hpp"
#include "common.hpp"

FeatureMutationSteps::FeatureMutationSteps(const ObservedSequenceMutationsFeatures &obs_seq_mut) {
  seq_str_steps.resize(obs_seq_mut.mut_order.val.size());

  NucSeq start_seq = obs_seq_mut.start_seq;

  seq_str_steps.at(0) = obs_seq_mut.start_seq;
  seq_feature_steps.at(0) = obs_seq_mut.start_seq_features;
}

SubmotifFeatureGenerator::SubmotifFeatureGenerator(int m) {
  cout << motif_len;
  flank_end_len = motif_len/2;
  feature_vec_len = 1 << (2 * motif_len);
}

unique_ptr<ObservedSequenceMutationsFeatures> create_base_features(
  string start_seq, string end_seq
) {
  return NULL;
}

shared_ptr<FeatureMutationSteps> SubmotifFeatureGenerator::create_for_mutation_steps(
  const ObservedSequenceMutationsFeatures &seq_mut_order
) {
  int num_steps = seq_mut_order.mut_order.val.size();
  shared_ptr<FeatureMutationSteps> feat_mut_steps(new FeatureMutationSteps(seq_mut_order));

  NucSeq intermediate_seq = seq_mut_order.start_seq;
  for (int i = 0; i < num_steps - 1; i++) {
    int mutating_pos = seq_mut_order.mut_order.val[i];
    intermediate_seq = common::mutate_string(
      intermediate_seq,
      mutating_pos,
      seq_mut_order.end_seq.val[mutating_pos]
    );

    // this->create_feature_vec_for_pos(position, nuc_seq);
  }

  return feat_mut_steps;
}

shared_ptr<FeatureMutationStepsWithThetaSums> SubmotifFeatureGenerator::create_for_mutation_steps(
  const ObservedSequenceMutationsFeatures &seq_mut_order,
  const vector<double> &theta
) {
  int num_steps = seq_mut_order.mut_order.val.size();
  shared_ptr<FeatureMutationStepsWithThetaSums> feat_mut_steps(new FeatureMutationStepsWithThetaSums(num_steps));
  return feat_mut_steps;
}

int SubmotifFeatureGenerator::create_feature_vec_for_pos(int position, NucSeq nuc_seq) {
  int idx = 1;
  for (int i = position + this->motif_len - 1; i >= position; i--) {
    idx = idx << (2 * nuc_seq.val[i]);
  }
  return idx - 1;
}
