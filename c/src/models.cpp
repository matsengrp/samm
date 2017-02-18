#include "models.hpp"

ObservedSample::ObservedSample(
  VectorNucleotide s,
  VectorNucleotide e,
  VectorFeature f
): start_nucs(s), end_nucs(e), start_features(f) {
    num_pos = s.val.size();
}

OrderedMutationSteps::OrderedMutationSteps(VectorOrder ord_v) {
  num_steps = ord_v.val.size();
  order_vec = ord_v;
  mut_steps.resize(num_steps);
}

void OrderedMutationSteps::set(int step_i, shared_ptr<MutationStep> mut_step) {
  mut_steps[step_i] = mut_step;
}
