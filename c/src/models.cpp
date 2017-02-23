#include <cassert>
#include "models.hpp"

ObservedSample::ObservedSample(
  VectorNucleotide s,
  VectorNucleotide e,
  VectorNucleotide l,
  VectorNucleotide r,
  VectorFeature f
): start_nucs(s), end_nucs(e), left_flank(l), right_flank(r), start_features(f) {
  assert(s.val.size() == e.val.size());
  num_pos = s.val.size();
  for (unsigned int i = 0; i < num_pos; i++) {
    if (s.val[i] != e.val[i]) {
      mutated_postions.push_back(i);
    }
  }
}

OrderedMutationSteps::OrderedMutationSteps(VectorOrder ord_v) {
  num_steps = ord_v.val.size();
  order_vec = ord_v;
  mut_steps.resize(num_steps);
}

void OrderedMutationSteps::set(int step_i, shared_ptr<MutationStep> mut_step) {
  mut_steps[step_i] = mut_step;
}
