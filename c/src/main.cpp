#include <iostream>
#include "models.hpp"
#include "common.hpp"
#include "feature_generator.hpp"
using namespace std;

int main()
{
  VectorNucleotide seq = {1, 0, 1, 1};
  int position = 0;
  Nuc target_nuc = 0;
  VectorNucleotide mut_seq = common::get_mutated_string(seq, position, target_nuc);
  cout << mut_seq[0] << mut_seq[1] << "\n";
  cout << seq[0] << seq[1] << "\n";

  SubmotifFeatureGenerator *f = new SubmotifFeatureGenerator(3);
  int feat_idx = f->get_feature_idx_for_pos(1, seq);
  cout << feat_idx<< '\n';

  shared_ptr<ObservedSample> obs_sample = f->create_observed_sample(
    {1, 0, 1, 1},
    {1, 2, 2, 3}
  );

  unique_ptr<OrderedMutationSteps> mm = f->create_for_mutation_steps(
    obs_sample,
    {1, 3, 2},
    make_pair(false, ThetaSums())
  );
  cout<<"========\n";
  for (int i = 0; i < mm->num_steps; i++) {
    for (auto n: mm->mut_steps[i]->nuc_vec) {
        cout << n;
    }
    cout << "\n";
  }

  return 0;
}
