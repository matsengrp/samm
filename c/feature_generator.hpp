#ifndef FEATURE_GEN_H
#define FEATURE_GEN_H

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include "models.hpp"

using namespace std;

class SubmotifFeatureGenerator {
  public:
    int motif_len;
    SubmotifFeatureGenerator(int motif_len);

    shared_ptr<ObservedSample> create_observed_sample(
      VectorNucleotide start_nuc,
      VectorNucleotide end_nuc,
      VectorOrder mut_order,
    );

    unique_ptr<FeatureMutationSteps> create_for_mutation_steps(
      const ObservedSample &obs_sample,
      const pair<bool, vector<double>> &theta
    );

  private:
    int flank_end_len;
    int feature_vec_len;
    int create_feature_vec_for_pos(int position, NucSeq nuc_seq);
};

#endif
