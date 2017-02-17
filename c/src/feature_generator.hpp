#ifndef FEATURE_GEN_HPP
#define FEATURE_GEN_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <utility>

#include "models.hpp"

using namespace std;

class SubmotifFeatureGenerator {
  // Use this in VectorFeature to indicate that this position has mutated
  const int MUTATED = -1;

  public:
    int motif_len;
    SubmotifFeatureGenerator(int motif_len);

    // Make an ObservedSample object from the raw observed data
    // Calculates the features before any mutations have occurred
    shared_ptr<ObservedSample> create_observed_sample(
      VectorNucleotide start_nucs,
      VectorNucleotide end_nucs
    );

    // Calculate features for all mutation steps
    // Before the first mutation to before the last mutation
    // Doesn't calculate the features after the last mutation since that is not relevant.
    // Calculates theta sums if theta is passed in (so the first elem in pair is positive)
    unique_ptr<OrderedMutationSteps> create_for_mutation_steps(
      const shared_ptr<ObservedSample> &obs_sample,
      VectorOrder mut_order,
      const pair<bool, vector<double>> &theta
    );

    // The feature index of a position is the base 4 number corresponding to the motif
    // at the position
    int get_feature_idx_for_pos(int position, const VectorNucleotide &nuc_seq);

  private:
    int motif_len_half;
    int feature_vec_len;
};

#endif
