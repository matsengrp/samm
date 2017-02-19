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
    int feature_vec_len;
    SubmotifFeatureGenerator(int motif_len);

    // Make an ObservedSample object from the raw observed data
    // Calculates the features before any mutations have occurred
    shared_ptr<ObservedSample> CreateObservedSample(
      VectorNucleotide start_nucs,
      VectorNucleotide end_nucs
    );

    // Calculate features for all mutation steps
    // Before the first mutation to before the last mutation
    // Doesn't calculate the features after the last mutation since that is not relevant.
    // Calculates theta sums if theta is passed in (so the first elem in pair is positive)
    unique_ptr<OrderedMutationSteps> CreateForMutationSteps(
      const shared_ptr<ObservedSample> &obs_sample,
      VectorOrder mut_order,
      const pair<bool, vector<double>> &theta
    );

    unique_ptr<OrderedMutationSteps> UpdateForMutationSteps(
      const shared_ptr<ObservedSample> &obs_sample,
      VectorOrder mut_order,
      const vector<int> update_steps,
      const unique_ptr<OrderedMutationSteps> &base_sample,
      const pair<bool, vector<double>> &theta
    );

    // The feature index of a position is the base 4 number corresponding to the motif
    // at the position
    int GetFeatureIdxForPos(int position, const VectorNucleotide &nuc_seq);

  private:
    int motif_len_half;

    /*
    * Does the heavy lifting for calculating feature vectors at a given mutation step
    * @param i: mutation step index
    * @param mutated_pos: the position that is mutating
    * @param ordered_mut_steps: the object that needs its i-th step updated
    * @param obs_sample: the observed sample
    * @param theta: if passed in, calculate theta sum values too
    *
    * @return the updated i-th mutation step
    */
    shared_ptr<MutationStep> UpdateMutationStep(
      int i,
      int mutated_pos,
      const unique_ptr<OrderedMutationSteps> &ordered_mut_steps,
      const shared_ptr<ObservedSample> &obs_sample,
      const pair<bool, vector<double>> &theta
    );
};

#endif
