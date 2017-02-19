#include "catch.hpp"

#include <iostream>
#include <string>
#include <map>
#include <utility>

#include "models.hpp"
#include "common.hpp"
#include "feature_generator.hpp"

TEST_CASE("Feature Generator") {

  int motif_len = 3;

  SubmotifFeatureGenerator feat_gen(motif_len);

  vector<double> theta;
  for (int i = 0; i < feat_gen.feature_vec_len; i++) {
    theta.push_back(i);
  }

  // Total of three mutations
  VectorNucleotide start_nucs = {{0,1,2,1,1,1,0,0,0}};
  VectorNucleotide end_nucs =   {{0,1,2,1,2,1,1,1,0}};
  VectorNucleotide left_flank = {{0,0}};
  VectorNucleotide right_flank = {{1,1}};

  vector<int> start_features_true = {1, 6, 25, 37, 21, 20, 16, 0, 1};
  vector<double> start_features_theta_sum = {1, 6, 25, 37, 21, 20, 16, 0, 1};

  shared_ptr<ObservedSample> obs_sample = feat_gen.CreateObservedSample(
    start_nucs,
    end_nucs,
    left_flank,
    right_flank
  );

  VectorOrder my_order = {obs_sample->mutated_postions};
  shared_ptr<OrderedMutationSteps> mm_base = feat_gen.CreateForMutationSteps(
    obs_sample,
    my_order,
    pair<bool, vector<double> >(true, theta)
  );

  SECTION("Check feature generation is correct")
  {
    REQUIRE(obs_sample->start_features.val == start_features_true);

    REQUIRE(mm_base->mut_steps.size() == my_order.val.size());
    REQUIRE(mm_base->mut_steps[0]->theta_sum_option.second.val == start_features_theta_sum);
    REQUIRE(mm_base->mut_steps[1]->theta_sum_option.second.val[my_order.val[0] - 1] == 38);
    REQUIRE(mm_base->mut_steps[1]->theta_sum_option.second.val[my_order.val[0] + 1 ] == 36);
    REQUIRE(mm_base->mut_steps[2]->theta_sum_option.second.val[my_order.val[0] + 1 ] == 37);
  }

  SECTION("Check feature update")
  {
    VectorOrder new_order = {
      {
        obs_sample->mutated_postions[0],
        obs_sample->mutated_postions[2],
        obs_sample->mutated_postions[1]
      }};

    // Create the mutation steps slowly
    shared_ptr<OrderedMutationSteps> mm_new = feat_gen.CreateForMutationSteps(
      obs_sample,
      new_order,
      pair<bool, vector<double> >(true, theta)
    );

    // Reuse the old computations to calculate the mutation steps quickly
    vector<int> update_steps = {1, 2};
    shared_ptr<OrderedMutationSteps> mm_new_fast = feat_gen.UpdateForMutationSteps(
      obs_sample,
      new_order,
      update_steps,
      mm_base,
      pair<bool, vector<double> >(true, theta)
    );

    // Check values match
    for (int i = 0; i < mm_new->num_steps; i++) {
        REQUIRE(mm_new_fast->mut_steps[i]->feature_vec.val == mm_new->mut_steps[i]->feature_vec.val);
    }

    // Make sure we didn't change the old object
    REQUIRE(mm_new_fast->mut_steps[2]->feature_vec.val != mm_base->mut_steps[2]->feature_vec.val);
  }
}
