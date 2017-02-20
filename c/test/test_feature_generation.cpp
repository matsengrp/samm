#include "catch.hpp"

#include <iostream>
#include <string>
#include <map>
#include <utility>
#include <chrono>

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

  SECTION("Time test"){
    // We're measuring time here for easy comparison to the python code.
    int VEC_LEN = 400;
    VectorNucleotide start_nucs = common::GetRandomVectorNucleotide(VEC_LEN);
    VectorNucleotide end_nucs;
    // Mutate every 10 nucleotides
    for (int i = 0; i < VEC_LEN; i++) {
      Nuc start_nuc = start_nucs.val[i];
      Nuc new_nucleotide;
      if (i % 10 == 0) {
        if (start_nuc == 0) {
          new_nucleotide = 3;
        } else {
          new_nucleotide = start_nuc - 1;
        }
      } else {
        new_nucleotide = start_nuc;
      }
      end_nucs.val.push_back(new_nucleotide);
    }

    auto start_time = chrono::system_clock::now();
    shared_ptr<ObservedSample> obs_sample = feat_gen.CreateObservedSample(
      start_nucs,
      end_nucs,
      left_flank,
      right_flank
    );
    auto end_time = chrono::system_clock::now();
    chrono::duration<double> diff = end_time-start_time;
    cout << "Time" << diff.count() << "\n";

    auto start_time1 = chrono::system_clock::now();
    VectorOrder my_order = {obs_sample->mutated_postions};

    // Try different mutation orders - similar calculations as a gibbs step
    vector<shared_ptr<OrderedMutationSteps> > all_the_mut_steps;

    // Position mutates last
    all_the_mut_steps.push_back(feat_gen.CreateForMutationSteps(
      obs_sample,
      my_order,
      pair<bool, vector<double> >(true, theta)
    ));
    for (int i = my_order.val.size() - 2; i >= 0; i--) {
      // Position is the i-th mutation

      // Create the new ordering
      VectorOrder new_order;
      for (int j = 0; j < i; j++) {
        new_order.val.push_back(my_order.val[j]);
      }
      new_order.val.push_back(my_order.val[my_order.val.size() - 1]);
      for (int j = i; j < my_order.val.size() - 1; j++) {
        new_order.val.push_back(my_order.val[j]);
      }

      // Indicate which steps need updating
      vector<int> update_steps;
      if (i != my_order.val.size() - 2) {
        update_steps = {i + 1, i + 2};
      } else {
        update_steps = {i + 1};
      }

      // Actually update the feature calculations
      all_the_mut_steps.push_back(feat_gen.UpdateForMutationSteps(
        obs_sample,
        new_order,
        update_steps,
        all_the_mut_steps[my_order.val.size() - i - 2],
        pair<bool, vector<double> >(true, theta)
      ));
    }

    auto end_time1 = chrono::system_clock::now();
    chrono::duration<double> diff1 = end_time1-start_time1;
    cout << "Time" << diff1.count() << "\n";
  }

}
