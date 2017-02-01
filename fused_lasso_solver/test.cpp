#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "graph.h"
#include "test.h"

using namespace std;

struct MotifThetaVals {
  // Container for motif and its corresponding node value
  string motif;
  double value;

  MotifThetaVals(string s, double v) {
    motif = s;
    value = v;
  }
};

struct ProblemSetting{
  double lambda;
  vector<MotifThetaVals> all_motif_vals;

  ProblemSetting(double l, vector<MotifThetaVals> a) {
    lambda = l;
    all_motif_vals = a;
  }
};

bool differ_by_one_char(string s1, string s2){
  // Determine if strings are off by one character
  int num_diff = 0;
  for (int k = 0; k < min(s1.length(), s2.length()); k++) {
    // printf("?? %s %s\n", s1[k], s2[k]);
    if (s1[k] != s2[k]) {
      num_diff++;
      if (num_diff > 1) {
        return false;
      }
    }
  }
  return num_diff == 1;
}

ProblemSetting read_graph_vals(const char *file_name) {
  vector<MotifThetaVals> graph_motifs;

  ifstream graph_file(file_name);

  double lambda;
  graph_file >> lambda;

  string feature_name; //most of this is going to be the motif
  int feature_idx; //not used right now, but nice for sanity checks
  double graph_val; //theta value from proximal gradient descent
  while (graph_file >> feature_name >> feature_idx >> graph_val) {
    // printf("%s %d %f\n", a.c_str(), i, b);
    graph_motifs.push_back(MotifThetaVals(feature_name, graph_val));
  }
  return ProblemSetting(lambda, graph_motifs);
}

void solve_fused_lasso(double lmin, double lmax, int numdeep, const char *infile_name, const char *outfile_name) {
  // This is going to find the minimizer of
  // 0.5 * || theta - g ||^2_2 + lambda * fused lasso penalty

  int numlevel = (1 << numdeep) - 1;
  printf("numlevel %d\n", numlevel);

  ProblemSetting prob = read_graph_vals(infile_name);
  double lambda = prob.lambda;
  vector<MotifThetaVals> graph_motifs = prob.all_motif_vals;

  int vector_size = graph_motifs.size();

  // construct levels for the dyadic TV solver
  double *lev0 = new double[numlevel + 1];
  double *lev = new double[numlevel + 1];

  double dl = (lmax - lmin) / (double)numlevel;
  lev[0] = lev0[0] = lmin;
  for (int l = 1; l <= numlevel; l++) {
    lev0[l] = dl + lev0[l - 1];
    lev[l] = lev0[l - 1] + dl / 2.0;
  }
  int idl = (1 + numlevel) >> 1;

  // Now construct the graph
  Graph::node_id *nodes = new Graph::node_id[vector_size];
  Graph *BKG = new Graph();

  for (int i = 0; i < vector_size; i++) {
    nodes[i] = BKG->add_node();
    BKG->set_tweights(
      nodes[i],
      graph_motifs[i].value,
      lev[idl] // start the dyadic TV solver at the middle level (lev[idl])
    );
  }

  // add edges between motifs that differ by one character
  for (int i = 0; i < vector_size - 1; i++) {
    for (int j = i + 1; j < vector_size; j++) {
      bool is_diff = differ_by_one_char(graph_motifs[i].motif, graph_motifs[j].motif);
      // printf("%s, %s: %d\n", motif_list[i].c_str(), motif_list[j].c_str(), is_diff);
      if (is_diff) {
        // add edge
        BKG->add_edge(nodes[i], nodes[j], lambda, lambda);
      }
    }
  }

  // solve max flow problem
  BKG->dyadicparametricTV(
    numdeep, // number of levels explored by the max-flow algo (higher means more accurate)
    idl * dl / 2 // delta perturbation during the max-flow algo
  );

  // Output the fitted values
  ofstream out_file(outfile_name);
  for (int i = 0; i < vector_size; i++) {
    // The solution to the graph total variation problem is the lev0 bin of the node.
    double node_val = lev0[BKG->what_label(nodes[i])];
    out_file << i << " " << node_val << "\n" ;
    printf("%f\n", node_val);
  }
  out_file.close();

  delete BKG;
  delete[] nodes;
  delete[] lev0;
  delete[] lev;
}

int main() {
  solve_fused_lasso();
}
