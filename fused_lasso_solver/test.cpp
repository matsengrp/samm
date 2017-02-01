#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "graph.h"
#include "test.h"


class MotifThetaVals {
  string motif;
  double value;

  public:
    MotifThetaVals(string s, double v) {
      motif = s;
      value = v;
    }
    string get_motif() {return motif;}
    double get_value() {return value;}
};

class ProblemSetting{
  double lambda;
  vector<MotifThetaVals> all_motif_vals;

  public:
    ProblemSetting(double l, vector<MotifThetaVals> a) {
      lambda = l;
      all_motif_vals = a;
    }
    double get_lambda() {return lambda;}
    vector<MotifThetaVals> get_motif_vals() {return all_motif_vals;}
};



bool differ_by_one_char(std::string s1, std::string s2){
  int num_diff = 0;
  for (int k = 0; k < 3; k++) {
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

  std::ifstream graph_file(file_name);

  double lambda;
  graph_file >> lambda;

  std::string feature_name; //most of this is going to be the motif
  int feature_idx; //not used right now, but nice for sanity checks
  double graph_val; //theta value from proximal gradient descent
  while (graph_file >> feature_name >> feature_idx >> graph_val) {
    // printf("%s %d %f\n", a.c_str(), i, b);
    graph_motifs.push_back(MotifThetaVals(feature_name, graph_val));
  }
  return ProblemSetting(lambda, graph_motifs);
}

int solve_fused_lasso(double lmin, double lmax, int numdeep, const char *infile_name, const char *outfile_name) {
  // This is going to find the minimizer of
  // 0.5 * || theta - g ||^2_2 + lambda * fused lasso penalty

  int numlevel = (1 << numdeep) - 1;
  printf("numlevel %d\n", numlevel);

  ProblemSetting prob = read_graph_vals(infile_name);
  double lambda = prob.get_lambda();
  vector<MotifThetaVals> graph_motifs = prob.get_motif_vals();

  int vector_size = graph_motifs.size();

  // construct levels for the dyadic TV solver
  double *lev0 = (double *) malloc((numlevel+1)*sizeof(double));
  double *lev = (double *) malloc((numlevel+1)*sizeof(double));

  double dl=(lmax-lmin)/(double)numlevel;
  lev[0]=lev0[0]=lmin; //lev[0] never used
  for (int l=1;l<=numlevel;l++) {
    lev0[l]=dl+lev0[l-1];
    lev[l]=lev0[l-1]+dl/2.;
  }
  int idl = (1+numlevel) >> 1;

  // Now construct the graph
  Graph::node_id * nodes = (Graph::node_id *) malloc(vector_size * sizeof(Graph::node_id));
  Graph *BKG = new Graph();

  for (int i=0; i < vector_size; i++) {
    nodes[i] = BKG->add_node();
    BKG->set_tweights(
      nodes[i],
      graph_motifs[i].get_value(),
      lev[idl] // start the dyadic TV solver at the middle level (lev[idl])
    );
  }

  // add edges between motifs that differ by one character
  for (int i=0; i < vector_size; i++) {
    for (int j=0; j < vector_size; j++) {
      if (i == j) {
        continue;
      }
      bool is_diff = differ_by_one_char(graph_motifs[i].get_motif(), graph_motifs[j].get_motif());
      // printf("%s, %s: %d\n", motif_list[i].c_str(), motif_list[j].c_str(), is_diff);
      if (is_diff) {
        // add edge
        BKG->add_edge(nodes[i], nodes[j], lambda, lambda);
      }
    }
  }

  // solve
  BKG->dyadicparametricTV(numdeep, idl*dl/2);
  //if (BKG->error()) { fprintf(stderr,"error in maxflow\n"); exit(0); }

  // OUTPUT the fitted values
  std::ofstream out_file(outfile_name);
  for (int k=0; k < vector_size; k++) {
    out_file << k << " " << lev0[BKG->what_label(nodes[k])] << "\n" ;
    printf("%f\n", lev0[BKG->what_label(nodes[k])]);
  }
  out_file.close();

  delete BKG;
  return 1;
}

int main() {
  solve_fused_lasso();
}
