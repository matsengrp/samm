#ifndef FUSED_LASSO_H
#define FUSED_LASSO_H

void solve_fused_lasso(
  double lmin = -5.0, // the minimum value of the solution
  double lmax = 5.0, // the maximum value of the solution
  int numdeep = 16,
  const char *infile_name = "in.txt",
  const char *outfile_name="out.txt"
);

#endif
