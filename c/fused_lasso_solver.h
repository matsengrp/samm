#ifndef FUSED_LASSO_H
#define FUSED_LASSO_H


// Minimize 0.5 * || theta - g ||^2_2 + lambda * fused lasso penalty on theta
// The fused lasso penalty is || theta[features_to_fuse1] - theta[features_to_fuse2] ||_1
// with respect to theta
void solve_fused_lasso(
  double *array_to_fill, // fills this array with the solution
  double *feature_vals, // the values for each
  int vector_size, // size of array_to_fill and feature_vals (should be the same)
  int *features_to_fuse1, // fused lasso penalty: first index of theta
  int *features_to_fuse2, // fused lasso penalty: second index of theta
  int num_features_to_fuse, // size of features_to_fuse1 and features_to_fuse2 (sould be the same)
  double penalty_param, // penalty parameter
  double lmin, // the minimum value of the solution
  double lmax, // the maximum value of the solution
  int numdeep = 16 // the degree of precision to use
);

#endif
