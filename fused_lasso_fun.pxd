# distutils: language=c++

cdef extern from "fused_lasso_solver/test.h":
    void solve_fused_lasso(
      double lmin,
      double lmax,
      int numdeep,
      const char *infile_name,
      const char *outfile_name
    )
