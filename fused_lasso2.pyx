# distutils: language=c++
# distutils: sources=["fused_lasso_solver/test.cpp"]

cimport fused_lasso_fun

def happy():
    fused_lasso_fun.solve_fused_lasso(
      -5.0,
      5.0,
      16,
      "in.txt",
      "out.txt"
    )
