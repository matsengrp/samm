# mobeef

## Fused Lasso installation
We need to compile the Cython component:
```
pip install --user cython
```
Compile the TVexact code (on stoat) in the TVexact folder:
```
cc -O -fPIC -c *.cpp
ar rus libTVH.a *.o
```
Compile the Cython code:
```
python setup.py build_ext --inplace
```

## EM installation
To fit the models, we're using CVXPY for now (until it barfs). Install this:
```
pip install --user cvxpy
```

## R package installation
To get `shmulate` to fit an S5F model, install packages in `R` using:
```
R --vanilla --slave -e 'install.packages(c("igraph", "seqinr", "R.utils", "chron", "plyr", "shazam"), repos="http://cran.rstudio.com/")'
```

## Run tests
Run all tests:
```
python -m unittest discover
```
Run specific tests:
```
python -m unittest test.test__YOUR_TEST__
```
Run specific test method:
```
python -m unittest test.test__YOUR_TEST__.__YOUR_TEST_CLASS__TestCase.__YOUR_TEST_METHOD__
```
