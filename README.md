# mobeef

## Fused Lasso installation
We need to compile the Cython component:
```
pip install --user cython
```
Compile the Cython code:
```
python setup.py build_ext --inplace
```
Compile the TVDyadic code (in linux) in the TVDyadic folder:
```
cc -O -c *.cpp
ar rus libTVH.a *.o
```

## EM installation
To fit the models, we're using CVXPY for now (until it barfs). Install this:
```
pip install --user cvxpy
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
