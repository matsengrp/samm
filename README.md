# mobeef

## Setup for shmulate

To include packages necessary for `shumlate`, first run

```
R --vanilla --slave -e 'install.packages(c("igraph", "lazyeval", "ggplot2", "plyr", "seqinr", "shazam"), repos="http://cran.rstudio.com/")'
```

## EM installation
To fit the models, we're using CVXPY for now (until it barfs). Install this:
```
pip install --user cvxpy
```
