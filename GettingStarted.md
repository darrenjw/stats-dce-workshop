# Getting started

## Introduction

If you would like to participate actively in the session, you should make sure that R and R-Studio are installed on your laptop in advance. Spending an hour or two starting to getting familiar with R and R-Studio would also be very beneficial.

## Installing R

You can download R from:

* https://www.r-project.org/

Select the download link and install the appropriate version of R for your OS.

## Installing R-Studio

Technically, R is all that you need. However, many people, especially R beginners, find R-Studio to be useful. R-Studio is a slick IDE for R that makes R much easier to use. You can download it from:

* https://www.rstudio.com/

Note that you should first install R before attempting to install R-Studio.

## Installing some R packages

R comes with lots of functionality out-of-the-box, but also has thousands of add-on packages that are usually very easy to install. Most packages are distributed via [CRAN](https://cran.r-project.org/), and these can typically installed simply by entering `install.packages("packagename")` into an R console (where `packagename` is the name of the required package).

We can install some packages that will be useful for this tutorial with:

```r
install.packages(c("rmarkdown","coda","smfsb","MASS", "lme4", "lattice", "lhs", "DiceKriging",
   "GPfit", "hetGP", "mlegp"))
```

You should be able to copy-and-paste this command from the web page. Package installation will take a few minutes.

### gRain

Part 1 will use an R package called `gRain`.

Unfortunately it is less easy to install than most R packages on [CRAN](https://cran.r-project.org/), since it has dependencies on some packages that are not in CRAN, but instead in another R package repository known as [Bioconductor](https://bioconductor.org/). So installing `gRain` in a recent version of R should proceed roughly as follows:

```r
install.packages("BiocManager")
BiocManager::install()
BiocManager::install(c("graph", "RBGL", "Rgraphviz"))
install.packages("gRain")
```

Again, copy-and-paste to avoid typos. You can test it worked with:
```r
library(gRain)
```

This shouldn't give an error if the package is correctly installed.

### JAGS and rjags

Part 2 will use some free software called JAGS, and an interface to it from R called `rjags`.

[JAGS](http://mcmc-jags.sourceforge.net/) (Just Another Gibbs Sampler) is an example of some general-purpose free software for carrying out Markov chain Monte Carlo (MCMC) simulation for (intractable) Bayesian hierarchical models. Note that JAGS is a standalone piece of software - it is not an R package, and must be installed separately (outside of R) before being used, and may require admin privileges. Follow the instructions on the JAGS website for installation.

* http://mcmc-jags.sourceforge.net/

Once JAGS is correctly installed on your system, there is an R package on CRAN called `rjags` which is straightforward to install, and makes it easy to use JAGS from within R. It should be possible to install with
```r
install.packages("rjags")
```
but note that installation of this package will fail if JAGS is not already correctly installed. 

Once JAGS and `rjags` are installed it can be loaded for use in the usual way.
```{r message=FALSE}
library(rjags)
```
This should return without error if everything is correctly installed.

## Getting familiar with R and R-Studio

Once you have R and R-Studio installed, it is worth spending a little time getting familiar with them. Below are a few links that might be worth browsing (in roughly the order listed), in order to begin to get familiar with R and how it works. It is not necessary to explore all links exhaustively.

* https://www.computerworld.com/article/2497143/business-intelligence-beginner-s-guide-to-r-introduction.html
* http://www.r-tutor.com/r-introduction
* http://www.mas.ncl.ac.uk/~ndjw1/teaching/sim/R-intro.html
* https://www.guru99.com/r-tutorial.html
* http://www.stats.bris.ac.uk/R/doc/manuals/R-intro.html#A-sample-session
* http://www.stats.bris.ac.uk/R/doc/manuals/R-intro.html
* https://www.rstudio.com/online-learning/
* http://portal.stats.ox.ac.uk/userdata/ruth/APTS2013/APTS.html


