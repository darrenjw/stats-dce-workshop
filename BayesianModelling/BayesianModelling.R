## ----conj----------------------------------------------------------------
curve(dgamma(x,34,21), 0,5, col=2,lwd=2, 
  ylab="Density", main="Prior and posterior")
curve(dgamma(x,2,1), 0,5, col=3,lwd=2, add=TRUE)
abline(v=32/20, col=4,lwd=2)


## ----eval=FALSE----------------------------------------------------------
## install.packages("rjags")


## ----message=FALSE-------------------------------------------------------
library(rjags)


## ----eval=FALSE----------------------------------------------------------
## help(package="rjags")
## ?"rjags-package"
## ?jags.model


## ------------------------------------------------------------------------
x=32
data=list(x=x)
init=list(theta=1)
modelstring="
  model {
    x~dpois(20*theta)
    theta~dgamma(2,1)
  }
"


## ------------------------------------------------------------------------
model=jags.model(textConnection(modelstring), 
  data=data, inits=init)


## ------------------------------------------------------------------------
update(model,n.iter=100)
output=coda.samples(model=model,
  variable.names=c("theta"), n.iter=2000, thin=1)


## ------------------------------------------------------------------------
print(summary(output))


## ----conj-jags-----------------------------------------------------------
plot(output)


## ------------------------------------------------------------------------
set.seed(1)
n = 100
x = rgamma(n, 5, 0.5)
summary(x)


## ----gamma-dat-----------------------------------------------------------
hist(x)


## ------------------------------------------------------------------------
library(MASS)
fit = fitdistr(x, "gamma")
fit


## ----gamma-fitd----------------------------------------------------------
hist(x,freq=FALSE)
curve(dgamma(x, fit$estimate[1], fit$estimate[2]),
  0, 30, add=TRUE, col=2, lwd=2)


## ------------------------------------------------------------------------
fitdistr(x, "log-normal")


## ------------------------------------------------------------------------
data=list(x=x, n=n)
init=list(a=1, b=1)
modelstring="
  model {
    for (i in 1:n) {
      x[i] ~ dgamma(a, b)
	}
    a ~ dgamma(1, 0.001)
    b ~ dgamma(1, 0.001)
  }
"


## ------------------------------------------------------------------------
model=jags.model(textConnection(modelstring), 
  data=data, inits=init)


## ------------------------------------------------------------------------
its=2500; thin=6
update(model, n.iter=2000)
output=coda.samples(model=model,
  variable.names=c("a","b"), n.iter=its*thin, thin=thin)


## ------------------------------------------------------------------------
print(summary(output))


## ----gamma-jags----------------------------------------------------------
plot(output)


## ------------------------------------------------------------------------
## convert MCMC output to a matrix
mat = as.matrix(output)
dim(mat)

## select 5 iterations at random
samples = sample(1:its, 5)
samples


## ----gamma-jags-uq-------------------------------------------------------
hist(x, freq=FALSE, ylim=c(0,0.15), main="Sample fits")
for (i in samples) curve(dgamma(x, mat[i,1],
  mat[i,2]), 0, 30, add=TRUE, col=2)


## ------------------------------------------------------------------------
library(lme4)
library(lattice)


## ----eval=FALSE----------------------------------------------------------
## ?sleepstudy
## example(sleepstudy)


## ----xy-sleep,echo=FALSE-------------------------------------------------
xyplot(Reaction ~ Days | Subject, sleepstudy,
  type = c("g","p","r"), index = function(x,y)
  coef(lm(y ~ x))[1], xlab = "Days of sleep deprivation",
  ylab = "Average reaction time (ms)", aspect = "xy")


## ----xy-slopes,echo=FALSE------------------------------------------------
xyplot(Reaction ~ Days | Subject, sleepstudy,
  type = c("g","p","r"), index = function(x,y)
  coef(lm(y ~ x))[1], xlab = "Depth",
  ylab = "Soil property", aspect = "xy")


## ------------------------------------------------------------------------
data=list(y=sleepstudy$Reaction, x=sleepstudy$Days, 
  index=as.numeric(sleepstudy$Subject), 
  n=dim(sleepstudy)[1])
init=list(mu=1, tau=1, tau.s=1)


## ------------------------------------------------------------------------
modelstring="
  model {
    for (i in 1:18) {
	  slope[i] ~ dnorm(mu, tau.s)
	  inter[i] ~ dnorm(0, 0.0001)
	}
    for (j in 1:n) {
      y[j] ~ dnorm(inter[index[j]] + 
	    slope[index[j]]*x[j], tau)
	}
    tau ~ dgamma(1, 0.001)
    tau.s ~ dgamma(1, 0.001)
	mu ~ dnorm(0, 0.0001)
  }
"


## ------------------------------------------------------------------------
model=jags.model(textConnection(modelstring), 
  data=data, inits=init)


## ------------------------------------------------------------------------
its=2500; thin=8
update(model, n.iter=2000)
output=coda.samples(model=model,
  variable.names=c("mu","tau","tau.s"), 
  n.iter=its*thin, thin=thin)


## ------------------------------------------------------------------------
print(summary(output))


## ----re-jags-------------------------------------------------------------
plot(output)

