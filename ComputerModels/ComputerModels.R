## ------------------------------------------------------------------------
Grid = seq(0,1,0.002)
Dist = as.matrix(dist(Grid))
K = function(d, lengthScale=0.4, scale=10)
  (scale^2)*exp(-(d/lengthScale)^2)
rGP = function(r=0.4, s=10) {
  Var = K(Dist,r,s) + (1e-5)*diag(length(Grid))
  t(chol(Var)) %*% rnorm(length(Grid))
  }
set.seed(1)


## ----gp1da---------------------------------------------------------------
plot(Grid,rGP(0.01),type="l",lwd=2,col="red")


## ----gp1db---------------------------------------------------------------
plot(Grid,rGP(0.05),type="l",lwd=2,col="red")


## ----gp1dc---------------------------------------------------------------
plot(Grid,rGP(0.1),type="l",lwd=2,col="red")


## ----gp1dd---------------------------------------------------------------
plot(Grid,rGP(0.5),type="l",lwd=2,col="red")


## ------------------------------------------------------------------------
Grid = seq(0,1,0.01)
xGrid = expand.grid(Grid,Grid)
Dist = as.matrix(dist(xGrid))
rGP2d = function(r=0.1, s=10) {
  Var = K(Dist,r,s) + (1e-5)*diag(length(Grid)^2)
  z = t(chol(Var)) %*% rnorm(length(Grid)^2)
  matrix(z,nrow=length(Grid))
  }


## ----gp2da,cache=TRUE----------------------------------------------------
image(rGP2d(0.05))


## ----gp2db,cache=TRUE----------------------------------------------------
image(rGP2d(0.1))


## ----gp2dc,cache=TRUE----------------------------------------------------
image(rGP2d(0.2))


## ----gp2dd,cache=TRUE----------------------------------------------------
image(rGP2d(0.5))


## ------------------------------------------------------------------------
library(DiceKriging)
set.seed(1)
nd = 20
x = as.matrix(runif(nd,0,5))
colnames(x)=c("x")
y = sin(x)+rnorm(nd,0,0.1)


## ------------------------------------------------------------------------
mod = km(~x,design=x,response=y,nugget=0.001)


## ------------------------------------------------------------------------
px = seq(0,6,0.01)
pred = predict(mod,px,type="SK")


## ----dk------------------------------------------------------------------
plot(px,pred$mean,type="l",ylim=c(-2,2),main="DiceKriging")
polygon(c(px,rev(px)),c(pred$upper,rev(pred$lower)),col="lightgrey",border=NA)
lines(px,pred$mean,type="l")
points(x,y,pch=19,col=2)


## ------------------------------------------------------------------------
library(mlegp)
mod = mlegp(x,y,constantMean=0,nugget.known=1,nugget=0.001)


## ------------------------------------------------------------------------
pred = predict(mod,as.matrix(px),se.fit=TRUE)
pred$upper = pred$fit + 2*pred$se.fit
pred$lower = pred$fit - 2*pred$se.fit


## ----mlegp---------------------------------------------------------------
plot(px,pred$fit,type="l",ylim=c(-2,2),main="mlegp")
polygon(c(px,rev(px)),c(pred$upper,rev(pred$lower)),col="lightgrey",border=NA)
lines(px,pred$fit,type="l")
points(x,y,pch=19,col=2)


## ------------------------------------------------------------------------
library(GPfit)
xScaled = x/5
mod = GP_fit(xScaled,y)


## ------------------------------------------------------------------------
pxScaled = px/5
pred = predict(mod,as.matrix(pxScaled))
pred$upper = pred$Y_hat + 2*pred$MSE
pred$lower = pred$Y_hat - 2*pred$MSE


## ----gpfit---------------------------------------------------------------
plot(px,pred$Y_hat,type="l",ylim=c(-2,2),main="GPfit")
polygon(c(px,rev(px)),c(pred$upper,rev(pred$lower)),col="lightgrey",border=NA)
lines(px,pred$Y_hat,type="l")
points(x,y,pch=19,col=2)


## ------------------------------------------------------------------------
library(hetGP)
mod = mleHomGP(x,y) # homoskedastic GP fit
pred = predict(mod,as.matrix(px))
pred$var = pred$sd2 + pred$nugs
pred$sd = sqrt(pred$var)
pred$upper = pred$mean + 2*pred$sd
pred$lower = pred$mean - 2*pred$sd


## ----hetgp---------------------------------------------------------------
plot(px,pred$mean,type="l",ylim=c(-2,2),main="hetGP (mleHomGP)")
polygon(c(px,rev(px)),c(pred$upper,rev(pred$lower)),col="lightgrey",border=NA)
lines(px,pred$mean,type="l")
points(x,y,pch=19,col=2)


## ------------------------------------------------------------------------
library(MASS)
mod = mleHomGP(mcycle$time,mcycle$accel) # homoskedastic
px=seq(0,60,0.1)
pred = predict(mod,as.matrix(px))
pred$var = pred$sd2 + pred$nugs
pred$sd = sqrt(pred$var)
pred$upper = pred$mean + 2*pred$sd
pred$lower = pred$mean - 2*pred$sd


## ----mcycle-hom----------------------------------------------------------
plot(px,pred$mean,ylim=c(-150,100),type="l",main="hetGP (mleHomGP)")
polygon(c(px,rev(px)),c(pred$upper,rev(pred$lower)),col="lightgrey",border=NA)
lines(px,pred$mean,type="l")
points(mcycle,pch=19,col=2)


## ------------------------------------------------------------------------
mod = mleHetGP(mcycle$time,mcycle$accel) # heteroskedastic
pred = predict(mod,as.matrix(px))
pred$var = pred$sd2 + pred$nugs
pred$sd = sqrt(pred$var)
pred$upper = pred$mean + 2*pred$sd
pred$lower = pred$mean - 2*pred$sd


## ----mcycle-het----------------------------------------------------------
plot(px,pred$mean,ylim=c(-150,100),type="l",main="hetGP (mleHetGP)")
polygon(c(px,rev(px)),c(pred$upper,rev(pred$lower)),col="lightgrey",border=NA)
lines(px,pred$mean,type="l")
points(mcycle,pch=19,col=2)


## ------------------------------------------------------------------------
library(lhs)
set.seed(1)


## ----eval=FALSE----------------------------------------------------------
## help(package="lhs")
## ?randomLHS
## ?maximinLHS
## vignette(package="lhs")
## vignette("lhs_basics")


## ----lhs2d,fig.height=4,fig.width=5--------------------------------------
x = maximinLHS(50,2)
plot(x, pch=19, col=2)


## ----lhs2d-margins,fig.height=3,fig.width=5------------------------------
op = par(mfrow=c(1,2))
hist(x[,1]); hist(x[,2])
par(op)


## ----lhs2d-aug,fig.height=4,fig.width=5----------------------------------
y = augmentLHS(x,20)
plot(y,pch=19,col=3); points(x,pch=19,col=2)

