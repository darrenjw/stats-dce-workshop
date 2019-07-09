## ----eval=FALSE----------------------------------------------------------
## install.packages("BiocManager")
## BiocManager::install()
## BiocManager::install(c("graph", "RBGL", "Rgraphviz"))
## install.packages("gRain")


## ----message=FALSE-------------------------------------------------------
library(gRain)


## ----eval=FALSE----------------------------------------------------------
## vignette("gRain-intro")


## ------------------------------------------------------------------------
tc = cptable(~cond, values=c(98,2),levels=c("Good","Bad"))
it = cptable(~init|cond, values=c(50,25,25,25,25,50),
      levels=c("+","?","-"))
ft = cptable(~final|init:cond, 
      values=c(10,0,8,2,0,10,10,0,2,8,0,10), 
	  levels=c("+","-"))
plist = compileCPT(list(tc,it,ft))
plist


## ------------------------------------------------------------------------
plist$final


## ------------------------------------------------------------------------
net = grain(plist)
net


## ----grain-plot----------------------------------------------------------
plot(net)


## ------------------------------------------------------------------------
querygrain(net, nodes=c("final"), type="marginal")

netP = setEvidence(net, evidence=list(final="+"))
querygrain(netP, nodes=c("cond"), type="marginal")

