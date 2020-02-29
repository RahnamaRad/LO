rm(list = ls())    #delete objects
cat("\014")
library(class)
library(ggplot2)
library(gridExtra)
library(dplyr)
library(glmnet)
library(alocv)
library(rmutil)
library(tictoc)
library(scales)
set.seed(1)
# generate data
p_               =     seq(400, 2000, 400)
delta            =     0.1
alpha_elnet      =     0.5
MCMCsamples      =     100
lo               =     matrix(rep(0, MCMCsamples* length(p_))  ,MCMCsamples, length(p_))
outErr           =     matrix(rep(0, MCMCsamples* length(p_))  ,MCMCsamples, length(p_))
MSE              =     rep(0,length(p_))
bias             =     rep(0,length(p_))

o                =   1
a                =   0
R2               =   0

for (i in 1:length(p_)){
  p              =     p_[i] 
  n              =     p * delta
  k              =     0.1*n
  for (s in 1:MCMCsamples){
    #beta.star       =     rep(1, p)
    beta.star      =    rep(0, p)
    iS             =    sample(1:p, k)
    beta.star[iS]  =    rlaplace(k, m=0, s=1/sqrt(2))
    X               =     matrix(rnorm( n*p, mean = 0, sd = 1/sqrt(n) ), nrow = n, ncol = p)
    y               =     X %*% beta.star + rnorm(n, mean = 0, sd = o)
    lambdaS         =     c(0.5,5)/n
    lo.glm          =     cv.glmnet(X, y,  alpha = alpha_elnet,  intercept = FALSE, standardize = FALSE, lambda = lambdaS, nfolds = n, type.measure = "deviance")
    fit             =     glmnet(X, y, alpha = alpha_elnet,  intercept = FALSE, standardize = FALSE, lambda = lo.glm$lambda[1])
    lo[s, i]        =     lo.glm$cvm[1]/2
    beta.hat        =     fit$beta
    outErr[s, i]    =     (o^2 +  sum((beta.star - beta.hat)^2)/n)/2

    bias[i]        =     mean(-lo[1:s,i] + outErr[1:s, i]) 
    MSE[i]         =     mean((lo[1:s,i] - outErr[1:s, i])^2) 
    if (i>1){
      ls.fit         =     lm(log(MSE[1:i])~log(p_[1:i]))
      a              =     ls.fit$coefficients[2] # should be close to -1 to get us 1/n scaling
      R2             =     summary(ls.fit)$r.squared
    }
    
    cat(sprintf("p= %4.f| s=%3.f| df=%3.f, lmb=%.3f| lo=%.3f| outErr=%.3f| MSE =%.3f | scaling = %.2f | R2 = %.2f| lo-outErr = %.4f   \n", p, s, fit$df, lo.glm$lambda[1]*n,  lo[s, i], outErr[s, i], MSE[i], a, R2,bias[i] ) )
  }
}



MSE.SE   =    apply((lo - outErr)^2, 2, sd)/sqrt(MCMCsamples)
apply((lo - outErr)^2, 2, mean)


