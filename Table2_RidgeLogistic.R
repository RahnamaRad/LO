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
p_               =     seq(100, 1100, 200)
delta            =     1
alpha_elnet      =     0
MCMCsamples      =     100
lo               =     matrix(rep(0, MCMCsamples* length(p_))  ,MCMCsamples, length(p_))
outErr           =     matrix(rep(0, MCMCsamples* length(p_))  ,MCMCsamples, length(p_))
MSE              =     rep(0,length(p_))
bias             =     rep(0,length(p_))


a                =   0
R2               =   0

for (i in 1:length(p_)){
  p              =     p_[i] 
  n              =     p * delta
  for (s in 1:MCMCsamples){
    beta.star      =     rep(1, p)
    X              =     matrix(rnorm( n*p, mean = 0, sd = 1/sqrt(n) ), nrow = n, ncol = p)
    py             =     exp(X %*% beta.star) / (1 + exp(X %*% beta.star))
    y              =     rep(0, n)
    for (ss in 1:n){
      y[ss]=rbinom(1,1,py[ss])
    }
    lambdaS         =     c(0.01,0.1)/n
    lo.glm          =     cv.glmnet(X, y, family = "binomial", alpha = alpha_elnet,  intercept = FALSE, standardize = FALSE, lambda = lambdaS, nfolds = n, type.measure = "deviance")
    fit             =     glmnet(X, y, family = "binomial", alpha = alpha_elnet,  intercept = FALSE, standardize = FALSE, lambda = lo.glm$lambda[1])
    lo[s, i]        =     lo.glm$cvm[1]/2
    
    dz              =      0.005
    oz              =      sqrt(sum(beta.star^2)/n) 
    z               =      seq(-5*oz, 5*oz, dz)
    Dz              =      exp(-z^2/(2*oz^2))/sqrt(2*pi*oz^2) * dz
    outErr[s, i]    =      - as.numeric(t(fit$beta) %*% beta.star) / sum(beta.star^2) * sum(z * exp(z)/(1 + exp(z)) *Dz) 
    
    oz              =      sqrt(sum(fit$beta^2)/n) 
    z               =      seq(-5*oz, 5*oz, dz)
    Dz              =      exp(-z^2/(2*oz^2))/sqrt(2*pi*oz^2) * dz
    outErr[s, i]    =      outErr[s, i]    + sum(log(1+ exp(z)) * Dz)
    
    # for debugging 
    if (FALSE) {
      n.test         =     10000*n
      X.test         =     matrix(rnorm( n.test*p, mean = 0, sd = 1/sqrt(n) ), nrow = n.test, ncol = p)
      Ey.Xtest       =     exp(X.test %*% beta.star) / (1 + exp(X.test %*% beta.star))
      bb             =     as.numeric(t(fit$beta) %*% beta.star) / sum(beta.star^2) * beta.star
      outErr_        =     mean( -(X.test %*% fit$beta) * Ey.Xtest + log(1 + exp(X.test %*% fit$beta) ))
      outErrr_       =     mean( -(X.test %*% bb) * Ey.Xtest + log(1 + exp(X.test %*% fit$beta) )  ) 
      cat(sprintf("p= %1.f| s=%1.f| lo=%.5f| outErr=%.5f | outErr_sample=%.5f |outErr_sample_=%.5f | MSE =%.6f | scaling = %.2f | R2 = %.2f| outErr-lo = %.4f   \n", p, s, lo[s, i], outErr[s, i], outErr_, outErrr_, MSE[i], a, R2,bias[i] ) )
      
    }
    bias[i]        =     mean(-lo[1:s,i] + outErr[1:s, i]) 
    MSE[i]         =     mean((lo[1:s,i] - outErr[1:s, i])^2) 
    if (i>1){
      ls.fit         =     lm(log(MSE[1:i])~log(p_[1:i]))
      a              =     ls.fit$coefficients[2] # should be close to -1 to get us 1/n scaling
      R2             =     summary(ls.fit)$r.squared
    }
    
    cat(sprintf("p= %4.f| s=%3.f| lo=%.5f| outErr=%.5f | MSE =%.6f | scaling = %.2f | R2 = %.2f| lo-outErr = %.4f   \n", p, s, lo[s, i], outErr[s, i], MSE[i], a, R2,bias[i] ) )
  }
}

     

