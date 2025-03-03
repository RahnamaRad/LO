# Clear all variables
rm(list = ls())
# Clear the screen
cat("\014")

#################### NEG LIKELIHOOD + RIDGE + LASSO FUNCTION #######################
compute_likelihood <- function(X, beta, lmb1, lmb2, y) {
  X_beta            =  X %*% beta
  ell               =  -y * X_beta + log(1 + exp(X_beta))
  likelihood        =  sum(ell) + lmb1 * sum(abs(beta)) + (lmb2 / 2) * sum(beta^2)
  return(likelihood)
}
############ GRADIENT of neg-Likelihood + Ridge ########################
compute_gradient <- function(X, beta, lmb2, y) {
  beta    = as.numeric(beta)
  X_beta  = X %*% beta
  dot_ell = -y + exp(X_beta) / (1 + exp(X_beta))
  gradient    =  t(X) %*% dot_ell + lmb2 * beta
  return(gradient)
}
##################################### FUNCTION TO PERFOM Proximal gradient to ELNET LOGISTIC #############################
# Optimality check function for elastic net logistic regression
check_optimality <- function(X, y, beta, lmb1, lmb2) {
  activeGrad     = 0 # the gradient wrt active betas
  opt_flag_count = 0 # number opt cond violated of non active betas
  n              = dim(X)[1]
  gradient       = compute_gradient(X, beta, lmb2, y) 
  gradient_min   = gradient #initialize and make the elements of the zero elements as close as 
  for (j in 1:length(beta)) {
    if (beta[j] != 0) {
      activeGrad       = max(activeGrad, abs(gradient[j] + lmb1 * sign(beta[j])) )
      gradient_min[j]  = gradient_min[j] + lmb1 * sign(beta[j]) 
    } else {
      if (lmb1 < gradient[j] || gradient[j] < -lmb1){
        gmin_a           = gradient_min[j] + lmb1
        gmin_b           = gradient_min[j] - lmb1
        gradient_min[j]  = ifelse(abs(gmin_a) < abs(gmin_b), gmin_a, gmin_b)
        opt_flag_count   = opt_flag_count + 1
        #cat(sprintf("Component %d: gradient[j]/lmb1 = [%.2f], (beta[%d] = 0), g[j]= %.2f \n", j, gradient[j]/lmb1, j, gradient_min[j]))
      } else {
        gradient_min[j]  = 0
      }
    }
  }
  #cat(sprintf("-----------------------\n"))
  return(list(activeGrad = activeGrad, opt_flag_count = opt_flag_count, gradient_min = gradient_min))
}
################################################################################
# Proximal operator for L1 norm (soft-thresholding)
proximal_operator <- function(beta, lambda, pos = FALSE) {
  
  if(pos){ return(pmax(0, abs(beta) - lambda))
         }else{ return(sign(beta) * pmax(0, abs(beta) - lambda))}
}
################################################################################
# Proximal Gradient Descent for Logistic Lasso
proximal_gradient_descent <- function(X, y, beta_init, lmb1, lmb2, tol, max_iter) {
  alpha             = 0.9
  min_learning_rate = 0.00001
  beta              = beta_init
  h                 = 0
  n                 = dim(X)[1]
  for (iter in 1:max_iter) {
    grad                = compute_gradient(X, beta, lmb2, y)
    learning_rate       = 1 # line search
    initial_likelihood  = compute_likelihood(X, beta, lmb1, lmb2, y)
    while (compute_likelihood(X, proximal_operator(beta - learning_rate * grad, learning_rate * lmb1), lmb1, lmb2, y) > initial_likelihood) {
      learning_rate = learning_rate * alpha
      if (learning_rate < min_learning_rate) break
    }
    # Update beta
    beta          =  proximal_operator(beta - learning_rate * grad, learning_rate * lmb1)
    if(pos){ beta[beta<0] = 0}
    df            = sum(beta != 0)/length(beta)
    optCond       = check_optimality(X, y, beta, lmb1, lmb2)
    #if (iter %% 1 == 0) {
    #  cat(sprintf("iter=%d| L=%.4f| df=%.2f| |g|_2^2/n=%.2f| max act g=%.4f| nonActViol=%d| lrn_rate=%.4f \n", iter, compute_likelihood(X, beta, lmb1, lmb2, y), df, mean(optCond$gradient_min^2), optCond$activeGrad, optCond$opt_flag_count, learning_rate))
    #}
    if (sqrt(sum(optCond$gradient_min^2)) < tol) {
      break
    }
    #if ((optCond$activeGrad < tol) && (optCond$opt_flag_count == 0)) {
    #  break
    #}
  }
  #cat(sprintf("-----------------------\n"))
  check_optimality(X, y, beta, lmb1, lmb2)
  return(beta)
}
##################################### MAIN CODE #############################
library(Matrix)
library(rstudioapi)
library(pracma)
#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

p_ = 1

pos         = FALSE

lo          = matrix(0, 1, 1)
outErr      = matrix(0, 1, 1)
inErr       = matrix(0, 1, 1)
MSE         = numeric(1)
MSE_SE      = numeric(1)
bias        = numeric(1)
var         = numeric(1)
a           = 0
R2          = 0
tol         = 0.01  # (sqrt(mean(grad^2)) of ELNET
max_iter    = 100
exec_time   = 0
results = matrix(NA, length(p_), 2)

Xtrain = read.csv("Xtrain.csv", header = FALSE)
Xtest = read.csv("Xtest.csv", header = FALSE)

Xtrain = as.matrix(Xtrain)
dimnames(Xtrain) = NULL

Xtest = as.matrix(Xtest)
dimnames(Xtest) = NULL


ytrain = read.csv("ytrain.csv", header = FALSE)
ytest = read.csv("ytest.csv", header = FALSE)

ytrain = unlist(ytrain)
ytest = unlist(ytest)


localc = function(lambda){

for (i in 1:1) {
  k_lo           = 5#nrow(Xtrain) #k_lo_[i]
  start_time     = proc.time()  # Record the start time
  for (s in 1:1) {
    
    X = Xtrain
    Xtest = Xtest

    p = ncol(X)
    n = nrow(X)

    X = t(t(X) - colMeans(X))
    Xtest = t(t(Xtest) - colMeans(Xtest))

    for(j in 1:p){
        X[,j] = X[,j]/(sqrt(p)*sd(X[,j]))
        Xtest[,j] = Xtest[,j]/(sqrt(p)*sd(Xtest[,j]))}

    y = unlist(ytrain)
    #y = y - mean(y)

    ytest = unlist(ytest)
    #ytest = ytest - mean(ytest)

    lmb1   = 0.1*lambda # ell_1
    lmb2   = 0.1*lambda  # ell_2



    beta_hat     = proximal_gradient_descent(X=X, y=y, beta_init = rep(0,p), lmb1 = lmb1, lmb2 = lmb2, tol = tol, max_iter=max_iter)

    # compute OO 
    
    py_hat       = plogis((X %*% beta_hat)[, 1])
    inErr[s, i]  = -mean(y * log(py_hat) + (1 - y) * log(1 - py_hat))
    
    py_test_hat  = plogis((Xtest %*% beta_hat)[, 1])
    outErr[s, i]  = -mean(ytest * log(py_test_hat) + (1 - ytest) * log(1 - py_test_hat))

    #outErr[s, i] = outErr[s, i] + sum(log(1+exp(w)) * Dw) 

    # compute LO

    samp_mat = matrix(sample(1:n), nrow = k_lo)

    for (j in 1:k_lo) {
      beta_hat_j     = proximal_gradient_descent(X=X[-samp_mat[j,], ], y=y[-samp_mat[j,]], beta_init = rep(0,p), lmb1 = lmb1, lmb2 = lmb2, tol = tol, max_iter=max_iter) 
      optCond        = check_optimality(X=X[-samp_mat[j,], ], y=y[-samp_mat[j,]], beta=beta_hat_j, lmb1, lmb2)
      df             = sum(beta_hat_j != 0)/length(beta_hat_j)
      #cat(sprintf("df=%.2f| max active g=%.4f| nonActiveViolations=%d \n", df, optCond$activeGrad, optCond$opt_flag_count))
      #plot(beta_hat_i, beta_star, main = paste("Iteration:", j))
      #hist(as.numeric(beta_hat_i-beta_star), main = paste("Iteration:", j, "|err|_2:", sum((beta_hat_i-beta_star)^2)))
      #readline(prompt = "Press [Enter] to continue to the next iteration...")
      #cat("-------------------------------------\n")
      bta_j_xj       = as.numeric((X[samp_mat[j,], ]) %*% beta_hat_j)
      pyj_j          = exp(bta_j_xj) / (1 + exp(bta_j_xj))
      lo[s, i]       = lo[s, i] - sum(y[samp_mat[j,]] * log(pyj_j) + (1 - y[samp_mat[j,]]) * log(1 - pyj_j))/n
      #if (j %% 10 == 0) {
       #cat(sprintf("s= %d|j=%d|p=%d| df=%.2f| max active g=%.2f| betaZeroViols=%d \n", s, j, p, df, optCond$activeGrad, optCond$opt_flag_count))
      #}
      
      if(j %% 100 == 0){ print(c(i, p_[i], j, df, sum((beta_hat - beta_hat_j)^2)))}

      if(i ==6){ if(j %% 50 == 0) {
					
		cat(sprintf("p=%4d|n=%4d|j=%4d|df=%.2f| inErr=%.2f|LO= %.2f|OO= %.2f\n", 
                  p, n, j, df, inErr[s, i], lo[s, i]*k_lo/j, outErr[s, i]))


		}}
    }
    #if (s %% 100 == 0) {
     # cat(sprintf("s= %d| p=%d| df=%.2f| max active g=%.2f| betaZeroViols=%d \n", s, p, df, optCond$activeGrad, optCond$opt_flag_count))
    #}
    #cat("----------------------------\n")

    
    
    
  } # END OF MCMC SAMPLES FOR LOOP
  
  #print(c(i, p_[i], outErr[s, i], lo[s,i]))
  cat(sprintf("p=%4d|n=%4d|s=%3d|df=%.2f| lmb1= %.1f|lmb2=%.1f| inErr=%.2f|LO= %.2f|OO= %.2f\n", 
                  p, n, s, df, lmb1, lmb2, inErr[s, i], lo[s, i], outErr[s, i]))

  
} # END OF THE LOOP going over the ps

return(c(inErr, outErr, lo))}

