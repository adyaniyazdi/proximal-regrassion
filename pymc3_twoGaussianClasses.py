#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:36:26 2017

@author: tarodz
Partially adapted from: http://austinrochford.com/posts/2015-09-16-mvn-pymc3-lkj.html
"""

import pymc3 as pm;
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import theano.tensor as T;

### START OF SOME FAKE DATASET GENERATION
np.random.seed(123)

#number of samples in total
N=1000

#true probability of class 1 and 0
p1=.6;
p0=1.0-p1;

#sample number of samples from class 1 n1 ~ Binomial(N,p1)
n1=sp.stats.binom.rvs(N,p1,size=1);
n0=N-n1;


# true means of the gaussians for class 1 and class 0
# 2 features, i.e. 2D guassians
true_mu1 = np.array([1,1]);
true_mu0 = np.array([-1,-1]);

# true covariance matrix, same for both classes
# 2 features, so covariance is 2x2 matrix

true_cov_sqrt = sp.stats.uniform.rvs(0, 2, size=(2, 2))
true_cov = np.dot(true_cov_sqrt.T, true_cov_sqrt);

# sample feature vectors (2D) from the true gaussians

x1 = sp.stats.multivariate_normal.rvs(true_mu1, true_cov, size=n1)
x0 = sp.stats.multivariate_normal.rvs(true_mu0, true_cov, size=n0)

#y_1 = np.ones((n1,))
#y_0 = np.zeros((n0,))

#### END OF FAKE DATASET GENERATION


#### START OF MODEL BUILDING AND ESTIMATION

# instantiate an empty PyMC3 model
basic_model = pm.Model()

# fill the model with details:
with basic_model:

    # parameters for priors for gaussian means
    mu_prior_cov = np.array([[1., 0.], [0., 1]])
    mu_prior_mu = np.zeros((2,))
    
    # Priors for gaussian means (Gaussian prior): mu1 ~ N(mu_prior_mu, mu_prior_cov), mu0 ~ N(mu_prior_mu, mu_prior_cov)
    mu1 = pm.MvNormal('mu1', mu=mu_prior_mu, cov=mu_prior_cov, shape=2)
    mu0 = pm.MvNormal('mu0', mu=mu_prior_mu, cov=mu_prior_cov, shape=2)
    
    # Prior for gaussian covariance matrix (LKJCorr prior):
    # see here for details: http://austinrochford.com/posts/2015-09-16-mvn-pymc3-lkj.html
    # and here: http://docs.pymc.io/notebooks/LKJ.html
    nu = pm.Uniform('nu', 0, 5)
    C_triu = pm.LKJCorr('C_triu', nu, 2) 
    C = pm.Deterministic('C', T.fill_diagonal(C_triu[np.zeros((2, 2), dtype=np.int64)], 1.))
    sigma_my = pm.Lognormal('sigma_my', np.zeros(2), np.ones(2), shape=2)
    sigma_diag = pm.Deterministic('sigma_diag', T.nlinalg.diag(sigma_my))
    cov_both = pm.Deterministic('cov', T.nlinalg.matrix_dot(sigma_diag, C, sigma_diag))
  
    # observations x1, x0 are supposed to be N(mu1,cov_both), N(mu0,cov_both)
    # here is where the Dataset (x1,x0) comes to influence the choice of paramters (mu1,mu0, cov_both)
    # this is done through the "observed = ..." argument; note that above we didn't have that
    
    x1_obs = pm.MvNormal('x1', mu=mu1,cov=cov_both, observed = x1);
    x0_obs = pm.MvNormal('x0', mu=mu0,cov=cov_both, observed = x0);
    
# done with setting up the model
    
    

# now perform maximum likelihood (actually, maximum a posteriori (MAP), since we have priors) estimation
# it's a dictionary: "parameter name" -> "it's estimated value"
map_estimate1 = pm.find_MAP(model=basic_model)


# calculate covariance estimate from C_triu estimate 
# (needed because distribution of covariance matrices is a bit convoluted through the LKJ prior)
C_triu_est=map_estimate1['C_triu'];
sigma_est=np.exp(map_estimate1['sigma_my_log_']);
C_est=C_triu_est[np.zeros((2, 2), dtype=np.int64)];
np.fill_diagonal(C_est, 1.);
sigma_diag_est = np.diag(sigma_est);
cov_est = np.dot(np.dot(sigma_diag_est, C_est), sigma_diag_est)

# we can also do MCMC sampling from the distribution over the parameters
# and e.g. get confidence intervals
#
with basic_model:

    # obtain starting values via MAP
    start = pm.find_MAP()

    # instantiate sampler
    step = pm.Slice()

    # draw 5000 posterior samples
    trace = pm.sample(5000, step=step, start=start)

pm.traceplot(trace)
pm.summary(trace)

