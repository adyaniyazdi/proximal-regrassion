#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:36:26 2017

@author: tarodz
"""

import pymc3 as pm;
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import theano.tensor as T;

### START OF SOME FAKE DATASET GENERATION
np.random.seed(123)

#number of samples in total
sCnt=1000;

# true parameters w and b
true_w1=-0.5;
true_w2=1.3;
true_b=-0.3;

# sample some random point in 2D feature space
X1 = np.random.randn(sCnt)
X2 = np.random.randn(sCnt) 

# calculate u=w^Tx+b
u = true_w1*X1 + true_w2*X2 + true_b;

# P(+1|x)=a(u) #see slides for def. of a(u)
pPlusOne=1.0/(1.0+np.exp(-1.0*u));

# sample realistic (i.e. based on pPlusOne, but not deterministic) class values for the dataset
# class +1 is comes from a probability distribution with probability "prob" for +1, and 1-prob for class 0
Y=np.random.binomial(1,pPlusOne);

#### END OF FAKE DATASET GENERATION

#### START OF MODEL BUILDING AND ESTIMATION

# instantiate an empty PyMC3 model
basic_model = pm.Model()

# fill the model with details:
with basic_model:

    # Priors for unknown model parameters
    w1 = pm.Normal('w1',0,100);
    w2 = pm.Normal('w2',0,100);
    b  = pm.Normal('b',0,100);

    # calculate u=w^Tx+b
    linfit = w1*X1 + w2*X2 + b;
    # P(+1|x)=a(u) #see slides for def. of a(u)
    prob = 1.0 / (1.0 + T.exp(-1.0*linfit));
    
    # class +1 is comes from a probability distribution with probability "prob" for +1, and 1-prob for class 0
    Y_obs=pm.Bernoulli('Y_obs',p=prob,observed = Y);
    
# done with setting up the model


# now perform maximum likelihood (actually, maximum a posteriori (MAP), since we have priors) estimation
# it's a dictionary: "parameter name" -> "it's estimated value"
map_estimate1 = pm.find_MAP(model=basic_model)

# we can also do MCMC sampling from the distribution over the parameters
# and e.g. get confidence intervals
with basic_model:

    # obtain starting values via MAP
    start = pm.find_MAP()

    # instantiate sampler
    step = pm.Slice()

    # draw 5000 posterior samples
    trace = pm.sample(50000, step=step, start=start)

pm.traceplot(trace)
pm.summary(trace)