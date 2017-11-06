#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 18:36:26 2017

@author: tarodz
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import theano.tensor as T;
import theano as th;

### START OF SOME FAKE DATASET GENERATION
np.random.seed(123)

#number of samples in total
sCnt=1000;

# true parameters w and b
true_w1=-0.5;
true_w2=1.3;
true_b=-0.3;

# sample some random point in 2D feature space
x_train = np.random.randn(sCnt,2).astype(dtype='float32');

# calculate u=w^Tx+b
u = true_w1*x_train[:,0] + true_w2*x_train[:,1] + true_b;

# P(+1|x)=a(u) #see slides for def. of a(u)
pPlusOne=1.0/(1.0+np.exp(-1.0*u));

# sample realistic (i.e. based on pPlusOne, but not deterministic) class values for the dataset
# class +1 is comes from a probability distribution with probability "prob" for +1, and 1-prob for class 0
y01_train=np.random.binomial(1,pPlusOne);
y_train=2*y01_train-1;
y_train=y_train.astype(dtype='float32');

plt.scatter(x_train[:,0],x_train[:,1],c=y_train)
#### END OF FAKE DATASET GENERATION

#### START OF LEARNING

## define variables for theano

## define non-shared variable types
# x will be our [#samples x #features] matrix of all training samples
# in this example, we'll have 2 features
x = T.matrix('x',dtype='float32');
# y will be our vector of classess for all training samples
y = T.vector('y',dtype='float32')

##define and initialize shared variables 
## (the variable persist, they encode the state of the classifier throughout learning via gradient descent)
# w is the feature weights, a 2D vector 
initialW=np.random.rand(2).astype(dtype='float32');
w = th.shared(value=initialW,name="w");

# b is the bias, so just a single number
initialB=0.0
b = th.shared(value=initialB,name="b");

## set up new variables that are functions/transformations of the above
#predicted class for each sample (a vector)
#T.dot(x,w) is a vector with #samples entries
# even though b is just a number, + will work (through "broadcasting")
# b will be "replicated" #samples times to make both sides of + have same dimension
#thre result is a vector with #samples entries
predictions=T.dot(x,w)+b
#loss for each sample (a vector)
loss=T.log(1.0+T.exp(-y*predictions))
#risk over all samples (a number)
risk=T.mean(loss);


#gradients over w, b
drdw=T.grad(risk,w);
drdb=T.grad(risk,b);


#define a function that performs a single step of gradient descent
train = th.function(
# x,y are inputs        
  inputs = [x,y],
#two outputs: risk, and predictions for each training sample
  outputs = [risk,predictions],
#two updates formulae: w <- w - 0.001 dr/dw , b <- b - 0.001 dr/db
  updates = [(w,w-0.1*drdw),(b,b-0.1*drdb)]
#in real-life code, consider using givens  
  )

#start the iterations of gradient descent
i=1;
done=False;
curr_risk=np.inf;
while (not done):
    prev_risk=curr_risk;
    #single iteration of gradient descent
    #use x_train as x, y_train as y
    #store resulting risk as curr_risk, predictions as predY
    curr_risk,predY=train(x_train,y_train); 
    i=i+1;
    print((i,risk,w.get_value(),b.get_value()))
    #until we stop making improvements
    if ((prev_risk-curr_risk)<0.000001):
        done=True;

## END OF LEARNING
        
print("True w,b: "+str([true_w1,true_w2])+" "+str(true_b));
print("Found w,b:"+str(w.get_value())+" "+str(b.get_value()));
        
