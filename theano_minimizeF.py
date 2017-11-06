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


minimum=[1,3]

#define some function using thaeano Ops (e.g. T.dot for matrix multiplication)
# this function is f(w)=||w-minimum||^2, and so has minimum at minimum, i.e. at vector [1,3]
def f(w):
    shiftedW=w-np.array(minimum);
    return T.dot(shiftedW.T,shiftedW);

#define starting value of W for gradient descent
#here, W is a 2D vector
initialW=np.random.rand(2)

#create a shared variable (i.e. a variable that persists between calls to a theano function)
w = th.shared(value=initialW,name="w");

#define output of applying f to w
#out goal will be to minimize f(w), i.e. find w with lowest possible f(w)
z=f(w);
#define gradient of f w.r.t. w
dzdw=T.grad(z,w);


#define a function that performs a single step of gradient descent
minimize = th.function(
# no inputs (w persists between calls, is not an input)        
  inputs = [],
#single output, current value of z
  outputs = [z],
#a single update formula: w <- w - 0.01 df/dw
  updates = [(w,w-0.01*dzdw)]
  )

#start the iterations of gradient descent
i=1;
done=False;
while (not done):
    z=minimize(); 
    #print: iternation #, current value of z, current vector w    
    print((i,z,w.get_value()))
    i=i+1;
    #until we found w with f(w)<0.001
    if (z[0]<0.001):
        done=True;

print("True minimum: "+str(minimum));
print("Found minimum:"+str(w.get_value()));
