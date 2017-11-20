import numpy as np
import random

def continuous_structure_beta(params, groups):
    real_beta = np.random.normal(0, 1, params.num_features)
    for i in range(params.num_features // params.training_feature_sparsity, params.num_features):
        real_beta[i] = 0.0
    return real_beta


def seperate_structure_beta(params, groups):
    real_beta = np.zeros(params.num_features)
    for i in range(groups.__len__()):
        if i % 20 == 0:
            for j in groups[i]:
                real_beta[j] = random.gauss(0, 1)
    #print("sparse groups beta:", real_beta)
    return real_beta


def unstructured_control_beta(params, groups):
    real_beta = np.zeros(params.num_features)
    for i in range(params.num_features):
        if i % 20 == 0:
            real_beta[i] = random.gauss(0, 1)
    #print("sparse alternating beta:", real_beta)
    return real_beta



def banner(msg):
    print("=" * 75)
    print(msg)
    print("=" * 65)

def minimize(f, low_lim, up_lim, accuracy):

    while (up_lim - low_lim > accuracy):
        difference = up_lim - low_lim
        l1 = low_lim + (difference/3.0)
        l2 = low_lim + (2*difference/3.0)
        # print("l1:", l1, "l2:", l2)
        if (f(l1) > f(l2)):
            low_lim = l1
        else:
            up_lim = l2
    return (up_lim + low_lim)/2