'''.learn() implements learning algorithm exactly as described in paper.

Convergence type tests are vestigial from solving the problem described in result 6'''

import numpy as np
import math
import random
import datetime

CONV_1ST_DEG = 1
CONV_2ND_DEG = 2
CONV_TIME_LIMIT = 3
CONV_NOT_DONE = 0


class Parameters:
    def __init__(self):
        self.num_features = None #Auto-generated
        self.num_groups = None
        self.group_overlap = None
        self.group_size = None
        self.num_examples = None
        self.sparsity_param = None
        self.desired_accuracy = None
        self.noise_variance = None
        self.time_limit = None
        self.convergence_limit = None # Auto-generated
        self.training_feature_sparsity = None


###########################################################
# Learning algorithm

# From top of page 4
def group_weight(group):
    return math.sqrt(group.__len__())


# From equation 4
def build_c(groups, sparsity_param, num_features):
    c_height = 0
    for group in groups:
        c_height += group.__len__()
    c = np.zeros((c_height, num_features))
    i_g_pair = 0
    for group in groups:
        w = group_weight(group)
        for j in group:
            c[i_g_pair, j] = sparsity_param * w
            i_g_pair += 1
    return c


# From equation 8
def abs_gamma(groups, num_examples, sparsity_param): #TODO switch inner and outer loops to improve time complexity
    weights_norms = [] #TODO figure out how to describe this
    for j in range(num_examples):
        sum_squares = 0.0
        for group in groups:
            if j in group:
                sum_squares += math.pow(group_weight(group),2)
        weights_norms.append(math.sqrt(sum_squares))
    # print(weight_totals)
    return sparsity_param * max(weights_norms)

# print("abs_gamma", abs_gamma(groups, params.num_examples, sparsity_param=sparsity_param))


# From equation 10
def lipschitz_constant(x, groups, num_examples, sparsity_param, mu):
    xt_x = np.matmul(np.transpose(x),x)
    eigens = np.linalg.eig(xt_x)[0].tolist()
    eigens = [elem.real for elem in eigens if elem.imag == 0.0]  #TODO confirm max real
    #print("eigens", eigens)
    max_eigen = max(eigens)
    #print("max_eigen", max_eigen)
    abs_g = abs_gamma(groups, num_examples, sparsity_param)

    return max_eigen + (math.pow(abs_g, 2) / mu) #TODO not sure what's going on with the power, seems to work better with 1


def shrinkage(arr):
    norm = np.linalg.norm(arr, ord=2) #TODO Make sure correct ord
    if norm > 1.0:
        return arr/norm
    else:
        return arr

#from lemma 1
def gen_opt_alpha(beta, groups, sparsity_param, mu):
    alpha_gs = []
    for group in groups:
        beta_g = []
        for j in group:
            beta_g.append(beta[j])
        bg = np.array(beta_g)
        ag = bg * sparsity_param * group_weight(group) / mu

        #print("b/a", beta_g, shrinkage(ag))
        # print("norm alpha", np.linalg.norm(np.array(alpha_g)))
        alpha_gs += shrinkage(ag).tolist()
    total_opt_alpha = np.array(alpha_gs)
    # print("alpha_star", alpha_gs)
    return total_opt_alpha  # vector with length = sum of group sizes, same as height of C

# print("opt alpha", gen_opt_alpha(b, groups, sparsity_param))


# Equation 9
def f_squiggle_gradient(x, y, b, c, groups, sparsity_param, mu):
    opt_alpha = gen_opt_alpha(b, groups, sparsity_param, mu)
    term2 = np.matmul(np.transpose(c), opt_alpha) #TODO check dimensions
    term1 = np.matmul(np.transpose(x), ((np.matmul(x, b)) - y))
    return term1 + term2


def test_convergence(t, params, betas_t, weights_t, z_t, gradient_t):
    if (t<2):
        return CONV_NOT_DONE
    change_arr = np.absolute(np.subtract(betas_t[t], betas_t[t - 1]))
    change_in_beta = np.mean(change_arr)
    prior_change_in_beta = np.mean(np.absolute(np.subtract(betas_t[t-1], betas_t[t - 2])))
    change_over_two_betas = np.mean(np.absolute(np.subtract(betas_t[t], betas_t[t - 2])))
    max_change = 0.0
    max_change_index = 0
    beta = betas_t[t]
    for i in range(beta.shape[0]):
        if change_arr[i] > max_change:
            max_change = change_arr[i]
            max_change_index = i

    if t % 100 == -1:
        i = max_change_index
        print("t", t, "change", change_in_beta, "max_c", max_change, "i", i,
              "mc_beta", betas_t[t][i], "mc_w", weights_t[t][i], "mc_z", z_t[t][i], "mc_gr", gradient_t[t][i])

    #if abs(change_in_beta - prior_change_in_beta) < 0.00000001 and change_in_beta > 0.01:
    if abs(change_over_two_betas) < params.convergence_limit/1000 and change_in_beta > params.convergence_limit * 10:
        print("Convergence due to 2nd-degree change in beta")
        return CONV_2ND_DEG
    if change_in_beta < params.convergence_limit:
        # print("Convergence due to 1st-degree change in beta")
        return CONV_1ST_DEG
    # if t > 2000:
    #     # print("No convergence, stopping")
    #     return CONV_TIME_LIMIT
    return CONV_NOT_DONE


def learn(x, y, groups, params):
    params.convergence_limit = 0.001 / params.desired_accuracy

    c = build_c(groups, 5.0, params.num_features)
    # print("c", c)
    mu = params.desired_accuracy / groups.__len__()

    lip_constant = lipschitz_constant(x, groups, params.num_examples, params.sparsity_param, mu)
    # print("lip constant", lip_constant)

    gradient_t = []
    weights_t = []
    beta_t = []
    z_t = []
    weights_t.append(np.zeros(params.num_features))
    #beta_t.append(np.zeros(params.num_features)) #TODO delete

    start = datetime.datetime.now()
    t = 0
    while True:
        #step 1
        gradient_t.append(f_squiggle_gradient(x, y, weights_t[t], c, groups, params.sparsity_param, mu))
        #gradient_t.append(f_squiggle_gradient(x, y, beta_t[t], groups, sparsity_param, mu)) #TODO delete
        #step 2
        #print("gradient/lip_constant", gradient, lip_constant, gradient/lip_constant)
        beta_t.append(weights_t[t] - (gradient_t[t]/(lip_constant))) # TODO tried dividing by 2 here
        #beta_t.append(beta_t[t] - (gradient_t[t] / (lip_constant))) #TODO delete
        #step 3
        z = -((t + 1) * gradient_t[t] / 2) / lip_constant
        if (t > 0): z += z_t[t-1]
        z_t.append(z)
        #print("z/lip_constant", z, lip_constant, z / lip_constant)
        #step 4
        weights = ((t+1)*beta_t[t] / (t+3)) + (2*z_t[t] / (t+3))
        weights_t.append(weights)
        convergence_type = test_convergence(t, params, beta_t, weights_t, z_t, gradient_t)

        end = datetime.datetime.now()
        tim = end - start
        runtime_ms = tim.seconds * 1000 + tim.microseconds / 1000
        if runtime_ms > params.time_limit:
            convergence_type = CONV_TIME_LIMIT
        if convergence_type:
            break
        t += 1

    learned_beta = beta_t[t]
    return learned_beta, runtime_ms, t, convergence_type














