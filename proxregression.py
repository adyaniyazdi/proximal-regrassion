import numpy as np
import math
import random
import datetime


class Parameters:
    def __init__(self):
        self.num_features = None
        self.num_groups = None
        self.group_overlap = None
        self.group_size = None
        self.num_examples = None

def generate_groups(params):
    groups=[]

    for i in range(params.num_groups):
        group_start_index = i * (params.group_size - params.group_overlap)
        groups.append(list(range(group_start_index, group_start_index + params.group_size)))

    # print("groups: ", groups)
    params.num_features = params.group_overlap + (params.num_groups * (params.group_size - params.group_overlap)) #J
    # print("num_features=", params.num_features)
    return groups

def generate_training_data(beta, params):
    x = np.random.normal(0, 1, (params.num_examples, params.num_features))
    # print("beta",beta)
    y = np.matmul(x,beta) #TODO add epsilon

    return x, y


def learn(x, y, groups, params, sparsity_param, desired_accuracy):
    # From top of page 4
    mu = desired_accuracy / groups.__len__()

    def group_weight(group):
        return math.sqrt(group.__len__())

    #From equation 4
    def build_c(groups, sparsity_param, num_features):
        c_height = 0
        for group in groups:
            c_height += group.__len__()
        c = np.zeros((c_height, num_features))
        i = 0
        for group in groups:
            w = group_weight(group)
            for member in group:
                c[i, member] = sparsity_param * w
                i += 1
        return c

    c = build_c(groups, 5.0, params.num_features)
    #print("c", c)


    # From equation 8
    def abs_gamma(groups, num_examples, sparsity_param): #TODO switch inner and outer loops to improve time complexity
        weight_totals = [] #TODO figure out how to describe this
        for j in range(num_examples):
            sum_squares = 0.0
            for group in groups:
                if j in group:
                    sum_squares += math.pow(group_weight(group),2)
            weight_totals.append(math.sqrt(sum_squares))
        # print(weight_totals)
        return sparsity_param * max(weight_totals)

    # print("abs_gamma", abs_gamma(groups, params.num_examples, sparsity_param=sparsity_param))


    #From equation 10
    def lipschitz_constant(x, groups, num_examples, sparsity_param):
        xt_x = np.matmul(np.transpose(x),x)
        eigens = np.linalg.eig(xt_x)[0].tolist()
        eigens = [elem.real for elem in eigens if elem.imag == 0.0]
        #print("eigens", eigens)
        max_eigen = max(eigens)
        #print("max_eigen", max_eigen)
        abs_g = abs_gamma(groups, num_examples, sparsity_param)

        return max_eigen + (abs_g / mu)

    lip_constant = lipschitz_constant(x, groups, params.num_examples, sparsity_param)
    # print("lip constant", lip_constant)

    w0 = np.zeros(params.num_features)



    def shrinkage(arr):
        norm = np.linalg.norm(arr)
        if norm > 1.0:
            return arr/norm
        else:
            return arr
    #from lemma 1
    def gen_opt_alpha(beta, groups, sparsity_param):
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
        return np.array(alpha_gs)

    # print("opt alpha", gen_opt_alpha(b, groups, sparsity_param))


    #Equation 9
    def f_squiggle_gradient(x, y, b, groups, sparsity_param):
        opt_alpha = gen_opt_alpha(b, groups, sparsity_param)
        term2 = np.matmul(np.transpose(c), opt_alpha) #TODO check dimensions
        term1 = np.matmul(np.transpose(x), ((np.matmul(x, b)) - y))
        return term1 + term2


    def testConvergence(t, betas_t):
        if (t<2):
            return False
        change_arr = np.absolute(np.subtract(betas_t[t], betas_t[t - 1]))
        change_in_beta = np.sum(change_arr)
        prior_change_in_beta = np.sum(np.absolute(np.subtract(betas_t[t-1], betas_t[t - 2])))
        max_change = 0.0
        max_change_index = 0
        beta = betas_t[t]
        for i in range(beta.shape[0]):
            if change_arr[i] > max_change:
                max_change = change_arr[i]
                max_change_index = i


        if (t % 1 == 0):
            print("t", t, "change", change_in_beta, "max_c", max_change, "i", max_change_index, "mc_val", betas_t[t][max_change_index])
        if abs(change_in_beta - prior_change_in_beta) < 0.00001: #and change_in_beta > 0.1:
            print("!!! Convergence due to 2nd-degree change in beta")
            return True
        if change_in_beta < 0.0001:
            print("!!! Convergence due to 1st-degree change in beta")
            return True
        return False

    # Algorithm

    gradient_t = []
    weights_t = []
    beta_t = []
    z_t = []
    weights_t.append(np.zeros(params.num_features))

    start = datetime.datetime.now()
    for t in range(1000):
        #step 1
        gradient_t.append(f_squiggle_gradient(x, y, weights_t[t], groups, sparsity_param))
        #step 2
        #print("gradient/lip_constant", gradient, lip_constant, gradient/lip_constant)
        beta_t.append(weights_t[t] - (gradient_t[t]/lip_constant))
        #step 3
        z = 0 - ((t + 1) * gradient_t[t] / 2) / lip_constant
        if (t > 0): z += z_t[t-1]
        z_t.append(z)
        #print("z/lip_constant", z, lip_constant, z / lip_constant)
        #step 4
        weights = ((t+1)*beta_t[t] / (t+3)) + (2*beta_t[t] / (t+3))
        weights_t.append(weights)
        if testConvergence(t, beta_t):
            break


    end = datetime.datetime.now()
    tim = end - start
    runtime_ms = tim.seconds * 1000 + tim.microseconds/1000

    learned_beta = beta_t[t]
    # print("learned_beta", learned_beta)
    # print("convergence",convergence)
    # print("lip constant", lip_constant)
    return learned_beta, runtime_ms, t

# (b, runtime) = learn(x, y, groups)
#
# print("runtime", runtime)
# print("b", b)


def test(learned_beta, real_beta, params):
    test_x = np.random.normal(0, 1, (params.num_examples, params.num_features))
    actual_y = np.matmul(test_x, real_beta)  # TODO add epsilon
    predicted_y = np.matmul(test_x, learned_beta)
    errors = np.subtract(actual_y, predicted_y)
    avg_error = np.sum(np.absolute(errors)) / params.num_examples #TODO use reduce and l2
    # print("avg_error", avg_error)
    return avg_error










