import numpy as np
import math
import random


class Parameters:
    def __init__(self):
        self.num_features = None
        self.num_groups = None
        self.group_overlap = None
        self.group_size = None
        self.num_examples = None
params = Parameters()
params.num_features = 6
params.num_examples = 500 #N
params.num_groups = 10
params.group_size = 10
params.group_overlap = 3
print("nf", params.num_features)


#lambda
def generate_sample_data(params):

    # num_groups = 10
    # group_size = 10
    # group_overlap = 3
    groups=[]

    for i in range(params.num_groups):
        group_start_index = i * (params.group_size - params.group_overlap)
        groups.append(list(range(group_start_index, group_start_index + params.group_size)))

    print("groups: ", groups)
    #print(list(range(7, 17)))





    params.num_features = params.group_overlap + (params.num_groups * (params.group_size - params.group_overlap)) #J
    print("j=", params.num_features)


    x = np.random.normal(0, 1, (params.num_examples, params.num_features))
    #b = np.random.normal(0, 1, num_features)

    b = np.zeros(params.num_features)
    for group in groups:
        r = random.gauss(0,1)
        for member in group:
            b[member] +=r
    print("org b", b)
    for i in range(params.num_features//2, params.num_features):
        b[i] = 0.0
    y = np.matmul(x,b) #TODO add epsilon
    # print("x", x)
    # print("b", b)
    # print("y", y)

    return x, y, b, groups

(x, y, b, groups) = generate_sample_data(params)

print("x", x)
print("b", b)
print("y", y)
print("groups", groups)
##Initialization

sparsity_param = 0.1
desired_accuracy = 0.01
mu = desired_accuracy / groups.__len__()

def learn(x, y, groups):
    # From top of page 4
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

    print("abs_gamma", abs_gamma(groups, params.num_examples, sparsity_param=sparsity_param))


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
    print("lip constant", )

    w0 = np.zeros(params.num_features)

    # Algorithm steps

    # def gen_alpha(beta, groups):
    #     alpha_gs = []
    #     for group in groups:
    #         beta_g = []
    #         for j in group:
    #             beta_g.append(beta[j])
    #         #beta_gs.append(beta_g)
    #         norm = np.linalg.norm(np.array(beta_g))
    #         #print("norm", norm)
    #         alpha_g = []
    #         for b in beta_g:
    #             a =  b/norm if norm > 0 else 0
    #             alpha_g.append(a)
    #         print("b/a", beta_g, alpha_g)
    #         #print("norm alpha", np.linalg.norm(np.array(alpha_g)))
    #         alpha_gs += alpha_g
    #     return alpha_gs
    #
    # print("alpha",gen_alpha(b, groups))

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

    print("opt alpha", gen_opt_alpha(b, groups, sparsity_param))


    #Equation 9
    def f_squiggle_gradient(x, y, b, groups, sparsity_param):
        opt_alpha = gen_opt_alpha(b, groups, sparsity_param)
        term2 = np.matmul(np.transpose(c), opt_alpha) #TODO check dimensions
        term1 = np.matmul(np.transpose(x), ((np.matmul(x, b)) - y))
        return term1 + term2

    # Algorithm

    gradient_t = []
    weights_t = []
    beta_t = []
    z_t = []
    weights_t.append(np.zeros(params.num_features))

    for t in range(10000):
        #step 1
        gradient_t.append(f_squiggle_gradient(x, y, weights_t[t], groups, sparsity_param))
        #step 2
        #print("gradient/lip_constant", gradient, lip_constant, gradient/lip_constant)
        beta_t.append(weights_t[t] - (gradient_t[t]/lip_constant))
        #step 3
        # z = 0
        # for i in range(t):
        #     z += (i+1) * gradient_t[i] / 2  #TODO why does it say "i" instead of "t" in paper
        z = 0 - ((t + 1) * gradient_t[t] / 2) / lip_constant
        if (t > 0): z += z_t[t-1]
        z_t.append(z)
        #print("z/lip_constant", z, lip_constant, z / lip_constant)
        #step 4
        weights = ((t+1)*beta_t[t] / (t+3)) + (2*beta_t[t] / (t+3))
        weights_t.append(weights)
        convergence = np.sum(np.absolute(np.subtract(beta_t[t], beta_t[t-1])))
        #convergence = abs(beta_t[t][0] - beta_t[t-1][0])
        print("t / convergence", t, convergence, beta_t[t][0])
        if convergence < 0.00001 and t>1:
            break
    print("b_t", beta_t[t])
    print("convergence",convergence)
    print("actual b", b)
    print("lip constant", lip_constant)
    return b



















