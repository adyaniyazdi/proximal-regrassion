import numpy as np
import math


desired_accuracy = 0.01
num_groups = 5
group_size = 5
group_overlap = 2
groups=[]

for i in range(num_groups):
    group_start_index = i * (group_size - group_overlap)
    groups.append(list(range(group_start_index, group_start_index + group_size)))

print("groups: ", groups)
#print(list(range(7, 17)))


num_examples = 12
num_features = group_overlap + (num_groups * (group_size - group_overlap))
print("j=", num_features)

x = np.arange(num_examples * num_features, dtype=np.float64).reshape((num_examples, num_features));
x0 = np.arange(num_features, dtype=np.float64).reshape((num_features, 1));
b = np.ones(num_features, dtype=np.float64)#.reshape((1, j));
y = np.arange(num_examples, dtype=np.float64).reshape((num_examples, 1));

y[1] = 5.0
b[2] = 1
x = np.random.normal(0, 1, (num_examples, num_features))
b = np.random.normal(0, 1, num_features)
for i in range(num_features//2, num_features):
    b[i] = 0.0
y = np.matmul(x,b) #TODO add epsilon
print("x", x)
print("b", b)
print("y", y)

##Initialization

# From top of page 4
def group_weight(group):
    return math.sqrt(group.__len__())

#From equation 4
def build_c(groups, sparsity_param, j):
    c_height = 0
    for group in groups:
        c_height += group.__len__()
    c = np.zeros((c_height, j))
    i = 0
    for group in groups:
        w = group_weight(group)
        for member in group:
            c[i, member] = sparsity_param * w
            i += 1
    return c

c = build_c(groups, 5.0, num_features)
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

print("abs_gamma", abs_gamma(groups, num_examples, sparsity_param=3.0))


#From equation 10
def lipschitz_constant(x, groups, num_examples, sparsity_param):
    xt_x = np.matmul(np.transpose(x),x)
    eigens = np.linalg.eig(xt_x)[0].tolist()
    print("eigens", eigens)
    max_eigen = max(eigens)
    #print("max_eigen", max_eigen)
    abs_g = abs_gamma(groups, num_examples, sparsity_param)
    mu = desired_accuracy / groups.__len__()
    return max_eigen + (abs_g / mu)

print("lip constant", lipschitz_constant(x, groups, num_examples, 3.0))

w0 = np.zeros(num_features)

# Algorithm steps

def gen_alpha(beta, groups):
    alpha_gs = []
    for group in groups:
        beta_g = []
        for j in group:
            beta_g.append(beta[j])
        #beta_gs.append(beta_g)
        norm = np.linalg.norm(np.array(beta_g))
        #print("norm", norm)
        alpha_g = []
        for b in beta_g:
            a =  b/norm if norm > 0 else 0
            alpha_g.append(a)
        print("b/a", beta_g, alpha_g)
        #print("norm alpha", np.linalg.norm(np.array(alpha_g)))
        alpha_gs += alpha_g
    return alpha_gs

print("alpha",gen_alpha(b, groups))








# Algorithm




















