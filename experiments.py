import random

import proxregression as pr
import numpy as np

print("import done")

def banner(msg):
    print("=" * 75)
    print(msg)
    print("=" * 65)


def run_experiment(params, generate_beta):
    groups = pr.generate_groups(params)
    real_beta = generate_beta(params, groups)
    (x, y) = pr.generate_training_data(real_beta,params)

    sparsity_param = 0.1
    desired_accuracy = 0.01
    (learned_beta, runtime, cycles) = pr.learn(x, y, groups, params, sparsity_param, desired_accuracy)

    avg_error = pr.test(learned_beta, real_beta, params)
    print("runtime",runtime, "cycles", cycles, "avg error", avg_error)

def gen_half_support_beta(params, groups):
    real_beta = np.random.normal(0, 1, params.num_features)
    for i in range(params.num_features // 2, params.num_features):
        real_beta[i] = 0.0
    return real_beta

def gen_alternating_support_beta(params, groups):
    real_beta = np.random.normal(0, 1, params.num_features)
    for i in range(params.num_features//2):
        real_beta[i*2] = 0.0
    return real_beta

def gen_uniform_beta(params, groups):
    return np.random.normal(0, 1, params.num_features)

def gen_one_group_zero_beta(params, groups):
    real_beta = np.random.normal(0, 1, params.num_features)
    for j in groups[0]:
        real_beta[j] = 0.0
    return real_beta

def gen_one_group_supported_beta(params, groups):
    real_beta = np.zeros(params.num_features)
    for j in groups[0]:
        real_beta[j] = random.gauss(0, 1)
    return real_beta


params = pr.Parameters()
params.num_examples = 500 #N
params.num_groups = 10
params.group_size = 10
params.group_overlap = 3

# (x, y, real_beta, groups) = pr.generate_sample_data(params)

repetitions = 2
banner("half support experiments:")
for i in range(repetitions):
    run_experiment(params, gen_half_support_beta)
banner("alternating support experiments:")
for i in range(repetitions):
    run_experiment(params, gen_alternating_support_beta)
# banner("uniform experiments:")
# for i in range(repetitions):
#     run_experiment(params, gen_uniform_beta)
banner("one-group-zero experiments:")
for i in range(repetitions):
    run_experiment(params, gen_one_group_zero_beta)
# banner("one-group-supported experiments:")
# for i in range(repetitions):
#     run_experiment(params, gen_one_group_supported_beta)

