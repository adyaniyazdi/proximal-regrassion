import random

import proxregression as pr
import numpy as np

verbose = True


def banner(msg):
    if verbose: print("=" * 75)
    print(msg)
    if verbose: print("=" * 65)


def run_experiment(params, generate_beta):
    # print("!!! Beginning experiment !!!")
    groups = pr.generate_groups(params)
    real_beta = generate_beta(params, groups)
    (x, y) = pr.generate_training_data(real_beta,params)


    (learned_beta, runtime, cycles, convergence_type) = pr.learn(x, y, groups, params)

    avg_error = pr.test(learned_beta, real_beta, params)
    return runtime, cycles, avg_error, convergence_type

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
# params.num_examples = 500 #N
# params.num_groups = 10
# params.group_size = 10
# params.group_overlap = 3
# params.sparsity_param = 0.1
# params.desired_accuracy = 0.01
# params.error_variance = 0.8
params.num_examples = 500 #N
params.num_groups = 80
params.group_size = 10
params.group_overlap = 3
params.sparsity_param = 0.1
params.desired_accuracy = 1000
params.convergence_limit = 0.0001
params.noise_variance = 0.01
params.time_limit = 15000


repetitions = 3
def run_experiment_set(params, gen_beta):
    total_runtime = 0.0
    total_cycles = 0
    total_avg_error = 0
    total_convergence = 0
    for i in range(repetitions):
        (runtime, cycles, avg_error, convergence_type) = run_experiment(params, gen_half_support_beta)
        if verbose: print("runtime:", int(runtime), "cycles:", cycles, "avg error:", round(avg_error, 3), "convergence:", convergence_type)
        total_runtime += runtime
        total_cycles += cycles
        total_avg_error += avg_error
        if convergence_type == 1:
            total_convergence += 1
    print("avg_runtime:", int(total_runtime/repetitions), "avg_cycles:", total_cycles/repetitions,
          "avg error:", round(total_avg_error/repetitions, 3),
          "convergence_rate:", total_convergence/repetitions)

banner("half support experiments:")
run_experiment_set(params, gen_half_support_beta)
# banner("one-group-zero experiments:")
# run_experiment_set(params, gen_one_group_zero_beta)
# banner("one-group-supported experiments:")
# run_experiment_set(params, gen_one_group_supported_beta)
banner("alternating support experiments:")
run_experiment_set(params, gen_alternating_support_beta)
# banner("uniform experiments:")
# run_experiment_set(params, gen_uniform_beta)

