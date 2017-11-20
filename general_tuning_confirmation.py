import random
import math
import proxregression as pr
import numpy as np
import tools

verbose = False

def experiment_with_fixed_params(params:pr.Parameters, gen_beta):
    # print("!!! Beginning experiment !!!")
    groups = pr.generate_groups(params)
    real_beta = gen_beta(params, groups)
    (x, y) = pr.generate_training_data(real_beta,params)

    # Re-run experiment with optimal sparsity parameter
    (learned_beta, runtime, cycles, convergence_type) = pr.learn(x, y, groups, params)
    avg_error = pr.test(learned_beta, real_beta, params)
    print("Applying sp", params.sparsity_param, "runtime:", int(runtime), "cycles:", cycles, "avg error:", round(avg_error, 3), "convergence:", convergence_type)

    return runtime, cycles, avg_error, convergence_type

def optimize_sp_for_fixed_beta(params:pr.Parameters, groups, real_beta):
    # print("!!! Beginning experiment !!!")
    (x, y) = pr.generate_training_data(real_beta,params)

    def f(log_sparsity_param):
        params.sparsity_param = math.pow(2, log_sparsity_param)
        (learned_beta, runtime, cycles, convergence_type) = pr.learn(x, y, groups, params)
        avg_error = pr.test(learned_beta, real_beta, params)
        return avg_error

    #Optimize sparsity parameter several times (stochastic optimization)
    opt_log_params = []
    for i in range(5):
        best_log_param = tools.minimize(f, 0.0, 12.0, 0.3)
        opt_log_params.append(best_log_param)
        print("fixed beta optimum param", math.pow(2, best_log_param))
    # print("sparsity params for fixed beta:", opt_log_params)
    opt_sparsity_param = math.pow(2, np.mean(opt_log_params))


    print("TRAINING SPARSITY:", params.training_feature_sparsity)
    print("   Optimium sparsity parameter: ", opt_sparsity_param, "= 2^", np.mean(opt_log_params))

    # Re-run experiment with optimal sparsity parameter
    params.sparsity_param = opt_sparsity_param
    (learned_beta, runtime, cycles, convergence_type) = pr.learn(x, y, groups, params)
    avg_error = pr.test(learned_beta, real_beta, params)
    print("With optimized sp", params.sparsity_param, "runtime:", int(runtime), "cycles:", cycles, "avg error:",
          round(avg_error, 3), "convergence:", convergence_type)

    return runtime, cycles, avg_error, convergence_type, opt_sparsity_param

def run_experiment_set(params, gen_beta, repetitions):
    total_runtime = 0.0
    total_cycles = 0
    total_avg_error = 0
    total_convergence = 0
    sparsity_params = []
    for i in range(repetitions):
        groups = pr.generate_groups(params)
        real_beta = gen_beta(params, groups)

        (runtime, cycles, avg_error, convergence_type, opt_sparsity_param) = optimize_sp_for_fixed_beta(params, groups, real_beta)
        if verbose: print("runtime:", int(runtime), "cycles:", cycles, "avg error:", round(avg_error, 3), "convergence:", convergence_type)
        total_runtime += runtime
        total_cycles += cycles
        total_avg_error += avg_error
        if convergence_type == pr.CONV_1ST_DEG:
            total_convergence += 1
        sparsity_params.append(opt_sparsity_param)
    avg_runtime = int(total_runtime/repetitions)
    avg_cycles = total_cycles/repetitions
    avg_error = round(total_avg_error/repetitions, 3)
    convergence_rate = total_convergence/repetitions

    print("SUMMARY: avg_runtime:", avg_runtime, "avg_cycles:", avg_cycles,
          "avg error:", avg_error, "convergence_rate:", convergence_rate, "sparsity param:", sparsity_params)
    params.sparsity_param = np.mean(sparsity_params)
    for i in range(5):
        experiment_with_fixed_params(params, gen_beta)

    return avg_runtime, avg_cycles, avg_error, convergence_rate, np.mean(sparsity_params)


if __name__ == "__main__":
    params = pr.Parameters()
    params.num_examples = 500
    params.num_groups = 50
    params.group_size = 10
    params.group_overlap = 3
    params.sparsity_param = 2048
    params.training_feature_sparsity = 10  # 1000
    params.desired_accuracy = 10  # 1000
    params.convergence_limit = 0.001 / params.desired_accuracy
    params.noise_variance = 0.1  # 0.1 # 0.0
    params.time_limit = 5000

    reps = 5
    tools.banner("continuous experiments:")
    run_experiment_set(params, tools.continuous_structure_beta, reps)


