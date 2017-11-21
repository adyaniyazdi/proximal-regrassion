import random

import testing as tst
import proxregression as pr
import numpy as np

verbose = True


def banner(msg):
    print("=" * 75)
    print(msg)
    print("=" * 65)


def run_experiment(params, generate_beta):
    groups = tst.generate_groups(params)
    real_beta = generate_beta(params, groups)
    (x, y) = tst.generate_training_data(real_beta,params)

    (learned_beta, runtime, cycles, convergence_type) = pr.learn(x, y, groups, params)

    avg_error = tst.test(learned_beta, real_beta, params)
    return runtime, cycles, avg_error, convergence_type


def run_experiment_set(params, gen_beta, repetitions):
    total_runtime = 0.0
    total_cycles = 0
    total_avg_error = 0
    total_convergence = 0
    total_oscilations = 0
    for i in range(repetitions):
        (runtime, cycles, avg_error, convergence_type) = run_experiment(params, gen_beta)
        if verbose: print("runtime:", int(runtime), "cycles:", cycles, "avg error:", round(avg_error, 3), "convergence:", convergence_type)
        total_runtime += runtime
        total_cycles += cycles
        total_avg_error += avg_error
        if convergence_type == pr.CONV_1ST_DEG:
            total_convergence += 1
        if convergence_type == pr.CONV_2ND_DEG:
            total_oscilations += 1
    avg_runtime = int(total_runtime/repetitions)
    avg_cycles = total_cycles/repetitions
    avg_error = round(total_avg_error/repetitions, 3)
    convergence_rate = total_convergence/repetitions
    oscillation_rate = total_oscilations/repetitions

    print("avg_runtime:", avg_runtime, "avg_cycles:", avg_cycles,
          "avg error:", avg_error, "convergence_rate:", convergence_rate, "oscillation_rate:", oscillation_rate)
    return avg_runtime, avg_cycles, avg_error, convergence_rate, oscillation_rate



params = pr.Parameters()

params.num_examples = 500 #N
params.num_groups = 10
params.group_size = 10
params.group_overlap = 3
params.sparsity_param = 0.1
params.desired_accuracy = 1000
params.training_feature_sparsity = 2
params.noise_variance = 1.0
params.time_limit = 2000

reps = 5
if __name__ == "__main__":
    banner("Continuous structure experiments:")
    run_experiment_set(params, tst.continuous_structure_beta, reps)


