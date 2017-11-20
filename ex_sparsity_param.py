import random
import matplotlib.pyplot as plt
import numpy as np
import experiments as ex
import math
import proxregression as pr


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


def scan_parameter(params:pr.Parameters, interval=0.5, intervals=12, reps_per_result=5):
    print("Reps:", reps_per_result)
    param = []
    continuous_err = []
    continuous_runtime = []
    seperate_err = []
    seperate_runtime = []
    control_err = []
    control_runtime = []
    for i in range(intervals):
        print("   PARAM:", params.sparsity_param)
        param.append(params.sparsity_param)

        results = ex.run_experiment_set(params, continuous_structure_beta, reps_per_result)
        (runtime, cycles, err, convergence, oscillation) = results
        continuous_err.append(results[2])
        continuous_runtime.append(results[0])

        results = ex.run_experiment_set(params, seperate_structure_beta, reps_per_result)
        seperate_err.append(results[2])
        seperate_runtime.append(results[0])

        results = ex.run_experiment_set(params, unstructured_control_beta, reps_per_result)
        control_err.append(results[2])
        control_runtime.append(results[0])

        params.sparsity_param *= interval
    fig, ax = plt.subplots()
    ax.loglog(param, continuous_err, 'k', label='Continuous Error', color='blue')
    ax.loglog(param, seperate_err, 'k', label='Seperate Error', color='green')
    ax.loglog(param, control_err, 'k', label='Control Error ', color='red')

    ax2 = ax.twinx()
    ax2.loglog(param, continuous_runtime, 'k:', label='Continous Runtime', color='blue')
    ax2.loglog(param, seperate_runtime, 'k:', label='Seperate Runtime', color='green')
    ax2.loglog(param, control_runtime, 'k:', label='Control Runtime', color='red')

    # ax.legend(loc='upper left', shadow=True, fontsize='large')
    # ax2.legend(loc='upper right', shadow=True, fontsize='large')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax.grid()

    ax.set_xlabel("$\lambda$ (sparsity parameter)")
    ax.set_ylabel(r"Avg error")  # ($MJ\,m^{-2}\,d^{-1}$)")
    ax2.set_ylabel(r"Avg Runtime ($ms$)")
    # ax2.set_ylim(0, 35)
    # ax.set_ylim(-20,100)

    param_text = \
        'groups={0}, group_size={1}, overlap={2}, num_examples={3}\n'.format(
            params.num_groups, params.group_size, params.group_overlap, params.num_examples) \
        + 'training_sparsity={0}, training_noise={1}, epsilon={2}'.format(
            params.training_feature_sparsity, params.noise_variance, params.desired_accuracy)
    fig.suptitle(param_text, fontsize=10)  # , fontsize=14, fontweight='bold')

    fig.savefig("result.png")
    plt.show()


params = pr.Parameters()
params.num_examples = 500
params.num_groups = 50
params.group_size = 10
params.group_overlap = 3
params.sparsity_param = 2048
params.training_feature_sparsity = 2 #1000

params.desired_accuracy = 100 #1000
params.convergence_limit = 0.001/params.desired_accuracy
params.noise_variance = 0.1 #0.1 # 0.0
params.time_limit = 5000
scan_parameter(params, interval=1/4, intervals=14, reps_per_result=3)











