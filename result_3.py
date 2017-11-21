import random
import matplotlib.pyplot as plt
import numpy as np
import batch_experiments as ex
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




params = pr.Parameters()
params.num_examples = 500
params.num_groups = 50
params.group_size = 10
params.group_overlap = 3
params.sparsity_param = 2048
params.training_feature_sparsity = 2 #1000

params.desired_accuracy = 10 #1000
params.convergence_limit = 0.001/params.desired_accuracy
params.noise_variance = 0.1 #0.1 # 0.0
params.time_limit = 5000

reps_per_result=5
print("Reps:", reps_per_result)
def err_for_log_sparsity_param(log_sparsity_param):
    params.sparsity_param = math.pow(2, log_sparsity_param)
    results = ex.run_experiment_set(params, seperate_structure_beta, reps_per_result)
    (runtime, cycles, err, convergence, oscillation) = results
    return err


def minimize(f, low_lim, up_lim, accuracy):

    while (up_lim - low_lim > accuracy):
        difference = up_lim - low_lim
        l1 = low_lim + (difference/3.0)
        l2 = low_lim + (2*difference/3.0)
        print("l1:", l1, "l2:", l2)
        if (f(l1) > f(l2)):
            low_lim = l1
        else:
            up_lim = l2
    return (up_lim + low_lim)/2

# opt_log_sparsity_param = minimize(err_for_log_sparsity_param, 0.0, 12.0, 2)
# opt_sparsity_param = math.pow(2, opt_log_sparsity_param)
# print("Optimium sparsity parameter: ", opt_sparsity_param, "= 2^",  opt_log_sparsity_param)

def scan(interval=math.sqrt(2), intervals=10):
    training_sparsity = []
    sparsity_param = []
    seperate_lamba = []
    for i in range(intervals):
        training_sparsity.append(params.training_feature_sparsity)

        opt_log_sparsity_param = minimize(err_for_log_sparsity_param, 0.0, 8.0, 0.3)
        opt_sparsity_param = math.pow(2, opt_log_sparsity_param)
        sparsity_param.append(opt_sparsity_param)

        print("TRAINING SPARSITY:", params.training_feature_sparsity)
        print("   Optimium sparsity parameter: ", opt_sparsity_param, "= 2^", opt_log_sparsity_param)
        print(training_sparsity)
        print(sparsity_param)

        params.training_feature_sparsity *= interval
    fig, ax = plt.subplots()
    ax.plot(training_sparsity, sparsity_param, 'k', label='Continuous Error', color='blue')

    # ax.legend(loc='upper left', shadow=True, fontsize='large')
    # ax2.legend(loc='upper right', shadow=True, fontsize='large')
    ax.legend(loc='upper left')

    ax.grid()

    ax.set_xlabel("Training set sparsity")
    ax.set_ylabel(r"Optimal $\lambda$ (sparsity parameter)")  # ($MJ\,m^{-2}\,d^{-1}$)")

    param_text = \
        'groups={0}, group_size={1}, overlap={2}, num_examples={3}\n'.format(
            params.num_groups, params.group_size, params.group_overlap, params.num_examples) \
        + 'training_noise={0}, epsilon={1}'.format(
            params.noise_variance, params.desired_accuracy)
    fig.suptitle(param_text, fontsize=10)  # , fontsize=14, fontweight='bold')

    fig.savefig("result.png")
    plt.show()

scan()