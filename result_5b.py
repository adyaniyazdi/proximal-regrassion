import proxregression as pr
import matplotlib.pyplot as plt
import numpy as np
import testing as tst
import math


def experiment_with_fixed_params(params:pr.Parameters, gen_beta):
    # print("!!! Beginning experiment !!!")
    groups = tst.generate_groups(params)
    real_beta = gen_beta(params, groups)
    (x, y) = tst.generate_training_data(real_beta,params)

    # Re-run experiment with optimal sparsity parameter
    (learned_beta, runtime, cycles, convergence_type) = pr.learn(x, y, groups, params)
    avg_error = tst.test(learned_beta, real_beta, params)
    #
    return runtime, cycles, avg_error, convergence_type


def scan_sparsity_for_fixed_num_examples(params:pr.Parameters, gen_beta):
    print("Number of examples:", params.num_examples)
    best_err = 99999
    best_rt = 999999999999
    best_sp = -1
    for i in range(6):
        params.sparsity_param = math.pow(2, i-1)
        (runtime, cycles, avg_error, convergence_type) = experiment_with_fixed_params(params, gen_beta)
        print("sp:", params.sparsity_param, "runtime:", int(runtime), "cycles:", cycles, "avg error:", round(avg_error, 3), "convergence:", convergence_type)
        if runtime<best_rt and avg_error < 1:   #avg_error<best_err:
            best_err = avg_error
            best_rt = runtime
            best_sp = params.sparsity_param
    return best_rt, best_sp


if __name__ == "__main__":
    params = pr.Parameters()
    params.num_examples = 0
    time =[]
    n =[]
    best_sps = []
    for x in range(20):
        params.num_examples += 1000
        params.num_groups = 50 #200 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        params.group_size = 10
        params.group_overlap = 3
        params.sparsity_param = 1#20
        params.training_feature_sparsity = 10  # 1000
        params.desired_accuracy = 10  # 1000
        params.convergence_limit = 0.001 / params.desired_accuracy
        params.noise_variance = 0.1  # 0.1 # 0.0
        params.time_limit = 8000


        #(runtime, cycles, avg_error, convergence_type) = experiment_with_fixed_params(params, tst.tools.seperate_structure_beta)
        (runtime, best_sp) = scan_sparsity_for_fixed_num_examples(params, tst.continuous_structure_beta)
        time.append(runtime/1000)
        best_sps.append(best_sp)
        n.append(params.num_examples)
    print(best_sps)
    print(time)
    print(n)


# Data for plotting
# t = np.arange(0.0, 2.0, 0.01)
# s = 10+ np.sin(2 * np.pi * t)
# x = [1,2,3,4]
# y = [1, 1, 2, 2]

# Note that using plt.subplots below is equivalent to using
# fig = plt.figure and then ax = fig.add_subplot(111)
fig, ax = plt.subplots()
ax.semilogy(n, time)
#ax.set_ylim(0.05, 2)
ax.set_ylim(0.009, .12)

ax.set(xlabel='No. Of Examples', ylabel='Time (seconds)',
       title='Scalability for Prox-Grad')
ax.grid()

fig.savefig("test.png")
plt.show()