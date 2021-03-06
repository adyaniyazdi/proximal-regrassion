import proxregression as pr
import matplotlib.pyplot as plt
import numpy as np
import testing as tst

def experiment_with_fixed_params(params:pr.Parameters, gen_beta):
    # print("!!! Beginning experiment !!!")
    groups = tst.generate_groups(params)
    real_beta = gen_beta(params, groups)
    (x, y) = tst.generate_training_data(real_beta,params)

    # Re-run experiment with optimal sparsity parameter
    (learned_beta, runtime, cycles, convergence_type) = pr.learn(x, y, groups, params)
    avg_error = tst.test(learned_beta, real_beta, params)
    print( "Number of Groups:", params.num_groups ,"runtime:", int(runtime), "cycles:", cycles, "avg error:", round(avg_error, 3), "convergence:", convergence_type)

    return runtime, cycles, avg_error, convergence_type



if __name__ == "__main__":
    params = pr.Parameters()
    params.num_groups = 50
    time =[]
    ngroups =[]
    for x in range(15):
        params.num_examples = 5000
        params.num_groups += 50
        params.group_size = 10
        params.group_overlap = 3
        params.sparsity_param = 20
        params.training_feature_sparsity = 10  # 1000
        params.desired_accuracy = 10  # 1000
        params.convergence_limit = 0.001 / params.desired_accuracy
        params.noise_variance = 0.1  # 0.1 # 0.0
        params.time_limit = 40000

        # tools.banner("...:")
        (runtime, cycles, avg_error, convergence_type) = experiment_with_fixed_params(params, tst.continuous_structure_beta)
        time.append(runtime/1000)
        ngroups.append(params.num_groups)
    # tools.banner("...:")
    # experiment_with_fixed_params(params, tools.seperate_structure_beta)
    # tools.banner("...:")
    # experiment_with_fixed_params(params, tools.unstructured_control_beta)

# Data for plotting
# t = np.arange(0.0, 2.0, 0.01)
# s = 10+ np.sin(2 * np.pi * t)
# x = [1,2,3,4]
# y = [1, 1, 2, 2]

# Note that using plt.subplots below is equivalent to using
# fig = plt.figure and then ax = fig.add_subplot(111)
fig, ax = plt.subplots()
ax.semilogy(ngroups, time)

ax.set(xlabel='No. Of Groups', ylabel='Time (seconds)',
       title='Scalability for Prox-Grad')
ax.grid()

fig.savefig("numberOfGroups.png")
plt.show()