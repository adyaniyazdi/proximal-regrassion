import matplotlib.pyplot as plt
import numpy as np
import experiments as ex
import math
import proxregression as pr






REPS_PER_RESULT = 2
param = []
group_err = []
group_runtime = []
feature_err = []
feature_runtime = []
def scan(params:pr.Parameters, interval, intervals):
    print("Reps:", REPS_PER_RESULT)
    for i in range(intervals):
        print("   PARAM:", params.sparsity_param)
        param.append(params.sparsity_param)

        results = ex.run_experiment_set(params, ex.gen_sparse_groups_beta, REPS_PER_RESULT)
        (runtime, cycles, err, convergence, oscillation) = results
        group_err.append(results[2])
        group_runtime.append(results[0])

        results = ex.run_experiment_set(params, ex.gen_sparse_alternating_beta, REPS_PER_RESULT)
        feature_err.append(results[2])
        feature_runtime.append(results[0])

        params.sparsity_param *= interval



params = pr.Parameters()
params.num_examples = 500
params.num_groups = 50
params.group_size = 10
params.group_overlap = 3
params.sparsity_param = 640

params.desired_accuracy = 10 #1000
params.convergence_limit = 0.001/params.desired_accuracy
params.noise_variance = 0.0 #0.1 # 0.0
params.time_limit = 5000
scan(params, interval=0.25, intervals=14)
# scan(4, .25, True,
#      [500, 50, 10, 3, 640,
#       10, 0.000001, 0.00, 5000])

# # Data for plotting
# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2 * np.pi * t)
# t = np.array(param)
# s = np.array(group_err)
#
# # Note that using plt.subplots below is equivalent to using
# # fig = plt.figure and then ax = fig.add_subplot(111)
# fig, ax = plt.subplots()
# ax.plot(t, s)
#
# ax.set(xlabel='time (s)', ylabel='voltage (mV)',
#        title='About as simple as it gets, folks')
# ax.grid()
#
# fig.savefig("test.png")
# plt.show()
#
# # Make some fake data.
# a = b = np.arange(0, 3, .02)
# c = np.exp(a)
# d = c[::-1]

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.loglog(param, group_err, 'k', label='Group Error', color='blue')
ax.loglog(param, feature_err, 'k', label='Feature Error ', color ='red')

ax2 = ax.twinx()
ax2.loglog(param, group_runtime, 'k--', label = 'Group Runtime', color='blue')
ax2.loglog(param, feature_runtime, 'k--', label = 'Feature Runtime', color ='red')


ax.legend(loc='upper left', shadow=True, fontsize='large')
ax2.legend(loc='upper right', shadow=True, fontsize='large')

# Put a nicer background color on the legend.
#legend.get_frame().set_facecolor('#00FFCC')

fig.savefig("result.png")
plt.show()








