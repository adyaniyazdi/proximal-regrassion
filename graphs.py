import matplotlib.pyplot as plt
import numpy as np
import batch_experiments as ex
import math
import proxregression as pr









x = 0/4
return

reps_per_result = 2
param = []
group_err = []
group_runtime = []
feature_err = []
feature_runtime = []
def scan(params:pr.Parameters, interval=0.5, intervals=12, reps_per_result=5):
    print("Reps:", reps_per_result)
    for i in range(intervals):
        print("   PARAM:", params.sparsity_param)
        param.append(params.sparsity_param)

        results = ex.run_experiment_set(params, ex.gen_sparse_groups_beta, reps_per_result)
        (runtime, cycles, err, convergence, oscillation) = results
        group_err.append(results[2])
        group_runtime.append(results[0])

        results = ex.run_experiment_set(params, ex.gen_sparse_alternating_beta, reps_per_result)
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
scan(params, interval=0.25, intervals=14, reps_per_result=2)
# scan(4, .25, True,
#      [500, 50, 10, 3, 640,
#       10, 0.000001, 0.00, 5000])

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.loglog(param, group_err, 'k', label='Group Error', color='blue')
ax.loglog(param, feature_err, 'k', label='Feature Error ', color ='red')

ax2 = ax.twinx()
ax2.loglog(param, group_runtime, 'k..', label = 'Group Runtime', color='blue')
ax2.loglog(param, feature_runtime, 'k..', label = 'Feature Runtime', color ='red')

ax.legend(loc='upper left', shadow=True, fontsize='large')
ax2.legend(loc='upper right', shadow=True, fontsize='large')

ax.set_xlabel("$\lambda$ (sparsity parameter)")
ax.set_ylabel(r"Avg error")#($MJ\,m^{-2}\,d^{-1}$)")
ax2.set_ylabel(r"Avg Runtime ($ms$)")
# ax2.set_ylim(0, 35)
# ax.set_ylim(-20,100)

# Put a nicer background color on the legend.
#legend.get_frame().set_facecolor('#00FFCC')

fig.savefig("result.png")
plt.show()








