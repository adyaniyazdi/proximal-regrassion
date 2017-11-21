import proxregression as pr
import testing as tst


def run_experiment(params:pr.Parameters, gen_beta):

    #Generate learning algorithm
    groups = tst.generate_groups(params)
    real_beta = gen_beta(params, groups)
    (x, y) = tst.generate_training_data(real_beta,params)

    #Learning algorithm
    (learned_beta, runtime, cycles, convergence_type) = pr.learn(x, y, groups, params)

    #Test accuracy
    avg_error = tst.test(learned_beta, real_beta, params)
    print("runtime:", int(runtime), "cycles:", cycles, "avg error:", round(avg_error, 3), "convergence:", convergence_type)

    return runtime, cycles, avg_error, convergence_type

params = pr.Parameters()
params.num_examples = 500
params.num_groups = 50
params.group_size = 10
params.group_overlap = 3
params.sparsity_param = 20
params.training_feature_sparsity = 10  # 1000
params.desired_accuracy = 100  # 1000
params.noise_variance = 0.5  # 0.1 # 0.0
params.time_limit = 5000

print("Continuous structure")
run_experiment(params, tst.continuous_structure_beta)
print("Seperate structure")
run_experiment(params, tst.seperate_structure_beta)
print("Unstructured control")
run_experiment(params, tst.unstructured_control_beta)
