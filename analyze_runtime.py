import proxregression as pr
import tools


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



if __name__ == "__main__":
    params = pr.Parameters()
    params.num_examples = 500
    params.num_groups = 50
    params.group_size = 10
    params.group_overlap = 3
    params.sparsity_param = 20
    params.training_feature_sparsity = 10  # 1000
    params.desired_accuracy = 10  # 1000
    params.convergence_limit = 0.001 / params.desired_accuracy
    params.noise_variance = 0.1  # 0.1 # 0.0
    params.time_limit = 5000

    reps = 5
    tools.banner("...:")
    experiment_with_fixed_params(params, tools.continuous_structure_beta)
    tools.banner("...:")
    experiment_with_fixed_params(params, tools.seperate_structure_beta)
    tools.banner("...:")
    experiment_with_fixed_params(params, tools.unstructured_control_beta)
