'''A tool for running repeated experiments for different values of a parameter'''

import batch_experiments as ex
import proxregression as pr
import testing as tst



def params_from_arr(arr):
    params = pr.Parameters()
    params.num_examples = arr[0]  # N
    params.num_groups = arr[1]
    params.group_size = arr[2]
    params.group_overlap = arr[3]
    params.sparsity_param = arr[4]

    params.desired_accuracy = arr[5]  # 1000
    params.training_feature_sparsity = arr[6]
    params.noise_variance = arr[7]
    params.time_limit = arr[8]
    return params


reps = 10
num_intervals = 10
def scan(param_index, interval, geometric, f_gen_beta, param_arr):
    print("Reps:", reps, "Starting parameters:", param_arr)
    for i in range(num_intervals):
        params = params_from_arr(param_arr)
        results = ex.run_experiment_set(params, f_gen_beta, reps)
        (runtime, cycles, err, convergence, oscillation) = results
        print("   PARAM[{0}]:".format(param_index), param_arr[param_index])
        if geometric:
            param_arr[param_index] *= interval
        else:
            param_arr[param_index] += interval

if __name__ == "__main__":
    # linear_scan(1, 200, [500, 10, 10, 3, 0.1,
    #                     1, 0.0001, 0.1, 1000])

    # ex.banner("Half Support")
    # scan(4, .1, True, ex.gen_half_support_beta,
    #      [500, 50, 10, 3, 1,
    #       10, 0.000001, 0.001, 5000])
    # ex.banner("Alternating Suport")
    # scan(4, .1, True, ex.gen_alternating_support_beta,
    #      [500, 50, 10, 3, 1,
    #       10, 0.000001, 0.001, 5000])
    # ex.banner("Alternating Support")
    # scan(6, .1, True, ex.gen_alternating_support_beta,
    #      [500, 50, 10, 3, 0.000001,
    #       1, 0.1, 0.1, 1000])
    # ex.banner("Uniform groups")
    # scan(4, 4, True, ex.gen_uniform_groups_beta,
    #      [500, 20, 10, 3, 0.0000001,
    #       1000, 0.0001, 0.1, 1000])
    # ex.banner("sparse groups")
    # scan(4, .5, True, ex.gen_sparse_groups_beta,
    #      [500, 100, 10, 3, 80,
    #       10, 0.000001, 0.00, 5000])
    ex.banner("sparse alternating")
    scan(1, 200, True, tst.continuous_structure_beta,
         [500, 10, 10, 3, 20,
          10, 2, 0.0, 5000])