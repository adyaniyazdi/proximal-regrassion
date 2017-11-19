import experiments as ex
import proxregression as pr

# params = pr.Parameters()
# params.num_examples = 500
# params.num_groups = 10
# params.group_size = 10
# params.group_overlap = 3
# params.sparsity_param = 0.1

# params.desired_accuracy = .01 #1000
# params.convergence_limit = 0.0001
# params.noise_variance = 1.0
# params.time_limit = 1000

# param_arr = [500, 10, 10, 3, 0.1,
#              1000, 0.0001, 0.1, 1000]

def params_from_arr(arr):
    params = pr.Parameters()
    params.num_examples = arr[0]  # N
    params.num_groups = arr[1]
    params.group_size = arr[2]
    params.group_overlap = arr[3]
    params.sparsity_param = arr[4]

    params.desired_accuracy = arr[5]  # 1000
    params.convergence_limit = 0.001/arr[5]#arr[6]
    params.noise_variance = arr[7]
    params.time_limit = arr[8]
    return params

gen_beta = ex.gen_half_support_beta
reps = 5


def linear_optimize(param_index, bracket, f_test, param_arr):
    stop_bracket = bracket/32
    while bracket > stop_bracket:
        params = params_from_arr(param_arr)
        results = ex.run_experiment_set(params, gen_beta, reps)
        (runtime, cycles, err, convergence, oscillation) = results
        test = f_test(results)
        if test:
            param_arr[param_index] -= bracket
        else:
            param_arr[param_index] += bracket
        print("PARAM:", param_arr[param_index])
        bracket /= 2

def geo_minimize(param_index, factor, f_test, param_arr):
    for i in range(50):
        params = params_from_arr(param_arr)
        results = ex.run_experiment_set(params, gen_beta, reps)
        (runtime, cycles, err, convergence, oscillation) = results
        test = f_test(results)
        print("PARAM:", param_arr[param_index])
        if test:
            param_arr[param_index] /= factor
        else:
            print("Minimal value of param:", param_arr[param_index]*2)
            break

def convergence_test(results):
    return results[3] > .75

# optimise sparsity param
#linear_optimize(5, 500, lambda results: results[3] > .75)

# geo_minimize(5, 2, convergence_test, [500, 10, 10, 3, 0.1,
#              1000, 0.0001, 0.1, 1000])

num_intervals = 8
def scan(param_index, interval, geometric, f_gen_beta, param_arr):
    print("Reps:", reps, "Starting parameters:", param_arr)
    for i in range(8):
        params = params_from_arr(param_arr)
        results = ex.run_experiment_set(params, f_gen_beta, reps)
        (runtime, cycles, err, convergence, oscillation) = results
        print("   PARAM[{0}]:".format(param_index), param_arr[param_index])
        if geometric:
            param_arr[param_index] *= interval
        else:
            param_arr[param_index] += interval

# linear_scan(1, 200, [500, 10, 10, 3, 0.1,
#                     1, 0.0001, 0.1, 1000])

ex.banner("Half Support")
scan(5, .1, True, ex.gen_half_support_beta,
     [500, 50, 10, 3, 0.000001,
      10000, 0.000001, 0.001, 5000])
# ex.banner("Alternating Support")
# scan(6, .1, True, ex.gen_alternating_support_beta,
#      [500, 50, 10, 3, 0.000001,
#       1, 0.1, 0.1, 1000])
# ex.banner("Uniform groups")
# scan(4, 4, True, ex.gen_uniform_groups_beta,
#      [500, 20, 10, 3, 0.0000001,
#       1000, 0.0001, 0.1, 1000])