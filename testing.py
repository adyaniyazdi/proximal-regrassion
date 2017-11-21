import numpy as np
import random

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


def generate_groups(params):
    groups=[]

    for i in range(params.num_groups):
        group_start_index = i * (params.group_size - params.group_overlap)
        groups.append(list(range(group_start_index, group_start_index + params.group_size)))

    # print("groups: ", groups)
    params.num_features = params.group_overlap + (params.num_groups * (params.group_size - params.group_overlap)) #J
    # print("num_features=", params.num_features)
    return groups

def generate_training_data(beta, params):
    #error_variance = 0.8 #TODO tune
    x = np.random.normal(0, 1, (params.num_examples, params.num_features))
    # print("beta",beta)
    y = np.matmul(x,beta) + np.random.normal(0, params.noise_variance, params.num_examples)

    return x, y


def test(learned_beta, real_beta, params):
    test_x = np.random.normal(0, 1, (params.num_examples, params.num_features))
    actual_y = np.matmul(test_x, real_beta)  # TODO add epsilon
    predicted_y = np.matmul(test_x, learned_beta)
    errors = np.subtract(actual_y, predicted_y)
    avg_error = np.mean(np.absolute(errors)) #TODO use reduce and l2
    # print("avg_error", avg_error)
    return avg_error