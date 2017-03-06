import csv
import math
import time

import numpy as np
import itertools
'''
    Sigmoid function in order to map values into range of [0,1]
'''
def sigmoid_function(w, X):
    unbounded_prob_function = np.dot(X, w)
    return float(1) / (1 + math.e ** (-(unbounded_prob_function)))

'''
    Gradient calculation
'''
def gradient(w, X, y):
    diff = sigmoid_function(w, X) - y
    grad = np.dot(diff.T, X)
    return grad

'''
    Gaussian function used to sample prior
'''
def gaussian(mean, variance, w):
    return (float(1) / (variance * np.sqrt(2 * np.pi))) * np.exp(
        (-float(1) / (2 * variance) * ((w - mean).T.dot(w - mean))))

'''
    Maximum a posteriori estimation where we define our
    likelihood functions for epithelial and stromal states.
    Also include log of Gaussian prior
'''
def map_function(w, X, t):
    prior = np.log(gaussian(mean=0, variance=1, w=w))

    estimations = sigmoid_function(w, X)
    likelihood_epithelial = t * np.log(estimations)
    likelihood_stroma = (1 - t) * np.log(1 - estimations)

    values = -likelihood_epithelial - likelihood_stroma - prior
    values = np.array([0 if abs(el) == np.inf else el for el in values])
    return (np.nanmean(values))

'''
    Gradient descent algorithm in order to minimise the function we defined
    earlier. This is a numerical approximation of the actual function.
'''
def gradient_descent(w, X, y, learning_rate=.1, convergence_threshold=.001):
    cost = map_function(w, X, y)
    change_in_cost = 1
    start = time.clock()
    while change_in_cost > convergence_threshold:
        # If does not converge, give it 5s.
        end = time.clock()
        if (end - start) > 5:
            return w

        old_cost = cost
        w = w - (learning_rate * gradient(w, X, y))
        cost = map_function(w, X, y)
        change_in_cost = abs(old_cost - cost)
    return w

'''
    If the probability is above or equal to 0.5 we predict state 1 (stromal superpixel),
    0 (epithelial) otherwise.
'''
def make_predictions(w_optimal, X):
    predictions_probabilities = sigmoid_function(w_optimal, X)
    predictions = [1 if prob >= 0.5 else 0 for prob in predictions_probabilities]
    return predictions



'''
    Load and format all data
'''
# Set the directory of data
data_file_dir = 'epi_stroma_data.tsv'

with open(data_file_dir, 'rb') as tsv:
    column_labels = tsv.readline()
    column_labels = column_labels.split("\t")
    tsv = csv.reader(tsv, delimiter='\t')
    all_data = {}
    for row in tsv:
        for i in xrange(len(row)):
            if i not in all_data.keys():
                all_data[i] = [float(row[i])]
            else:
                all_data[i].append(float(row[i]))
data = np.array([np.array(x) for x in all_data.values()])
data = np.matrix(data)
data = data.T

# Set number of folds for CV
cross_validation_folds = 10


fold_size = int(data.shape[0] / float(cross_validation_folds))
predicted_values = []

means = {}
# Gives the combinations of parameter intervals
rand_params = list(itertools.combinations(range(1, data.shape[1]), 2))

'''
    The first column in data is the dependent variable we want to predict
    and the rest are the attributes we use to predict
'''
for params in rand_params:
    predicted_values = []
    for i in range(cross_validation_folds):
        testing_start = i * fold_size
        training_set_left = [0, testing_start]
        training_set_right = [testing_start + fold_size, data.shape[0]]

        if i != 0 and i != data.shape[0]:
            X = np.asarray(np.concatenate([data[training_set_left[0]:training_set_left[1], params[0]:params[1]],
                                           data[training_set_right[0]:training_set_right[1], params[0]:params[1]]]))
            y = np.concatenate([data[training_set_left[0]:training_set_left[1], 0],
                                data[training_set_right[0]:training_set_right[1], 0]]).astype(int).T
        elif i == 0:
            X = np.asarray(data[training_set_right[0]:training_set_right[1], params[0]:params[1]])
            y = np.asarray(data[training_set_right[0]:training_set_right[1], 0]).astype(int).T
        else:
            X = np.asarray(data[training_set_left[0]:training_set_left[1], params[0]:params[1]])
            y = np.asarray(data[training_set_left[0]:training_set_left[1], 0]).astype(int).T
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        y = np.asarray(y)[0] - 1
        w = np.zeros(X.shape[1])
        w_optimised = gradient_descent(w, X, y)

        evaluation_set = np.asarray(data[testing_start:testing_start + fold_size, params[0]:params[1]])
        evaluation_set = (evaluation_set - np.mean(evaluation_set, axis=0)) / np.std(evaluation_set, axis=0)

        predicted_y = make_predictions(w_optimised, evaluation_set)

        true_values = data[testing_start:testing_start + fold_size:, 0].astype(int).T
        true_values = np.asarray(true_values)[0] - 1

        # Check how many values are correct
        predicted_values.append(np.sum(true_values == predicted_y) / float(evaluation_set.shape[0]))

    print "These are cross validation results: ", predicted_values
    print "With mean: ", np.mean(predicted_values)
    means[(np.mean(predicted_values))] = params

key = max(means.keys())
print "Max mean accuracy from all the parameter combinations: ", key, means[key]
