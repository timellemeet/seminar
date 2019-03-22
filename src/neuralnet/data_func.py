import numpy as np


def vectorize_labels(labels):
    vector_labels = np.zeros((len(labels),10))
    for i in range(len(labels)):
        vector_labels[i, int(labels[i])] = 1
    return vector_labels


def k_fold(training_data,training_labels, k, n):
    observations,__ = training_labels.shape
    if observations%k != 0:
        raise Exception("Difficult division, make sure {}%{} is zero".format(observations,k))
    foldsize = int(observations/k)
    validation_data = training_data[(n-1)*foldsize:n*foldsize]
    validation_labels = training_labels[(n-1)*foldsize:n*foldsize]
    new_training_data = np.concatenate((training_data[n*foldsize:],training_data[:(n-1)*foldsize]))
    new_training_labels = np.concatenate((training_labels[n*foldsize:],training_labels[:(n-1)*foldsize]))
    return new_training_data, new_training_labels, validation_data, validation_labels

