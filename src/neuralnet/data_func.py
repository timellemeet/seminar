import numpy as np
import os

def vectorize_labels(labels):
    vector_labels = np.zeros((len(labels),10))
    for i in range(len(labels)):
        vector_labels[i, int(labels[i])] = 1
    return vector_labels


def k_fold(training_data,training_labels, k, i):
    observations = training_labels.shape[0]
    if observations%k != 0:
        raise Exception("Difficult division, make sure {}%{} is zero".format(observations,k))
    foldsize = int(observations/k)
    validation_data = training_data[(i-1)*foldsize:i*foldsize]
    validation_labels = training_labels[(i-1)*foldsize:i*foldsize]
    new_training_data = np.concatenate((training_data[i*foldsize:],training_data[:(i-1)*foldsize]))
    new_training_labels = np.concatenate((training_labels[i*foldsize:],training_labels[:(i-1)*foldsize]))
    return {
        "x_train":new_training_data,
        "y_train":new_training_labels,
        "x_val":validation_data,
        "y_val":validation_labels
    }


def import_data(size=60000, normalize=True, knearest = False):
    # import data
    dataset = np.load("../dataset.npz")
    training = dataset['arr_0'][:size]  # training_img
    labels = vectorize_labels(dataset['arr_2'][:size])  # training_labels
    test = dataset['arr_1']  # test_img
    original_test_labels = dataset['arr_3']  # test_labels
    test_labels = vectorize_labels(original_test_labels)
    np.random.seed(10)

    if normalize:
    # normalize data
        training /= 255
        test /= 255
    if knearest:
        return training, dataset['arr_2'][:size], test, original_test_labels, test_labels
    else:
        return training, labels, test, original_test_labels, test_labels
        
