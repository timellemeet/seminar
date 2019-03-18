import numpy as np


def vectorize_labels(labels):
    vector_labels = np.zeros((len(labels),10))
    for i in range(len(labels)):
        vector_labels[i, int(labels[i])] = 1
    return vector_labels