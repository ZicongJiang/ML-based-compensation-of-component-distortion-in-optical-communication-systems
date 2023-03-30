import random

import numpy as np


def gaussian(x, variance, mean):
    output = 1 / (np.sqrt(2 * np.pi * variance)) * np.exp(-(x - mean) ** 2 / (2 * variance))
    return output


def gaussian_mixture(x, n_components):
    np.random.seed(10)
    variance = np.random.randint(1, 5, n_components)
    mean = np.random.randint(-5, 5, n_components)
    gmm = np.zeros([n_components, len(x)])
    for i in range(n_components):
        gmm[i] = 1 / (np.sqrt(2 * np.pi * variance[i])) * np.exp(-(x - mean[i]) ** 2 / (2 * variance[i] ** 2))
    gmm_all = gmm.sum(0)
    return gmm, gmm_all
