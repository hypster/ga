import numpy as np


def softmax(f, T=1):
    '''
    :param f: the utility for each individual
    :param T: the temperature, hyperparameter to control the hardness of prob
    :return: the prob distribution
    '''
    # using shift invariant property here to avoid under/overflow: softmax(x) = softmax(x+c)
    _f = f / T
    _f -= max(_f)

    return np.exp(_f) / np.sum(np.exp(_f))