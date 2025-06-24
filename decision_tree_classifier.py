import numpy as np



def gini_impurity(labels):
    # when the set is empty , it is considered to be pure
    if len(labels) == 0:
        return 0
    
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions ** 2)


def entropy(pos_fraction):
    entropy_measure = -(pos_fraction * np.log2(pos_fraction) + (1- pos_fraction) * np.log2(1 - pos_fraction))
    return entropy_measure 