from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    data_array = np.array(data)
    y = data_array[:, 0]
    X = data_array[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """Split the data x,y into two random subsets.
    
    input:
        x: set of images
        y: labels
        fract_tr: float, percentage of samples to put in the training set.
            If necessary, number of samples in the training set is rounded to
            the lowest integer number.
    
    output:
        Xtr: set of images (numpy array, training set)
        Xts: set of images (numpy array, test set)
        ytr: labels (numpy array, training set)
        yts: labels (numpy array, test set)
    
    """

    num_samples = y.size
    n_tr = int(num_samples * tr_fraction)

    idx = np.array(range(0, num_samples))
    np.random.shuffle(idx)  # shuffle the elements of idx

    tr_idx = idx[0:n_tr]
    ts_idx = idx[n_tr:]

    Xtr = x[tr_idx, :]
    ytr = y[tr_idx]

    Xts = x[ts_idx, :]
    yts = y[ts_idx]

    return Xtr, ytr, Xts, yts
