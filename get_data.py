from sklearn import datasets
from numpy import zeros

def get_iris_data(return_X_y=True, one_hot=True):
    X, Y = datasets.load_iris(return_X_y)
    N, d, C = len(Y), X.shape[1], Y.max()+1
    if one_hot:
        L = zeros((N, C))
        c = 0
        for y in Y:
            L[c, y] = 1.0
            c += 1
        return X, L, {'n_sample': N, 'dim': d, 'n_class': C}
    else:
        return X, Y, {'n_sample': N, 'dim': d, 'n_class': C}