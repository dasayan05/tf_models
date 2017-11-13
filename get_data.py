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

def get_mini_mnist(file='mnist_mini.data', bsize=None, as_image=False):
    import pickle

    with open(file, 'rb') as f:
        D, L = pickle.load(f)
        if as_image:
            D = D.reshape((-1, 28, 28))

    if bsize == None:
        return D, L
    else:
        return D[:bsize,...], L[:bsize,...]