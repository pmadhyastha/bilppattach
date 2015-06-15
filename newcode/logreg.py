import numpy as np

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict(w, x):
        return logistic(np.dot(w, x)) > 0.5 or -1

def log_likelihood(X, Y, w, C=0.1):
    return np.sum(np.log(logistic(Y * np.dot(X, w)))) - C/2 * np.dot(w, w)

def log_likelihood_grad(X, Y, w, C=0.1):
    K = len(w)
    N = len(X)
    s = np.zeros(K)

    for i in range(N):
        s += Y[i] * X[i] * logistic(-Y[i] * np.dot(X[i], w))

    s -= C * w

    return s
def grad_num(X, Y, w, f, eps=0.00001):
    K = len(w)
    ident = np.identity(K)
    g = np.zeros(K)

    for i in range(K):
        g[i] += f(X, Y, w + eps * ident[i])
        g[i] -= f(X, Y, w - eps * ident[i])
        g[i] /= 2 * eps

    return g

def test_log_likelihood_grad(X, Y):
    n_attr = X.shape[1]
    w = np.array([1.0 / n_attr] * n_attr)

    print "with regularization"
    print log_likelihood_grad(X, Y, w)
    print grad_num(X, Y, w, log_likelihood)

    print "without regularization"
    print log_likelihood_grad(X, Y, w, C=0)
    print grad_num(X, Y, w, lambda X,Y,w: log_likelihood(X,Y,w,C=0))

import scipy.optimize

def train_w(X, Y, C=0.1):
    def f(w):
        print -log_likelihood(X, Y, w, C)

        return -log_likelihood(X, Y, w, C)

    def fprime(w):
        return -log_likelihood_grad(X, Y, w, C)

    K = X.shape[1]
    initial_guess = np.zeros(K)

    return scipy.optimize.fmin_bfgs(f, initial_guess, fprime, disp=False)

def accuracy(X, Y, w):
    n_correct = 0
    for i in range(len(X)):
        if predict(w, X[i]) == Y[i]:
            n_correct += 1
    return n_correct * 1.0 / len(X)

def fold(arr, K, i):
    N = len(arr)
    size = np.ceil(1.0 * N / K)
    arange = np.arange(N) # all indices
    heldout = np.logical_and(i * size <= arange, arange < (i+1) * size)
    rest = np.logical_not(heldout)
    return arr[heldout], arr[rest]

def kfold(arr, K):
    return [fold(arr, K, i) for i in range(K)]

def avg_accuracy(all_X, all_Y, C):
    s = 0
    K = len(all_X)
    for i in range(K):
        X_heldout, X_rest = all_X[i]
        Y_heldout, Y_rest = all_Y[i]
        w = train_w(X_rest, Y_rest, C)
        s += accuracy(X_heldout, Y_heldout, w)
    return s * 1.0 / K

def train_C(X, Y, K=10):
    all_C = np.arange(0, 1, 0.1) # the values of C to try out
    all_X = kfold(X, K)
    all_Y = kfold(Y, K)
    all_acc = np.array([avg_accuracy(all_X, all_Y, C) for C in all_C])
    return all_C[all_acc.argmax()]

def read_data(filename, sep=",", filt=int):

    def split_line(line):
        return line.split(sep)

    def apply_filt(values):
        return map(filt, values)

    def process_line(line):
        return apply_filt(split_line(line))

    f = open(filename)
    lines = map(process_line, f.readlines())
    # "[1]" below corresponds to x0
    X = np.array([[1] + l[1:] for l in lines])
    # "or -1" converts 0 values to -1
    Y = np.array([l[0] or -1 for l in lines])
    f.close()

    return X, Y
def main():
    X_train, Y_train = read_data("SPECT.train")

    # Uncomment the line below to check the gradient calculations
    #test_log_likelihood_grad(X_train, Y_train); exit()

    C = train_C(X_train, Y_train)
    print "C was", C
    w = train_w(X_train, Y_train, C)
    print "w was", w

    X_test, Y_test = read_data("SPECT.train")
    print "accuracy was", accuracy(X_test, Y_test, w)

if __name__ == "__main__": main()
