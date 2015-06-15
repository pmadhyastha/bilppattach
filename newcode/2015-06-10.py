
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
import billogistic
import scipy.optimize
def train_w(X, Y, C=0.1):
    def f(w):
        return - billogistic.log_likelihood(X, Y, w, C)
    def fprime(w):
        return - billogistic.grad(X, Y, w, C)
    K = X.shape[1]
    initial_guess = np.zeros(K) 
    return scipy.optimize.fmin_bfgs(f, initial_guess, fprime, disp=False)

def accuracy(X, Y, w):
    n_correct = 0
    for i in range(len(X)):
        if predict(w, X[i]) == Y[i]:
            n_correct += 1
    return n_correct * 1.0 / len(X) 

get_ipython().system('wget http://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train ')
def read_data(filename, sep=",", filt=int):
    def split_line(line):
        return line.split(sep)
    def apply_filt(values):
        return map(filt, values)
    def process_line(line):
        return apply_filt(split_line(line))
    f = open(filename)
    lines = map(process_line, f.readlines())
    X = np.array([[1]+l[1:] for l in lines])
    Y = np.array([l[0] or -1 for l in lines])
    f.close()
    return X, Y

X_train, Y_train = read_data("SPECT.train")
def read_data(filename, sep=",", filt=int):
    def split_line(line):
        return line.split(sep)
    def apply_filt(values):
        return map(filt, values)
    def process_line(line):
        return apply_filt(split_line(line))
    f = open(filename)
    lines = map(process_line, f.readlines())
    X = np.array([[1] + l[1:] for l in lines])
    Y = np.array([l[0] or -1 for l in lines])
    f.close()
    return X, Y

X_train, Y_train = read_data("SPECT.train")
def read_data(filename, sep=",", filt=int):
    def split_line(line):
        return line.split(sep)
    def apply_filt(values):
        return map(filt, values)
    def process_line(line):
        return apply_filt(split_line(line))
    f = open(filename)
    lines = map(process_line, f.readlines())
    X = np.array([l[1:] for l in lines])
    Y = np.array([l[0] or -1 for l in lines])
    f.close()
    return X, Y

X_train, Y_train = read_data("SPECT.train")
get_ipython().magic('paste')
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
X_train, Y_train = read_data("SPECT.train")
quit()

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
import scipy.optimize
def train_w(X, Y, C=0.1):
    def f(w):
        return - billogistic.log_likelihood(X, Y, w, C)
    def fprime(w):
        return - billogistic.grad(X, Y, w, C)
    K = X.shape[1]
    initial_guess = np.zeros(K) 
    return scipy.optimize.fmin_bfgs(f, initial_guess, fprime, disp=False)

def accuracy(X, Y, w):
    n_correct = 0
    for i in range(len(X)):
        if predict(w, X[i]) == Y[i]:
            n_correct += 1
    return n_correct * 1.0 / len(X)

get_ipython().magic(u'paste')
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
X_train, Y_train = read_data("SPECT.train")
X_train
w = train_w(X_train, Y_train, C=0.001)
import billogistic
w = train_w(X_train, Y_train, C=0.001)
w
accuracy(X_train, Y_train, w) 
def accuracy(X, Y, w):
    n_correct = 0
    for i in range(len(X)):
        if billogistic.predict(w, X[i]) == Y[i]:
            n_correct += 1
    return n_correct * 1.0 / len(X)

accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.01)
accuracy(X_train, Y_train, w)
len(X)
len(X_train)
X_train
X_train.shape
Y
Y_train
w = train_w(X_train, Y_train, C=0.1)
accuracy(X_train, Y_train, w)
def accuracy(X, Y, w):
    n_correct = 0
    for i in range(len(X)):
        if billogistic.predict(w, X[i]) == Y[i]:
            n_correct += 1
    print n_correct
    
accuracy(X_train, Y_train, w)
relaod(billogistic) 
reload(billogistic)
w = train_w(X_train, Y_train, C=0.1)
reload(billogistic)
w = train_w(X_train, Y_train, C=0.1)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.01)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.001)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.0001)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.01)
accuracy(X_train, Y_train, w)
def accuracy(X, Y, w):
    n_correct = 0
    for i in range(len(X)):
        if billogistic.predict(w, X[i]) == Y[i]:
            n_correct += 1
    return n_correct * 1.0 / len(X)

accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.05)
accuracy(X_train, Y_train, w)
accuracy(X_train, Y_train, 0.02)
w = train_w(X_train, Y_train, C=0.02)
accuracy(X_train, Y_train, 0.02)
accuracy(X_train, Y_train, w)
accuracy(X_train, Y_train, 0.01)
w = train_w(X_train, Y_train, C=0.01)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.03)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.04)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.035)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.031)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.032)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.035)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.034)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.033)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.032)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.0325)
accuracy(X_train, Y_train, w)
w = train_w(X_train, Y_train, C=0.0326)
accuracy(X_train, Y_train, w)
def accuracy(X, Y, w):
    n_correct = 0
    for i in range(len(X)):
        if billogistic.predict(w, X[i]) == Y[i]:
            n_correct += 1
    return n_correct * 1.0 / len(X)
reload(billogistic)

w = train_w(X_train, Y_train, C=0.0326)
accuracy(X_train, Y_train, w)
def train_w(X, Y, C=0.1):
    def f(w):
        return - billogistic.log_likelihood(X, Y, w, C)
    def fprime(w):
        return - billogistic.grad(X, Y, w, C)
    K = X.shape[1]; print K
    initial_guess = np.zeros(K) 
    return scipy.optimize.fmin_bfgs(f, initial_guess, fprime, disp=False)

w = train_w(X_train, Y_train, C=0.0326)
def train_w(X, Y, C=0.1):
    def f(w):
        return - billogistic.log_likelihood(X, Y, w, C)
    def fprime(w):
        return - billogistic.grad(X, Y, w, C)
    K = X.shape[0]; print K
    initial_guess = np.zeros(K) 
    return scipy.optimize.fmin_bfgs(f, initial_guess, fprime, disp=False)

w = train_w(X_train, Y_train, C=0.0326)
def train_w(X, Y, C=0.1):
    def f(w):
        return - billogistic.log_likelihood(X, Y, w, C)
    def fprime(w):
        return - billogistic.grad(X, Y, w, C)
    K = X.shape[1]; print K
    initial_guess = np.zeros(K) 
    return scipy.optimize.fmin_bfgs(f, initial_guess, fprime, disp=False)

w = train_w(X_train, Y_train, C=0.0326)
accuracy(X_train, Y_train, w)
x = np.matrix(np.random.rand(4))
x
x = np.matrix(np.random.rand(4)).transpose() 
w = np.matrix(np.random.rand(4,4)).transpose()
np.trace(w,x)
np.trace(w*x)
np.trace(x*w)
np.trace(x.transpose()*w)
w
x
x = np.matrix(np.random.rand(4,10)).transpose()
np.trace(x.transpose()*w)
np.trace(w*x)
np.trace(x*w)
x = np.matrix(np.random.rand(4)).transpose()
np.trace(x*w)
