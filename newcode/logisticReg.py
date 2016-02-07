#!/usr/bin/python2.7
from __future__ import division
from time import time
import numpy as np
import scipy.sparse as ss
#import scipy.optimize

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def word_features(l):
    '''
    Extracting features from tokens
    '''
    return (l[0]+l[1]+l[2]+l[3], l[0]+l[1]+l[2], \
    l[1]+l[2]+l[3], l[0]+l[1], l[1]+l[2], l[2]+l[3], \
    l[0]+l[2], l[0]+l[3], l[1]+l[3], l[0], l[1],\
    l[2], l[3])

def dataextract(pp='in'):
    '''
    Simple stuff,relatively:
    1) extract features - mono, bi, tri and quad for each example
    2) maintain a dict - {eg: feats}
    3) maintain a set of all feats
    4) sort the feats - use the same feats for dev sets too
    5) use sparse lil_matrix(len(eg), len(feats))
    6) fillit up
    7) This is the feat matrix.
    '''

    traindata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in
                open('datasets/cleantrain.txt') if d.strip().split()[3] == pp]
    devdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d
               in open('datasets/cleandev.txt') if d.strip().split()[3] == pp]

    trainX = {eg: word_features(traindata[eg][0]) for eg in xrange(len(traindata))}
    Ytrain = [1 if y[1] == 'v' else -1 for y in traindata]
    devX = {eg: word_features(devdata[eg][0]) for eg in xrange(len(devdata))}
    Ydev = [1 if y[1] == 'v' else -1 for y in devdata]
    featset = []
    for wvals in trainX.values():
        for val in wvals:
            featset.append(val)

    setfeats = set(featset)
    featset = np.array(sorted(list(setfeats)))
    Xtrain = ss.lil_matrix((len(trainX), len(featset)))
    for eg, egfeat in trainX.items():
        onearr = featset.searchsorted(egfeat)
        Xtrain[eg,onearr] = np.ones(len(onearr))
    Xdev = ss.lil_matrix((len(devX), len(featset)))
    for deg, degfeat in devX.items():
        donearr = featset.searchsorted(list(setfeats.intersection(degfeat)))
        Xdev[deg,donearr] = np.ones(len(donearr))

    return np.array(Xtrain.todense()), Ytrain, np.array(Xdev.todense()), Ydev

#def train_w(X, Y, C=0.1):
#    def f(w):
#        return -log_likelihood(X, Y, w, C)
#
#    def fprime(w):
#        return -log_likelihood_grad(X, Y, w, C)
#
#    K = X.shape[1]
#    initial_guess = np.zeros(K)
#    print 'here with C = ', C
#    return scipy.optimize.fmin_cg(f, initial_guess, fprime,
#                                    disp=True)

class Fobos(object):
    def __init__(self, eta, tau):
        self.eta = eta
        self.tau = tau
        self.iteration = 1
        self.lr = self.eta / np.sqrt(self.iteration)

    def fobos_l1(self, w):
        nu = self.lr * self.tau
        return np.multiply(np.sign(w), np.max(np.abs(w) - nu, 0))

    def fobos_l2(self, w):
        nu = self.lr * self.tau
        return w / (1 + nu)

    def optimize(self, w_k, grad, reg_type='l2'):
        self.lr = self.eta / np.sqrt(self.iteration)
        w_k1 = w_k - self.lr * grad
        if reg_type is 'l2':
            w_k2 = self.fobos_l2(w_k1)
            norm = np.linalg.norm(w_k, 2)**2  / 2
        elif reg_type is 'l1':
            w_k2 = self.fobos_l1(w_k1)
            norm = np.linalg.norm(w_k, 1)

        self.iteration += 1

        return w_k2, norm

class Linear(object):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.nsamples = len(Y)
        self.w = np.zeros(X.shape[1])

    def predict(self, w, x):
        return logistic(np.dot(w, x)) > 0.5 or -1

    def accuracy(self, w):
        n_correct = 0
        for i in range(self.nsamples):
            if self.predict(w, self.X[i]) == self.Y[i]:
                n_correct += 1
        return n_correct * 1.0 / self.nsamples

    def log_l(self, w):
        ll = 0
        for n in xrange(self.nsamples):
            ll +=  np.log(logistic(self.Y[n] * np.dot(w, self.X[n])))
        return ll

    def log_likelihood(self):
        self.ll = 0
        for n in xrange(self.nsamples):
            self.ll +=  np.log(logistic(self.Y[n] *
                                        np.dot(self.w,self.X[n])))

    def log_l_grad(self, w):
        grad = np.zeros(self.X.shape[1], dtype=np.float)
        for n in range(self.nsamples):
            grad = grad +  self.Y[n] * self.X[n] * logistic(-self.Y[n] *
                                                    (np.dot(self.w, self.X[n])))
        return grad

    def objective(self, w, tau, norm):
        ll = self.log_l(w)
        return - (ll - tau*norm)

    def logl(self):
        self.log_likelihood()
        return self.ll

    def update(self, w_k, norm):
        self.w = w_k
        self.norm = norm

    def output(self, w, tau):
        grad = self.log_l_grad(w)
        grad = grad - tau * w
        return - grad



def log_likelihood_grad(X, Y, w, C=0.1):
    K = len(w)
    N = len(X)
    s = np.zeros(K)

    for i in range(N):
        s += Y[i] * X[i] * logistic(-Y[i] * np.dot(X[i], w))

    s -= C * w
    return s


def main(maxiter=10, tau=0.01, eta=0.01, prep='into'):
#    all_C = np.arange(0.0, 1, 0.01)
    Xtrain, Ytrain, Xdev, Ydev = dataextract(pp=prep)
    operator = Linear(Xtrain, Ytrain)
    doperator = Linear(Xdev, Ydev)
    optimizer = Fobos( eta=0.1, tau=0.1)
    l = Xtrain.shape[1]
    print 'Number of Training Examples = ', len(Ytrain), \
        ' Number of Dev Examples = ', len(Ydev), ' Dimensionality = ', l
    w_k = np.zeros(l, dtype=np.float)
    norm = 0
    for i in xrange(int(maxiter)):
        start_loop = time()
        cost = operator.objective(w_k, float(tau), norm)
        grad = operator.output(w_k, float(tau))
        w_k1, norm = optimizer.optimize(w_k, grad)
        operator.update(w_k, norm)
        end_loop = time()
        print '%d cost=%.2f norm=%.2f tracc=%.2f devacc=%.2f time=%.2f' % (i+1,
        cost, norm, operator.accuracy(w_k), doperator.accuracy(w_k), end_loop -
                                                                         start_loop)
        w_k = w_k1
#    for C in all_C:
#       w = train_w(Xtrain, Ytrain, C)
#       print accuracy(Xdev, Ydev, w)

if __name__ == '__main__':
    import plac
    plac.call(main)
