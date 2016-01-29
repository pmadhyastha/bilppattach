from __future__ import division
from scipy.io import mmread
from time import time
import numpy as np
from sklearn import preprocessing

def logistic(z):
    return (1.0 / (1.0 + np.exp(-z)))

class Bilinear(object):

    def __init__(self, samples, Vdict, Ndict, Mdict, Y, Winit='random'):
        self.samples = samples
        self.Vdict = Vdict
        self.Ndict = Ndict
        self.Mdict = Mdict
        self.ll = None
        self.Y = Y
        self.dim = len(Vdict.values()[0])
        self.nsamples = len(self.samples)
        self.grad = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        if Winit is 'random':
            self.Wmat = np.matrix(np.random.rand(self.dim, self.dim))
        self.norm = np.linalg.norm(self.Wmat)

    def grad_init(self):
        self.grad = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))

    def scale(self, v, n, m):
        return preprocessing.scale((m*(v-n).T))

    def predict(self,W, X):
        if logistic(np.trace(W*X)) > 0.5:
            return 1
        else:
            return -1

    def accuracy(self):
        n_correct = 0
        for i in range(self.nsamples):
            if self.predict(self.Wmat, self.Xpi[i]) == self.Y[i]:
                n_correct += 1
        return n_correct * 1.0 / self.nsamples

    def log_likelihood(self):
        if self.ll is None:
            self.ll = 0
        for n in xrange(self.nsamples):
            self.ll +=  np.log(logistic(self.Y[n] * np.trace(self.Wmat * self.Xpi[n])))

    def log_l(self, Wmat, C):
        ll = 0
        for n in xrange(self.nsamples):
            ll +=  np.log(logistic(self.Y[n] * np.trace(Wmat * self.Xpi[n])))
        return ll - C * (np.linalg.norm(Wmat,2)**2) / 2

    def gradient(self):
        if not self.grad:
            self.grad = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        for n in range(self.nsamples):
            self.grad +=  self.Y[n] * self.Xpi[n].T * logistic(-self.Y[n] *
                                                    np.trace(np.dot(self.Wmat,self.Xpi[n])))



