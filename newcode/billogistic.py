from __future__ import division
from sklearn import preprocessing
from scipy.io import mmread
from time import time
import numpy as np

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

class Bilnear(object):

    def __init__(self, samples, Vdict, Ndict, Mdict, Y, Winit='random'):
        self.samples = samples
        self.Vdict = Vdict
        self.Ndict = Ndict
        self.Mdict = Mdict
        self.Y = Y
        self.dim = len(Vdict.values()[0])
        self.nsamples = len(self.samples)
        self.Xi = {}
        self.grad = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        if Winit is 'random':
            self.Wmat = np.matrix(np.random.rand(self.dim, self.dim))
        elif Winit is 'zeros':
            self.Wmat = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        elif Winit is 'identity':
            self.Wmat = np.matrix(np.identity(self.dim, dtype=np.float))
        self.norm = np.linalg.norm(self.Wmat)
        self.lsigma = None
        self.rsigma = None
        self.pcaSigma = None
        self.zcaSigma = None
        self.ryotaSigma = None
        self.Xpi = {}

    def _lsigma(self):
        self.lsigma = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        if self.Xi == {}:
            for iter_ in xrange(self.nsamples):
#                v, n, m = self.samples[iter_]
#                self.Xi[iter_] = self.scale(self.Vdict[v], self.Ndict[n], self.Mdict[m])
                self.Xi[iter_] = self.scale(self.Vdict[iter_], self.Ndict[iter_],
                                            self.Mdict[iter_])
                self.lsigma += self.Xi[iter_].T * self.Xi[iter_]
        else:
            for iter_ in xrange(self.nsamples):
                self.rsigma += self.Xi[iter_] * self.Xi[iter_].T

        self.lsigma = self.lsigma / self.nsamples

    def _rsigma(self):
        self.rsigma = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        if self.Xi == {}:
            for iter_ in xrange(self.nsamples):
#                v, n, m = self.samples[iter_]
#                self.Xi[iter_] = self.scale(self.Vdict[v], self.Ndict[n], self.Mdict[m])
                self.Xi[iter_] = self.scale(self.Vdict[iter_], self.Ndict[iter_],
                                            self.Mdict[iter_])
                self.rsigma += self.Xi[iter_] * self.Xi[iter_].T
        else:
            for iter_ in xrange(self.nsamples):
                self.rsigma += self.Xi[iter_] * self.Xi[iter_].T
        self.rsigma = self.rsigma / self.nsamples

    def grad_init(self):
        self.grad = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))

    def _pcaSigma(self, epsilon=1e-10):
        if self.rsigma is None:
            self._rsigma()
        u, s, vt = np.linalg.svd(self.rsigma)
        self.pcaSigma = np.diag(1. / np.sqrt(np.matrix(np.diag(s)) + epsilon)) * np.matrix(u).T

    def _zcaSigma(self, epsilon=1e-10):
        if self.rsigma is None:
            self._rsigma()
        u, s, vt = np.linalg.svd(self.rsigma)
        self.zcaSigma = np.matrix(u) * np.diag(1. / np.sqrt(np.matrix(np.diag(s)) +
                                                         epsilon)) * np.matrix(u).T

    def _ryotaSigma(self, epsilon=1e-10):
        if self.rsigma is None:
            self._rsigma()
        if self.lsigma is None:
            self._lsigma
        ul, sl, vtl = np.linalg.svd(self.lsigma)
        ur, sr, vtr = np.linalg.svd(self.rsigma)
        rsig = np.diag(1. / np.sqrt(np.sqrt(np.matrix(np.diag(sr)) +
                                                 epsilon))) * np.matrix(ur).T
        lsig = np.diag(1. / np.sqrt(np.sqrt(np.matrix(np.diag(sl)) +
                                                 epsilon))) * np.matrix(ul).T
        self.ryotaSigma = (rsig, lsig)

    def scale(self, v, n, m):
        return preprocessing.scale((m*(v-n).T))

    def preprocess(self, sigmatype=None):
        if sigmatype is 'pca':
            if self.pcaSigma is None:
                self._pcaSigma()
            for i,xi in self.Xi.items():
                self.Xpi[i] = self.pcaSigma * xi
        elif sigmatype is 'zca':
            if self.zcaSigma is None:
                self._zcaSigma()
            for i,xi in self.Xi.items():
                self.Xpi[i] = self.zcaSigma * xi
        elif sigmatype is 'ryota':
            if self.ryotaSigma is None:
                self._ryotaSigma()
            rsig, lsig = self.ryotaSigma()
            for i,xi in self.Xi.items():
                self.Xpi[i] = rsig * xi * lsig
        else:
            self._pcaSigma()
            self.Xpi = self.Xi

    def predict(self,W, X):
        return logistic(np.trace(W*X)) > 0 or -1


    def accuracy(self):
        n_correct = 0
        for i in range(self.nsamples):
            if self.predict(self.Wmat, self.Xpi[i]) == self.Y[i]:
                n_correct += 1
        return n_correct * 1.0 / self.nsamples

    def log_likelihood(self):
        self.ll = 0
        for n in xrange(self.nsamples):
            self.ll +=  np.log(logistic(self.Y[n] * np.trace(self.Wmat * self.Xpi[n])))

    def log_l(self, Wmat, C):
        ll = 0
        for n in xrange(self.nsamples):
            ll +=  np.log(logistic(self.Y[n] * np.trace(Wmat * self.Xpi[n])))
        return ll - C * (np.linalg.norm(Wmat,2)**2) / 2

    def gradient(self):
        self.grad = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        for n in range(self.nsamples):
            self.grad +=  self.Y[n] * self.Xpi[n].T * logistic(-self.Y[n] *
                                                    np.trace(self.Wmat * self.Xpi[n]))

    def log_l_grad(self, Wmat,C):
        grad = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        for n in range(self.nsamples):
            grad +=  self.Y[n] * self.Xpi[n].T * logistic(-self.Y[n] *
                                                    np.trace(Wmat * self.Xpi[n]))
        grad -= C * Wmat
        return grad

    def objective(self, tau):
        self.log_likelihood()
        return (self.ll + tau*self.norm)

    def logl(self):
        self.log_likelihood()
        return self.ll

    def update(self, w_k, norm):
        self.Wmat = w_k
        self.norm = norm

    def output(self):
        self.gradient()
        return self.Wmat, self.grad

class Fobos(object):
    def __init__(self, eta, tau):
        self.eta = eta
        self.tau = tau
        self.iteration = 1
        self.lr = self.eta / np.sqrt(self.iteration)

    def fobos_nn(self, w_k):
        nu = self.tau * self.lr
        u, s, vt = np.linalg.svd(w_k)
        sdash = np.maximum(s - nu, 0)
        return (np.matrix(u) * np.matrix(np.diag(sdash) * np.matrix(vt))), s

    def fobos_l1(self, w_k):
        nu = self.lr * self.tau
        return np.multiply(np.sign(w_k), np.max(np.abs(w_k) - nu, 0))

    def fobos_l2(self, w_k):
        nu = self.lr * self.tau
        return w_k / (1 + nu)

    def optimize(self, w_k, grad, reg_type='nn'):
        self.lr = self.eta / np.sqrt(self.iteration)
        w_k = w_k - self.lr * grad
        if reg_type is 'nn':
            w_k, s = self.fobos_nn(w_k)
            norm = np.sum(sum(s))
        elif reg_type is 'l2':
            w_k = self.fobos_l2(w_k)
            norm = np.linalg.norm(w_k, 2)**2  / 2
        elif reg_type is 'l1':
            w_k = self.fobos_l1(w_k)
            norm = np.linalg.norm(w_k, 1)

        self.iteration += 1

        return w_k, norm

def extdata(pp='for'):
    samples = []
    Vdict = {}
    Ndict = {}
    Mdict = {}
    Y = []
    sam = [(l.strip().split()[1:5], l.strip().split()[5]) for l in
               open('datasets/cleantrain.txt')]
    hdata = [l.strip() for l in open('datasets/forhead.txt')]
    mdata = [l.strip() for l in open('datasets/formod.txt')]
    hmat = np.matrix(mmread('datasets/trainhw2v.mtx').todense())
    mmat = np.matrix(mmread('datasets/trainmw2v.mtx').todense())
    print len(hdata), hmat.shape
    for s,y in sam:
        if s[2] == pp:
            samples.append(list(s[i] for i in [0,1,3]))
            if y is 'v':
                Y.append(1)
            elif y is 'n':
                Y.append(-1)

    for iter_ in xrange(len(samples)):
        print samples[iter_][0]
        Vdict[iter_] = hmat[hdata.index(samples[iter_][0])]
        Ndict[iter_] = hmat[hdata.index(samples[iter_][1])]
        Mdict[iter_] = mmat[mdata.index(samples[iter_][2])]

    return samples, Vdict, Ndict, Mdict, Y

def main(samples, Vdict, Ndict, Mdict, Y, maxiter, eta, tau):

    operator = Bilnear(samples, Vdict, Ndict, Mdict, Y)
    optimizer = Fobos(eta, tau)
    operator.preprocess()
    for i in xrange(maxiter):
        start_loop = time()
        operator.grad_init()
        cost = -operator.objective(tau)
        w_k, grad = operator.output()
        w_k, norm = optimizer.optimize(w_k, -grad)
        operator.update(w_k, norm)
        end_loop = time()
        print i, cost, norm, operator.accuracy(), end_loop - start_loop












