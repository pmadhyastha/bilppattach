from __future__ import division
from sklearn import preprocessing
from scipy.io import mmread
from time import time
import numpy as np

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))



class Combo(object):

    def __init__(self, samples, Vdict, Ndict, Mdict, Y, fmat,Winit='random'):
        self.samples = samples
        self.Vdict = Vdict
        self.Ndict = Ndict
        self.Mdict = Mdict
        self.Y = Y
        self.dim = (Vdict.values()[0]).shape[1]
        self.nsamples = len(self.samples)
        self.Xi = {}
        self.fmat = {}
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
        self.ldim = len(fmat[0])

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

                xi = self.scale(self.Vdict[iter_], self.Ndict[iter_],
                                            self.Mdict[iter_])
                self.Xi[iter_]  = xi
                self.rsigma += xi * xi.T
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
        return np.matrix(preprocessing.scale((m.T*(v.T-n.T).T)))


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

    def predict(self,w, x, W, X):
        if logistic(np.dot(w,x) + np.trace(W*X)) > 0.5:
            p = 1
        else:
            p = -1
        return p


    def accuracy(self, w, Wmat):
        n_correct = 0
        for i in range(self.nsamples):
            if self.predict(w, self.fmat[i], Wmat, self.Xpi[i]) == self.Y[i]:
                n_correct += 1
        return n_correct * 1.0 / self.nsamples

    def log_likelihood(self):
        self.ll = 0
        for n in xrange(self.nsamples):
            self.ll +=  np.log(logistic(self.Y[n] *(np.dot(x,self.fmat[i]) + np.trace(self.Wmat *
                                                              self.Xpi[n]))))

    def log_l(self,w, Wmat):
        ll = 0
        for n in xrange(self.nsamples):
            ll +=  np.log(logistic(self.Y[n] *(np.dot(x,self.fmat[i]) +
                                                np.trace(Wmat * self.Xpi[n]))))
        return ll

    def gradients(self):
        self.gradb = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        self.gradl = np.array(np.zeros(self.ldim))

        for n in range(self.nsamples):
            self.gradb +=  self.Y[n] * self.Xpi[n].T * logistic(-self.Y[n] *
                                     (np.dot(x,self.fmat[n]) +
                                      np.trace(self.Wmat * self.Xpi[n])))
            self.gradl +=  self. Y[n] * self.fmat[n] * logistic(-self.Y[n] *
                                            (np.dot(x,self.fmat[n]) +
                                      np.trace(self.Wmat * self.Xpi[n])))


    def log_l_grad(self, Wmat):
        grad = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        for n in range(self.nsamples):
            grad = grad +  self.Y[n] * self.Xpi[n].T * logistic(-self.Y[n] * np.trace(Wmat * self.Xpi[n]))
        return grad

    def objective(self, Wmat, tau, norm):
        ll = self.log_l(Wmat)
        return - (ll - tau*norm)

    def logl(self):
        self.log_likelihood()
        return self.ll

    def update(self, w_k, norm):
        self.Wmat = w_k
        self.norm = norm

    def output(self, Wmat, tau):
        grad = self.log_l_grad(Wmat)
        grad = grad - tau * Wmat
        return - grad




def extdata(pp='for'):

    def word_features(l):
        '''
        Extracting features from tokens
        '''
        return (l[0]+l[1]+l[2]+l[3], l[0]+l[1]+l[2], \
        l[1]+l[2]+l[3], l[0]+l[1], l[1]+l[2], l[2]+l[3], \
        l[0]+l[2], l[0]+l[3], l[1]+l[3], l[0], l[1],\
        l[2], l[3])


    samples = []
    Vdict = {}
    Ndict = {}
    Mdict = {}
    lsample = []
    LinDict = {}
    Y = []
    sam = [(l.strip().split()[1:5], l.strip().split()[5]) for l in
               open('datasets/cleantrain.txt')]
    hdata = [l.strip() for l in open('datasets/forhead.txt')]
    mdata = [l.strip() for l in open('datasets/formod.txt')]
    hmat = np.matrix(mmread('datasets/trainhw2v.mtx').todense())
    mmat = np.matrix(mmread('datasets/trainmw2v.mtx').todense())
    for s,y in sam:
        if s[2] == pp:
            samples.append(list(s[i] for i in [0,1,3]))
            lsample.append(word_features(s))
            if y is 'v':
                Y.append(1)
            elif y is 'n':
                Y.append(-1)

    for iter_ in xrange(len(samples)):
        print samples[iter_][0]
        Vdict[iter_] = hmat[hdata.index(samples[iter_][0])]
        Ndict[iter_] = hmat[hdata.index(samples[iter_][1])]
        Mdict[iter_] = mmat[mdata.index(samples[iter_][2])]
        LinDict[iter_] = lsample[iter_]

    feats = []
    for s in lsample:
        for i in s:
            feats.append(i)

    uniqfeats =  list(set(feats))
    fmat = np.zeros((len(samples), len(uniqfeats)))
    featarr = np.array(sorted(uniqfeats))
    for s, f in LinDict.items():
        oneshot = np.ones(len(featarr.searchsorted(np.array(f))))
        fmat[s][featarr.searchsorted(np.array(f))] = oneshot

    return samples, Vdict, Ndict, Mdict, Y, fmat

