from __future__ import division
from sklearn import preprocessing
from scipy.io import mmread
from time import time
import numpy as np

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

class Bilnear(object):

    def __init__(self, samples, Vdict, Ndict, Mdict, Y, Winit='zeros'):
        self.samples = samples
        self.Vdict = Vdict
        self.Ndict = Ndict
        self.Mdict = Mdict
        self.ll = None
        self.Y = Y
<<<<<<< HEAD
        self.dim = Vdict.values()[0].shape[1]
=======
        self.dim = (Vdict.values()[0]).shape[1]
>>>>>>> 6d9e0afd15b1b06156df0a1805e3082f22a53d7f
        self.nsamples = len(self.samples)
        self.Xi = {}
        self.grad = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        if Winit is 'random':
            self.Wmat = np.matrix(np.random.rand(self.dim, self.dim))
        elif Winit is 'zeros':
            self.Wmat = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        elif Winit is 'identity':
            self.Wmat = np.matrix(np.identity(self.dim, dtype=np.float))
        else:
            self.Wmat = Winit
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
        print self.dim
        self.rsigma = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        if self.Xi == {}:
            for iter_ in xrange(self.nsamples):
#                v, n, m = self.samples[iter_]
#                self.Xi[iter_] = self.scale(self.Vdict[v], self.Ndict[n], self.Mdict[m])

                xi = self.scale(self.Vdict[iter_], self.Ndict[iter_],
                                            self.Mdict[iter_])
<<<<<<< HEAD

                self.rsigma += self.Xi[iter_] * self.Xi[iter_].T
=======
                self.Xi[iter_]  = xi
                self.rsigma += xi * xi.T
>>>>>>> 6d9e0afd15b1b06156df0a1805e3082f22a53d7f
        else:
            for iter_ in xrange(self.nsamples):
                self.rsigma += self.Xi[iter_] * self.Xi[iter_].T
        self.rsigma = self.rsigma / self.nsamples

    def grad_init(self):
        self.grad = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))

    def _pcaSigma(self, epsilon=1e-10):
        if self.lsigma is None:
            self._lsigma()
        u, s, vt = np.linalg.svd(self.lsigma)
        self.pcaSigma = np.matrix(np.diag(1. / np.sqrt(s + epsilon))) * np.matrix(u).T

    def _zcaSigma(self, epsilon=1e-10):
        if self.rsigma is None:
            self._rsigma()
        u, s, vt = np.linalg.svd(self.rsigma)
        self.zcaSigma = np.matrix(u) * np.matrix(np.diag(1. / np.sqrt(s +
                                                         epsilon))) * np.matrix(u).T

    def _ryotaSigma(self, epsilon=1e-10):
        if self.rsigma is None:
            self._rsigma()
        if self.lsigma is None:
            self._lsigma()
        ul, sl, vtl = np.linalg.svd(self.lsigma)
        ur, sr, vtr = np.linalg.svd(self.rsigma)
        rsig = np.matrix(np.diag(1. / np.sqrt(np.sqrt(sr) + epsilon)))* np.matrix(ur).T
        lsig = np.matrix(np.diag(1. / np.sqrt(np.sqrt(sl) + epsilon))) * np.matrix(ul).T

        self.ryotaSigma = (rsig, lsig)

    def get_sigma(sigtype='pca'):
        if self.pcaSigma:
            return self.pcaSigma
        else:
            self._pcaSigma()
            return self.pcaSigma

    def scale(self, v, n, m):
<<<<<<< HEAD
        return preprocessing.scale(np.dot(m.T,(v-n)))

=======
        return np.matrix(preprocessing.scale((m.T*(v.T-n.T).T)))


>>>>>>> 6d9e0afd15b1b06156df0a1805e3082f22a53d7f
    def preprocess(self, sigmatype='pca'):
        if sigmatype is 'pca':
            if self.pcaSigma is None:
                self._pcaSigma()
            for i,xi in self.Xi.items():
        #        self.Xpi[i] = self.pcaSigma * xi
                self.Xpi[i] = xi
        elif sigmatype is 'zca':
            if self.zcaSigma is None:
                self._zcaSigma()
            for i,xi in self.Xi.items():
                self.Xpi[i] = self.zcaSigma * xi
        elif sigmatype is 'ryota':
            if self.ryotaSigma is None:
                self._ryotaSigma()
            rsig, lsig = self.ryotaSigma
            for i,xi in self.Xi.items():
                self.Xpi[i] = rsig * xi * lsig
<<<<<<< HEAD
#        else:
#            self._pcaSigma()
#            self.Xpi = self.Xi
#
    def predict(self,W, X):
        if logistic(np.trace(np.dot(W,X))) > 0.5:
            return 1
        else:
            return -1
=======
        else:
            self._pcaSigma()
            self.Xpi = self.Xi

    def preprocessVal(self, Xi, sigmatype=None, lsigma=None, rsigma=None):
         if sigmatype is 'pca':
            for i,xi in Xi.items():
                Xpi[i] = rsigma * xi
         elif sigmatype is 'zca':
            for i,xi in Xi.items():
                Xpi[i] = zcaSigma * xi
         elif sigmatype is 'ryota':
            for i,xi in Xi.items():
                Xpi[i] = rsigma * xi * lsigma
         else:
            self._pcaSigma()
            self.Xpi = self.Xi

    def predict(self,W, X):
        if logistic(np.trace(W*X)) > 0.5:
            p = 1
        else:
            p = -1
        return p
>>>>>>> 6d9e0afd15b1b06156df0a1805e3082f22a53d7f

    def accuracy_gen(self, Wmat, X, Y):
        n_correct = 0
        for i,v in X:
            if self.predict(Wmat, v) == Y[i]:
                n_correct += 1
        return n_correct * 1.0 / len(X.keys())

    def accuracy(self, Wmat):
        n_correct = 0
        for i in range(self.nsamples):
            if self.predict(Wmat, self.Xpi[i]) == self.Y[i]:
                n_correct += 1
        return n_correct * 1.0 / self.nsamples

    def log_likelihood(self):
        self.ll = 0
        for n in xrange(self.nsamples):
            self.ll +=  np.log(logistic(self.Y[n] *
                                        np.trace(np.dot(self.Wmat,self.Xpi[n]))))

    def log_l(self, Wmat):
        ll = 0
        for n in xrange(self.nsamples):
<<<<<<< HEAD
            ll +=  np.log(logistic(self.Y[n] * np.trace(np.dot(Wmat,
                                                               self.Xpi[n]))))
        return ll - C * (np.linalg.norm(Wmat,2)**2) / 2
=======
            ll +=  np.log(logistic(self.Y[n] * np.trace(Wmat * self.Xpi[n])))
        return ll
>>>>>>> 6d9e0afd15b1b06156df0a1805e3082f22a53d7f

    def gradient(self):
        for n in range(self.nsamples):
            self.grad +=  self.Y[n] * np.dot(self.Xpi[n].T, logistic(-self.Y[n] *
                                                    np.trace(np.dot(self.Wmat,
                                                                    self.Xpi[n]))))

    def log_l_grad(self, Wmat):
        grad = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        for n in range(self.nsamples):
<<<<<<< HEAD
            grad +=  self.Y[n] * np.dot(self.Xpi[n].T, logistic(-self.Y[n] *
                                                    np.dot(np.trace(Wmat,
                                                                    self.Xpi[n]))))
        grad -= C * Wmat
=======
            grad = grad +  self.Y[n] * self.Xpi[n].T * logistic(-self.Y[n] * np.trace(Wmat * self.Xpi[n]))
>>>>>>> 6d9e0afd15b1b06156df0a1805e3082f22a53d7f
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

class Fobos(object):
    def __init__(self, eta, tau):
        self.eta = eta
        self.tau = tau
        self.iteration = 1
        self.lr = self.eta / np.sqrt(self.iteration)

    def fobos_nn(self, w):
        nu = self.tau * self.lr
        u, s, vt = np.linalg.svd(w)
        sdash = np.maximum(s - nu, 0)
        return (np.matrix(u) * np.matrix(np.diag(sdash) * np.matrix(vt))), s

    def fobos_l1(self, w):
        nu = self.lr * self.tau
        return np.multiply(np.sign(w), np.max(np.abs(w) - nu, 0))

    def fobos_l2(self, w):
        nu = self.lr * self.tau
        return w / (1 + nu)

    def optimize(self, w_k, grad, reg_type='l2'):
        self.lr = self.eta / np.sqrt(self.iteration)
        w_k1 = w_k - self.lr * grad
        if reg_type is 'nn':
            w_k2, s = self.fobos_nn(w_k1)
            norm = np.sum(sum(s))
        elif reg_type is 'l2':
            w_k2 = self.fobos_l2(w_k1)
            norm = np.linalg.norm(w_k, 2)**2  / 2
        elif reg_type is 'l1':
            w_k2 = self.fobos_l1(w_k1)
            norm = np.linalg.norm(w_k, 1)

        self.iteration += 1

        return w_k2, norm

<<<<<<< HEAD
def extdata(pp='in'):
=======
def extdata(datat='train',pp='for'):
>>>>>>> 6d9e0afd15b1b06156df0a1805e3082f22a53d7f
    samples = []
    Vdict = {}
    Ndict = {}
    Mdict = {}
    Y = []

    if datat is 'train':
        sam = [(l.strip().split()[1:5], l.strip().split()[5]) for l in
                open('datasets/cleantrain.txt')]
        hdata = [l.strip() for l in open('datasets/forhead.txt')]
        mdata = [l.strip() for l in open('datasets/formod.txt')]
        hmat = np.matrix(mmread('datasets/trainhw2v.mtx').todense())
        mmat = np.matrix(mmread('datasets/trainmw2v.mtx').todense())
    elif datat is 'dev':
        sam = [(l.strip().split()[1:5], l.strip().split()[5]) for l in
                open('datasets/cleandev.txt')]
        hdata = [l.strip() for l in open('datasets/devheads.txt')]
        mdata = [l.strip() for l in open('datasets/devmods.txt')]
        hmat = np.matrix(mmread('datasets/devhw2v.mtx').todense())
        mmat = np.matrix(mmread('datasets/devmw2v.mtx').todense())

    for s,y in sam:
        if s[2] == pp:
            samples.append(list(s[i] for i in [0,1,3]))
            if y is 'v':
                Y.append(1)
            elif y is 'n':
                Y.append(-1)

    for iter_ in xrange(len(samples)):
        Vdict[iter_] = hmat[hdata.index(samples[iter_][0])]
        Ndict[iter_] = hmat[hdata.index(samples[iter_][1])]
        Mdict[iter_] = mmat[mdata.index(samples[iter_][2])]

    return samples, Vdict, Ndict, Mdict, Y

<<<<<<< HEAD
def extdevdata(pp='in'):
    samples = []
    Vdict = {}
    Ndict = {}
    Mdict = {}
    Y = []
    sam = [(l.strip().split()[1:5], l.strip().split()[5]) for l in
               open('datasets/cleandev.txt')]
    hdata = [l.strip() for l in open('datasets/devheads.txt')]
    mdata = [l.strip() for l in open('datasets/devmods.txt')]
    hmat = np.matrix(mmread('datasets/devhw2v.mtx').todense())
    mmat = np.matrix(mmread('datasets/devmw2v.mtx').todense())
    print len(hdata), hmat.shape
    for s,y in sam:
        if s[2] == pp:
            samples.append(list(s[i] for i in [0,1,3]))
            if y is 'v':
                Y.append(1)
            elif y is 'n':
                Y.append(-1)

    for iter_ in xrange(len(samples)):
        Vdict[iter_] = hmat[hdata.index(samples[iter_][0])]
        Ndict[iter_] = hmat[hdata.index(samples[iter_][1])]
        Mdict[iter_] = mmat[mdata.index(samples[iter_][2])]

    return samples, Vdict, Ndict, Mdict, Y


def training(prep='for', maxiter=100, eta=0.001, tau=0.00001, reg='nn'):
    samples, Vdict, Ndict, Mdict, Y = extdata(pp=prep)
    dsamples, dVdict, dNdict, dMdict, dY = extdevdata(pp=prep)
=======
def main(maxiter, tau, eta):
>>>>>>> 6d9e0afd15b1b06156df0a1805e3082f22a53d7f

    samples, Vdict, Ndict, Mdict, Y = extdata()
    dsamples, dVdict, dNdict, dMdict, dY = extdata(datat='dev')
    operator = Bilnear(samples, Vdict, Ndict, Mdict, Y)
<<<<<<< HEAD
    operator.preprocess()

    doperator = Bilnear(dsamples, dVdict, dNdict, dMdict, dY)
    doperator.preprocess()


    optimizer = Fobos(eta, tau)
    for i in xrange(maxiter):
        start_loop = time()
        operator.grad_init()
        cost = -operator.objective(tau)
        w_k, grad = operator.output()
        w_k, norm = optimizer.optimize(w_k, -grad, reg_type=reg)
        operator.update(w_k, norm)
        doperator.update(w_k, norm)
        end_loop = time()
        print ("%d cost = %f norm =  %f accuracy = %f dev-accuracy = %f time = %f" %
               (i+1, cost, norm, operator.accuracy(), doperator.accuracy(), end_loop - start_loop))
=======
    doperator = Bilnear(dsamples, dVdict, dNdict, dMdict, dY)
    optimizer = Fobos(eta, tau)
    operator.preprocess()
    doperator.preprocess()
    l = (Vdict.values()[0]).shape[1]
    w_k = np.matrix(np.zeros((l,l), dtype=np.float))
    norm = 0
    for i in xrange(maxiter):
        start_loop = time()
        operator.grad_init()
        cost = operator.objective(w_k, tau,norm)
        grad = operator.output(w_k, tau)
        w_k1, norm = optimizer.optimize(w_k, grad)
#        operator.update(w_k, norm)
        end_loop = time()
        print '%d cost=%.2f norm=%.2f tracc=%.2f devacc=%.2f time=%.2f' % (i,
        cost, norm, operator.accuracy(w_k), doperator.accuracy(w_k), end_loop -
                                                                         start_loop)
        w_k = w_k1








>>>>>>> 6d9e0afd15b1b06156df0a1805e3082f22a53d7f



