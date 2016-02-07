#!/usr/bin/python2.7
from __future__ import division
from sklearn import preprocessing
from scipy.io import mmread
from time import time
import scipy.sparse as ss
import numpy as np
from scipy import linalg
from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator


class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=10e-05, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X):
        X = as_float_array(X, copy=self.copy)
        self._mean = np.mean(X, axis=0)
        X -= self._mean
        sigma = np.dot(X.T,X) / X.shape[1]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1 / np.sqrt(S + self.regularization)))
        self._components = np.dot(tmp, U.T)
        return self

    def transform(self, X):
        X_transformed = X - self._mean
        X_transformed = np.dot(X_transformed, self._components.T)
        return X_transformed

class PCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=10e-05, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X):
        X = as_float_array(X, copy=self.copy)
        self.mean = np.mean(X, axis=0)
        X -= self.mean
        sigma = np.dot(X.T, X) / X.shape[1]
        U, S, V = linalg.svd(sigma)
        self.components = np.dot(np.diag(1 / np.sqrt(S + self.regularization)),
                                 U.T)
        return self

    def transform(self, X):
        X_transformed = X - self.mean
        X_transformed = np.dot(X_transformed, self.components)
        return X_transformed


def word_features(l):
    '''
    Extracting features from tokens
    '''
    return (l[0]+l[1]+l[2]+l[3], l[0]+l[1]+l[2], \
    l[1]+l[2]+l[3], l[0]+l[1], l[1]+l[2], l[2]+l[3], \
    l[0]+l[2], l[0]+l[3], l[1]+l[3], l[0], l[1],\
    l[2], l[3])


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

class Combo(object):

    def __init__(self, samples, Vdict, Ndict, Mdict, Xlin, Y, Winit='zeros'):
        self.samples = samples
        self.Vdict = Vdict
        self.Ndict = Ndict
        self.Mdict = Mdict
        self.Xlin = Xlin
        self.ll = None
        self.Y = Y
        self.dim = (Vdict.values()[0]).shape[1]
        self.dimlin = Xlin.shape[1]
        self.nsamples = len(self.samples)
        self.Xi = {}
        self.grad_bil = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        self.grad_lin = np.array(np.zeros((self.dimlin), dtype=np.float))
        if Winit is 'random':
            self.Wmat = np.matrix(np.random.rand(self.dim, self.dim))
        elif Winit is 'zeros':
            self.Wmat = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        elif Winit is 'identity':
            self.Wmat = np.matrix(np.identity(self.dim, dtype=np.float))
        else:
            self.Wmat = Winit
        self.warr = np.zeros(self.dimlin, dtype=np.float)
        self.normlin = np.linalg.norm(self.warr)
        self.norm = np.linalg.norm(self.Wmat)
        self.lsigma = None
        self.rsigma = None
        self.pcaSigma = None
        self.zcaSigma = None
        self.ryotaSigma = None
        self.Xpi = {}

    def grad_init(self):
        self.grad_bil = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        self.grad_lin = np.zeros(self.dim, dtype=np.float)

    def scale(self, v, n, m):
        return np.matrix(preprocessing.scale((m.T*(v-n))))


    def preprocess(self):
        zca = ZCA()
        for s in xrange(self.nsamples):
            self.Xpi[s] = zca.fit_transform(self.scale(self.Vdict[s], self.Ndict[s],
                                        self.Mdict[s]))

    def predict(self,w, x, W, X):
        if logistic(np.dot(w,x) + np.trace(W*X)) > 0.5:
            p = 1
        else:
            p = -1
        return p

    def accuracy_gen(self, warr, x, Wmat, X, Y):
        n_correct = 0
        for i,v in X:
            if self.predict(warr, x, Wmat, v) == Y[i]:
                n_correct += 1
        return n_correct * 1.0 / len(X.keys())

    def accuracy(self, warr, Wmat):
        n_correct = 0
        for i in range(self.nsamples):
            if self.predict( warr, self.Xlin[i], Wmat, self.Xpi[i]) == self.Y[i]:
                n_correct += 1
        return n_correct * 1.0 / self.nsamples

    def log_likelihood(self):
        self.ll = 0
        for n in xrange(self.nsamples):
            self.ll +=  np.log(logistic(self.Y[n] *
                                        (np.dot(self.warr, self.Xlin[n]) + np.trace(np.dot(self.Wmat,self.Xpi[n])))))

    def log_l(self, warr, Wmat):
        ll = 0
        for n in xrange(self.nsamples):
            ll +=  np.log(logistic(self.Y[n] * (np.dot(warr, self.Xlin[n])+ np.trace(Wmat * self.Xpi[n]))))
        return ll

    def gradientbil(self):
        for n in range(self.nsamples):
            self.grad_bil +=  self.Y[n] * np.dot(self.Xpi[n].T, logistic(-self.Y[n] *
                                     (np.dot(self.warr, self.Xlin[n]) + np.trace(np.dot(self.Wmat,
                                                                    self.Xpi[n])))))

    def gradientlin(self):
        for n in range(self.nsamples):
            self.grad_lin +=  self.Y[n] * np.dot(self.Xlin[n], logistic(-self.Y[n] *
                                     (np.dot(self.warr, self.Xlin[n]) + np.trace(np.dot(self.Wmat,
                                                                    self.Xpi[n])))))

    def log_l_grad(self, warr, Wmat):
        gradbil = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        gradlin = np.zeros(self.dimlin, dtype=np.float)
        for n in range(self.nsamples):
            gradlin +=  self.Y[n] * np.dot(self.Xlin[n], logistic(-self.Y[n] *
                                        (np.dot(warr, self.Xlin[n]) + np.trace(np.dot(Wmat,
                                                                    self.Xpi[n])))))
            gradbil +=  self.Y[n] * np.dot(self.Xpi[n].T, logistic(-self.Y[n] *
                                        (np.dot(warr, self.Xlin[n]) + np.trace(np.dot(Wmat,
                                                                    self.Xpi[n])))))
        return gradlin, gradbil

    def objective(self, warr, Wmat, taul, taub, normlin, norm):
        ll = self.log_l(warr, Wmat)
        return - (ll - taub*norm - taub * normlin)

    def logl(self):
        self.log_likelihood()
        return self.ll

    def update(self, w_l, w_k, normlin, norm):
        self.Wmat = w_k
        self.warr = w_l
        self.norm = norm
        self.normlin = normlin

    def output(self, warr, Wmat, taul, taub):
        gradlin, gradbil = self.log_l_grad(warr, Wmat)
        gradbil = gradbil - taub * Wmat
        gradlin = gradlin - taul * warr
        return (- gradlin, - gradbil)

class Fobos(object):
    def __init__(self, etal, etab, taul, taub):
        self.etal = etal
        self.etab = etab
        self.taub = taub
        self.taul = taul
        self.iteration = 1
        self.lrl = self.etal / np.sqrt(self.iteration)
        self.lrb = self.etab / np.sqrt(self.iteration)

    def fobos_nn(self, w):
        nu = self.taub * self.lrb
        u, s, vt = np.linalg.svd(w)
        sdash = np.maximum(s - nu, 0)
        return (np.matrix(u) * np.matrix(np.diag(sdash) * np.matrix(vt))), s

    def fobos_l1(self, w):
        nu = self.lrl * self.taul
        return np.multiply(np.sign(w), np.max(np.abs(w) - nu, 0))

    def fobos_l2(self, w):
        nu = self.lrl * self.taul
        return w / (1 + nu)

    def optimize(self, w_l, w_k, gradlin, gradbil, reg_type='nn'):
        self.lrl = self.etal / np.sqrt(self.iteration)
        self.lrb = self.etab / np.sqrt(self.iteration)
        w_k1 = w_k - self.lrb * gradbil
        w_l1 = w_l - self.lrl * gradlin
        if reg_type is 'nn':
            w_k2, sb = self.fobos_nn(w_k1)
            normbil = np.sum(sum(sb))
            w_l2 = self.fobos_l2(w_l1)
            normlin = np.linalg.norm(w_k, 2)**2  / 2

        elif reg_type is 'l2':
            w_k2 = self.fobos_l2(w_k1)
            normlin = np.linalg.norm(w_k, 2)**2  / 2
        elif reg_type is 'l1':
            w_k2 = self.fobos_l1(w_k1)
            normlin = np.linalg.norm(w_k, 1)

        self.iteration += 1

        return w_l2, w_k2, normlin, normbil

def dataextract(pp='in'):
    '''
    Simple stuff,relatively:
    '''
    trainV = {}
    trainN = {}
    trainM = {}
    traindata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in
                open('datasets/cleantrain.txt') if d.strip().split()[3] == pp]
    trainX = [list(t[0][i] for i in [0,1,3]) for t in traindata]
    trainY = [1 if y[1] == 'v' else -1 for y in traindata]
    tHf = [l.strip() for l in open('datasets/forhead.txt')]
    tMf = [l.strip() for l in open('datasets/formod.txt')]
    trH = np.matrix(mmread('datasets/trainhw2v.mtx').todense())
    trM = np.matrix(mmread('datasets/trainmw2v.mtx').todense())
    for eg in xrange(len(traindata)):
        trainV[eg] = trH[tHf.index(trainX[eg][0])]
        trainN[eg] = trH[tHf.index(trainX[eg][1])]
        trainM[eg] = trM[tMf.index(trainX[eg][2])]
    trainXl = {eg: word_features(traindata[eg][0]) for eg in xrange(len(traindata))}
    featset = []
    for wvals in trainXl.values():
        for val in wvals:
            featset.append(val)
    setfeats = set(featset)
    featset = np.array(sorted(list(setfeats)))
    Xltrain = ss.lil_matrix((len(trainXl), len(featset)))
    for eg, egfeat in trainXl.items():
        onearr = featset.searchsorted(egfeat)
        Xltrain[eg,onearr] = np.ones(len(onearr))


    devV = {}
    devN = {}
    devM = {}
    devdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d
               in open('datasets/cleandev.txt') if d.strip().split()[3] == pp]
    devX = [list(d[0][i] for i in [0,1,3]) for d in devdata]
    devY = [1 if y[1] == 'v' else -1 for y in devdata]
    dHf = [l.strip() for l in open('datasets/devheads.txt')]
    dMf = [l.strip() for l in open('datasets/devmods.txt')]
    deH = np.matrix(mmread('datasets/devhw2v.mtx').todense())
    deM = np.matrix(mmread('datasets/devmw2v.mtx').todense())
    for eg in xrange(len(devdata)):
        devV[eg] = deH[dHf.index(devX[eg][0])]
        devN[eg] = deH[dHf.index(devX[eg][1])]
        devM[eg] = deM[dMf.index(devX[eg][2])]

    devXl = {eg: word_features(devdata[eg][0]) for eg in xrange(len(devdata))}
    Xldev = ss.lil_matrix((len(devXl), len(featset)))
    for deg, degfeat in devXl.items():
        donearr = featset.searchsorted(list(setfeats.intersection(degfeat)))
        Xldev[deg,donearr] = np.ones(len(donearr))


    return trainX, trainY, Xltrain, trainV, trainN, trainM, devX, devY, Xldev, devV, devN, devM


def main(maxiter=100, taul=0.001, etal=0.01, taub=0.0000001, etab=0.001,
         prep='for'):

    trX, trY, trXl, trV, trN, trM, deX, deY, deXl, deV, deN, deM = dataextract(pp=prep)
    operator = Combo(trX, trV, trN, trM, np.array(trXl.todense()), trY)
    doperator = Combo(deX, deV, deN, deM, np.array(deXl.todense()), deY)
    optimizer = Fobos(float(etal), float(etab), float(taul), float(taub))
    operator.preprocess()
    doperator.preprocess()
    bildim = (trV.values()[0]).shape[1]
    lindim = trXl.shape[1]
    print 'Number of Training Examples = ', len(trY), \
        ' Number of Dev Examples = ', len(deY), ' Dimensionality = ', bildim
    w_k = np.matrix(np.zeros((bildim,bildim), dtype=np.float))
    w_l = np.zeros(lindim, dtype=np.float)
    normlin = 0
    normbil = 0
    for i in xrange(int(maxiter)):
        start_loop = time()
        operator.grad_init()
        cost = operator.objective(w_l, w_k, float(taul), float(taub), normlin, normbil)
        gradlin, gradbil = operator.output(w_l, w_k, float(taul), float(taub))
        w_l1, w_k1, normlin, normbil = optimizer.optimize(w_l, w_k, gradlin,
                                                          gradbil)
        operator.update(w_l, w_k, normlin, normbil)
        end_loop = time()
        print '%d cost=%.2f normlin=%.2f normbil=%.2f tracc=%.2f devacc=%.2f time=%.2f' % (i+1,
        cost, normlin, normbil, operator.accuracy(w_l, w_k), doperator.accuracy(w_l, w_k), end_loop -
                                                                         start_loop)
        w_k = w_k1
        w_l = w_l1


if __name__ == '__main__':
    import plac
    plac.call(main)
