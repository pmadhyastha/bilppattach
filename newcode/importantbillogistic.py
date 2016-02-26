from __future__ import division
from sklearn import preprocessing
from scipy.io import mmread
from time import time
import numpy as np
from scipy import linalg
from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator
import sys
sys.stdout.flush()


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
        self.dim = (Vdict.values()[0]).shape[1]
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
                self.Xi[iter_]  = xi
                self.rsigma += xi * xi.T
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

    def get_sigma(self,sigtype='pca'):
        if self.pcaSigma:
            return self.pcaSigma
        else:
            self._pcaSigma()
            return self.pcaSigma

    def scale(self, v, n, m):
        return np.matrix(preprocessing.scale((m.T*(v-n))))


    def preprocess(self, sigmatype='pca'):
        if sigmatype is 'pca':
            #zca = ZCA()
            if self.pcaSigma is None:
                self._pcaSigma()
            for i,xi in self.Xi.items():
        #        self.Xpi[i] = self.pcaSigma * xi
                #self.Xpi[i] = pca.fit_transform(xi)
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
        else:
            self._pcaSigma()
            self.Xpi = self.Xi

    def preprocessVal(self, Xi, sigmatype=None, lsigma=None, rsigma=None):
         if sigmatype is 'pca':
            for i,xi in Xi.items():
                self.Xpi[i] = rsigma * xi
         elif sigmatype is 'zca':
            for i,xi in Xi.items():
                self.Xpi[i] = self.zcaSigma * xi
         elif sigmatype is 'ryota':
            for i,xi in Xi.items():
                self.Xpi[i] = rsigma * xi * lsigma
         else:
            self._pcaSigma()
            self.Xpi = self.Xi

    def predict(self,W, X):
        if logistic(np.trace(W*X)) > 0.5:
            p = 1
        else:
            p = -1
        return p

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
            ll +=  np.log(logistic(self.Y[n] * np.trace(Wmat * self.Xpi[n])))
        return ll

    def gradient(self):
        for n in range(self.nsamples):
            self.grad +=  self.Y[n] * np.dot(self.Xpi[n].T, logistic(-self.Y[n] *
                                                    np.trace(np.dot(self.Wmat,
                                                                    self.Xpi[n]))))

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

    def optimize(self, w_k, grad, reg_type='nn'):
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

def dataextract(pp='in'):
    '''
    Simple stuff,relatively:
    '''
    trainV = {}
    trainN = {}
    trainM = {}
    traindata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in
                open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/cleantrain.txt') if d.strip().split()[3] == pp]
    trainX = [list(t[0][i] for i in [0,1,3]) for t in traindata]
    trainY = [1 if y[1] == 'v' else -1 for y in traindata]
    tHf = [l.strip() for l in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/forhead.txt')]
    tMf = [l.strip() for l in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/formod.txt')]
    trH = np.matrix(mmread('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/trainhw2v.mtx').todense())
    trH = np.matrix(np.hstack([trH, np.matrix(np.ones(len(trH))).T]))
    trM = np.matrix(mmread('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/trainmw2v.mtx').todense())
    trM = np.matrix(np.hstack([trM, np.matrix(np.ones(len(trM))).T]))
    for eg in xrange(len(traindata)):
        trainV[eg] = trH[tHf.index(trainX[eg][0])]
        trainN[eg] = trH[tHf.index(trainX[eg][1])]
        trainM[eg] = trM[tMf.index(trainX[eg][2])]


    devV = {}
    devN = {}
    devM = {}
    devdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d
               in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/cleandev.txt') if d.strip().split()[3] == pp]
    devX = [list(d[0][i] for i in [0,1,3]) for d in devdata]
    devY = [1 if y[1] == 'v' else -1 for y in devdata]
    dHf = [l.strip() for l in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/devheads.txt')]
    dMf = [l.strip() for l in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/devmods.txt')]
    deH = np.matrix(mmread('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/devhw2v.mtx').todense())
    deH = np.matrix(np.hstack([deH, np.matrix(np.ones(len(deH))).T]))
    deM = np.matrix(mmread('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/devmw2v.mtx').todense())
    deM = np.matrix(np.hstack([deM, np.matrix(np.ones(len(deM))).T]))
    for eg in xrange(len(devdata)):
        devV[eg] = deH[dHf.index(devX[eg][0])]
        devN[eg] = deH[dHf.index(devX[eg][1])]
        devM[eg] = deM[dMf.index(devX[eg][2])]

    return trainX, trainY, trainV, trainN, trainM, devX, devY, devV, devN, devM


def dataextractTest(pp='in'):
    '''
    Simple stuff,relatively:
    '''

    testV = {}
    testN = {}
    testM = {}
    testdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d
               in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/cleantest.txt') if d.strip().split()[3] == pp]
    testX = [list(d[0][i] for i in [0,1,3]) for d in testdata]
    testY = [1 if y[1] == 'v' else -1 for y in testdata]
    tHf = [l.strip() for l in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/testheads.txt')]
    tMf = [l.strip() for l in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/testmods.txt')]
    teH = np.matrix(mmread('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/testhw2v.mtx').todense())
    teH = np.matrix(np.hstack([teH, np.matrix(np.ones(len(teH))).T]))
    teM = np.matrix(mmread('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/testmw2v.mtx').todense())
    teM = np.matrix(np.hstack([teM, np.matrix(np.ones(len(teM))).T]))
    for eg in xrange(len(testdata)):
        testV[eg] = teH[tHf.index(testX[eg][0])]
        testN[eg] = teH[tHf.index(testX[eg][1])]
        testM[eg] = teM[tMf.index(testX[eg][2])]

    return testX, testY, testV, testN, testM


def test(prep, model):

    teX, teY, teV, teN, teM = dataextractTest(pp=prep)
    toperator = Bilnear(teX, teV, teN, teM, teY)
    toperator.preprocess()
    testacc = toperator.accuracy(model)
    print 'Computing test accuracy, ... ' 
    print 'accuracy over test = ',testacc


def main(maxiter=10, tau=0.01, eta=0.01, prep='into'):
#   buffsize = 0
#   foutfile = open('output.txt', 'w', buffsize)
    trX, trY, trV, trN, trM, deX, deY, deV, deN, deM = dataextract(pp=prep)
    operator = Bilnear(trX, trV, trN, trM, trY)
    if len(deY) != 0:
        doperator = Bilnear(deX, deV, deN, deM, deY)
    else:
        doperator = operator
    optimizer = Fobos(float(eta), float(tau))
    operator.preprocess()
    doperator.preprocess()
    l = (trV.values()[0]).shape[1]
    print 'Preposition =', prep, 'Number of Training Examples = ', len(trY), \
        ' Number of Dev Examples = ', len(deY), ' Dimensionality = ', l
    w_k = np.matrix(np.zeros((l,l), dtype=np.float))
    norm = 0
    bestacc = 0.0
    for i in xrange(int(maxiter)):
        start_loop = time()
        operator.grad_init()
        cost = operator.objective(w_k, float(tau), norm)
        grad = operator.output(w_k, float(tau))
        w_k1, norm = optimizer.optimize(w_k, grad)
        operator.update(w_k, norm)
        end_loop = time()
        devacc = doperator.accuracy(w_k)
        if devacc >= bestacc:
            print 'saving model file .... as dev acc is greater by ', devacc - bestacc
            bestacc = devacc
            bestmodel = w_k
            np.save('model'+maxiter+tau+eta+prep+'.npy', bestmodel)

        print '%d cost=%.2f norm=%.2f tracc=%.2f devacc=%.2f time=%.2f' % (i+1,
        cost, norm, operator.accuracy(w_k), devacc, end_loop - start_loop)
        w_k = w_k1
    test(prep, bestmodel)
#       foutfile.flush()
#   foutfile.close()


if __name__ == '__main__':
    import plac
    plac.call(main)

