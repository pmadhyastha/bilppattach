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

def extract_reps(filename, wordlist):

    fdata = [(w.strip().split()[0], w.strip().split()[1:]) for w
                in open(filename)]
    fdict = dict(fdata)
    dimensions = len(fdict.values()[0])
    fmat = np.zeros((len(wordlist), dimensions), dtype=np.float)
    
    for i, w in enumerate(wordlist):
        if w in fdict: 
            fmat[i] = fdict[w]
    
    return fmat



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

#class PCA(BaseEstimator, TransformerMixin):
#
#   def __init__(self, regularization=10e-05, copy=False):
#       self.regularization = regularization
#       self.copy = copy
#
#   def fit(self, X):
#       X = as_float_array(X, copy=self.copy)
#       self.mean = np.mean(X, axis=0)
#       X -= self.mean
#       sigma = np.dot(X.T, X) / X.shape[1]
#       U, S, V = linalg.svd(sigma)
#       self.components = np.dot(np.diag(1 / np.sqrt(S + self.regularization)),
#                                U.T)
#       return self
#
#   def transform(self, X):
#       X_transformed = X - self.mean
#       X_transformed = np.dot(X_transformed, self.components)
#       return X_transformed
#

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

    def __init__(self, samples, Vdict, Ndict, Pdict, Mdict, Xlin, Y, Winit='zeros'):
        self.samples = samples
        self.Vdict = Vdict
        self.Ndict = Ndict
        self.Pdict = Pdict
        self.Mdict = Mdict
        self.Xlin = Xlin
        self.ll = None
        self.Y = Y
        self.dim1 = (Vdict.values()[0]).shape[1]
        self.dim2 = self.dim1 * self.dim1 
        self.dimlin = Xlin.shape[1]
        self.nsamples = len(self.samples)
        self.Xi = {}
        self.grad_bil = np.matrix(np.zeros((self.dim1, self.dim2), dtype=np.float))
        self.grad_lin = np.array(np.zeros((self.dimlin), dtype=np.float))
        if Winit is 'random':
            self.Wmat = np.matrix(np.random.rand(self.dim1, self.dim2))
        elif Winit is 'zeros':
            self.Wmat = np.matrix(np.zeros((self.dim1, self.dim2), dtype=np.float))
        elif Winit is 'identity':
            self.Wmat = np.matrix(np.identity(self.dim1, dtype=np.float))
        else:
            self.Wmat = Winit
        self.warr = np.zeros(self.dimlin, dtype=np.float)
        self.normlin = np.linalg.norm(self.warr)
        self.norm = np.linalg.norm(self.Wmat)
#       self.lsigma = None
#       self.rsigma = None
#       self.pcaSigma = None
#       self.zcaSigma = None
#       self.ryotaSigma = None

    def grad_init(self):
        self.grad_bil = np.matrix(np.zeros((self.dim1, self.dim2), dtype=np.float))
        self.grad_lin = np.zeros(self.dimlin, dtype=np.float)

    def scale(self, v, n, m):
        return np.matrix(preprocessing.scale((m.T*(v-n))))


    def preprocess(self):
#       zca = ZCA()
        for s in xrange(self.nsamples):
            mod = np.kron(self.Pdict[s], self.Mdict[s])
#           self.Xpi[s] = zca.fit_transform(self.scale(self.Vdict[s], self.Ndict[s],
#                                       self.Mdict[s]))
            self.Xi[s] = [self.Vdict[s], self.Ndict[s], mod] #, mod, ] 

    def predict(self,w, x, W, X):
        if ss.issparse(x):
            x = np.array(x.todense())[0]
        else: 
            x = np.array(x)
        if logistic(np.dot(w,x) + (np.dot(X[0],np.dot(W, X[2].T)) - np.dot(X[1],np.dot(W, X[2].T)))[0,0]) > 0.5:
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
            if self.predict( warr, self.Xlin[i], Wmat, self.Xi[i]) == self.Y[i]:
                n_correct += 1
        return n_correct * 1.0 / self.nsamples

    def log_likelihood(self):
        self.ll = 0
        for n in xrange(self.nsamples):
            self.ll +=  np.log(logistic(self.Y[n] *
                                        (np.dot(self.warr, np.array(self.Xlin[n].todense())[0]) + (np.dot(self.Xi[n][0],np.dot(self.Wmat, self.Xi[n][2].T)) - np.dot(self.Xi[n][1],np.dot(self.Wmat, self.Xi[n][2].T)))[0,0])))

    def log_l(self, warr, Wmat):
        ll = 0
        for n in xrange(self.nsamples):
            ll +=  np.log(logistic(self.Y[n] * (np.dot(warr, np.array(self.Xlin[n].todense())[0])+ (np.dot(self.Xi[n][0],np.dot(Wmat, self.Xi[n][2].T)) - np.dot(self.Xi[n][1],np.dot(Wmat, self.Xi[n][2].T)))[0,0])))
        return ll

    def gradientbil(self):
        for n in range(self.nsamples):
            dif = np.outer(self.Xi[n][0].T, self.Xi[n][2]) - np.outer(self.Xi[n][1].T, self.Xi[n][2])
            self.grad_bil +=  self.Y[n] * dif * logistic(-self.Y[n] *
                                     (np.dot(self.warr, np.array(self.Xlin[n].todense())[0]) + (np.dot(self.Xi[n][0],np.dot(Wmat, self.Xi[n][2].T)) - np.dot(self.Xi[n][1],np.dot(Wmat, self.Xi[n][2].T)))[0,0]))

    def gradientlin(self):
        for n in range(self.nsamples):
            self.grad_lin +=  self.Y[n] * np.dot(np.array(self.Xlin[n].todense())[0], logistic(-self.Y[n] *
                                     (np.dot(self.warr, np.array(self.Xlin[n].todense())[0]) + (np.dot(self.Xi[n][0],np.dot(Wmat, self.Xi[n][2].T)) - np.dot(self.Xi[n][1],np.dot(Wmat, self.Xi[n][2].T)))[0,0])))

    def log_l_grad(self, warr, Wmat):
        gradbil = np.matrix(np.zeros((self.dim1, self.dim2), dtype=np.float))
        gradlin = np.zeros(self.dimlin, dtype=np.float)
        for n in range(self.nsamples):
            gradlin +=  self.Y[n] * np.dot(np.array(self.Xlin[n].todense())[0], logistic(-self.Y[n] * (np.dot(self.warr, np.array(self.Xlin[n].todense())[0]) + (np.dot(self.Xi[n][0],np.dot(Wmat, self.Xi[n][2].T)) - np.dot(self.Xi[n][1],np.dot(Wmat, self.Xi[n][2].T)))[0,0])))
            dif = np.outer(self.Xi[n][0].T, self.Xi[n][2]) - np.outer(self.Xi[n][1].T, self.Xi[n][2])
            gradbil +=  self.Y[n] * dif * logistic(-self.Y[n] *
                                            (np.dot(self.warr, np.array(self.Xlin[n].todense())[0]) + (np.dot(self.Xi[n][0],np.dot(Wmat, self.Xi[n][2].T)) - np.dot(self.Xi[n][1],np.dot(Wmat, self.Xi[n][2].T)))[0,0]))

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
        sdashtemp = np.diag(sdash)
        sdashzeros = np.zeros((u.shape[0], vt.shape[0]), dtype=np.float)
        sdashzeros[:sdashtemp.shape[0], :sdashtemp.shape[1]] = sdashtemp
        return (np.matrix(u) * np.matrix(sdashzeros * np.matrix(vt))), s

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
    trainP = {}
    trainM = {}

    if pp != 'all':
        traindata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in
                    open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/cleantrain.txt') if d.strip().split()[3] == pp]
    else:
        traindata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in
                    open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/cleantrain.txt')]

    trainX = [list(t[0][i] for i in [0,1,2,3]) for t in traindata]
    trainY = [1 if y[1] == 'v' else -1 for y in traindata]

    trainvocab = [w.strip() for w in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/trainvocab.txt')]

    trainMat = extract_reps(filename='/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/train.skipdep.txt', wordlist=trainvocab)
    trainMat = np.matrix(np.hstack([np.matrix(trainMat), np.matrix(np.ones(len(trainMat))).T]))
    for eg in xrange(len(traindata)):
        trainV[eg] = trainMat[trainvocab.index(trainX[eg][0])]
        trainN[eg] = trainMat[trainvocab.index(trainX[eg][1])]
        trainP[eg] = trainMat[trainvocab.index(trainX[eg][2])]
        trainM[eg] = trainMat[trainvocab.index(trainX[eg][3])]

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
    devP = {}
    devM = {}
    if  pp != 'all':
        devdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d
                    in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/cleandev.txt') if d.strip().split()[3] == pp]
    else:
        devdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d
                    in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/cleandev.txt')]

    devX = [list(d[0][i] for i in [0,1,2,3]) for d in devdata]
    devY = [1 if y[1] == 'v' else -1 for y in devdata]

    devvocab = [w.strip() for w in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/devvocab.txt')]
    devMat = extract_reps(filename='/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/dev.skipdep.txt', wordlist=devvocab)
    devMat = np.matrix(np.hstack([np.matrix(devMat), np.matrix(np.ones(len(devMat))).T]))
    for eg in xrange(len(devdata)):
        devV[eg] = devMat[devvocab.index(devX[eg][0])]
        devN[eg] = devMat[devvocab.index(devX[eg][1])]
        devP[eg] = devMat[devvocab.index(devX[eg][2])]
        devM[eg] = devMat[devvocab.index(devX[eg][3])]

    devXl = {eg: word_features(devdata[eg][0]) for eg in xrange(len(devdata))}
    Xldev = ss.lil_matrix((len(devXl), len(featset)))
    for deg, degfeat in devXl.items():
        donearr = featset.searchsorted(list(setfeats.intersection(degfeat)))
        Xldev[deg,donearr] = np.ones(len(donearr))

    return trainX, trainY, Xltrain, trainV, trainN, trainP, trainM, devX, devY, Xldev, devV, devN, devP, devM, featset

def extractTest(pp, featset):
    setfeats = set(featset.tolist())
    testV = {}
    testN = {}
    testP = {}
    testM = {}

    if pp != 'all':
        testdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in
                    open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/cleantest.txt') if d.strip().split()[3] == pp]
    else:
        testdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in
                    open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/cleantest.txt')]

    testX = [list(t[0][i] for i in [0,1,2,3]) for t in testdata]
    testY = [1 if y[1] == 'v' else -1 for y in testdata]
    testvocab = [w.strip() for w in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/testvocab.txt')]
    testMat = extract_reps(filename='/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/test.skipdep.txt', wordlist=testvocab)
    testMat = np.matrix(np.hstack([np.matrix(testMat), np.matrix(np.ones(len(testMat))).T]))
    for eg in xrange(len(testdata)):
        testV[eg] = testMat[testvocab.index(testX[eg][0])]
        testN[eg] = testMat[testvocab.index(testX[eg][1])]
        testP[eg] = testMat[testvocab.index(testX[eg][2])]
        testM[eg] = testMat[testvocab.index(testX[eg][3])]

    testXl = {eg: word_features(testdata[eg][0]) for eg in xrange(len(testdata))}
    Xltest = ss.lil_matrix((len(testXl), len(featset)))
    for teg, tegfeat in testXl.items():
        donearr = featset.searchsorted(list(setfeats.intersection(tegfeat)))
        Xltest[teg,donearr] = np.ones(len(donearr))

    return testX, testY, Xltest, testV, testN, testP, testM

def test(prep, modelBil, modelLin, featset):
    try:
        teX, teY, teXl, teV, teN, teP, teM = extractTest(prep, featset)
        toperator = Combo(teX, teV, teN, teP,teM, teXl.tocsr(), teY)
        toperator.preprocess()
        testacc = toperator.accuracy(modelLin, modelBil)
        print 'Computing test accuracy, ... ' 
        print 'accuracy over test = ',testacc
    except:
        print prep, ' not found ... '



def main(maxiter=100, taul=0.001, etal=0.01, taub=0.0000001, etab=0.001, prep='for'):

    trX, trY, trXl, trV, trN, trP, trM, deX, deY, deXl, deV, deN, deP, deM, featset = dataextract(pp=prep)
    print trXl.shape, deXl.shape
    operator = Combo(trX, trV, trN, trP, trM, trXl.tocsr(), trY)
    if len(deY) != 0:
        doperator = Combo(deX, deV, deN, deP, deM, deXl.tocsr(), deY)
    else:
        doperator = operator
    optimizer = Fobos(float(etal), float(etab), float(taul), float(taub))
    operator.preprocess()
    doperator.preprocess()
    bildim1 = (trV.values()[0]).shape[1]
    bildim2 = bildim1 * bildim1
    lindim = trXl.shape[1]
    print 'Preposition = ', prep, 'Number of Training Examples = ', len(trY), \
        ' Number of Dev Examples = ', len(deY), ' Dimensionality = ', bildim1
    w_k = np.matrix(np.zeros((bildim1,bildim2), dtype=np.float))
    w_l = np.zeros(lindim, dtype=np.float)
    normlin = 0
    normbil = 0
    bestacc = 0.0
    for i in xrange(int(maxiter)):
        start_loop = time()
        operator.grad_init()
        cost = operator.objective(w_l, w_k, float(taul), float(taub), normlin, normbil)
        gradlin, gradbil = operator.output(w_l, w_k, float(taul), float(taub))
        w_l1, w_k1, normlin, normbil = optimizer.optimize(w_l, w_k, gradlin,
                                                          gradbil)
        operator.update(w_l, w_k, normlin, normbil)
        end_loop = time()
        devacc = doperator.accuracy(w_l, w_k)
        if devacc >= bestacc:
            print 'saving model file .... as dev acc is greater by ', devacc - bestacc
            bestacc = devacc
            bestBilmodel = w_k
            bestLinmodel = w_l
            np.save('modelbil'+maxiter+taub+etab+prep+'.npy', bestBilmodel)
            np.save('modellin'+maxiter+taul+etal+prep+'.npy', bestLinmodel)

        print '%d cost=%.2f normlin=%.2f normbil=%.2f tracc=%.2f devacc=%.2f time=%.2f' % (i+1,
        cost, normlin, normbil, operator.accuracy(w_l, w_k), devacc, end_loop -
                                                                         start_loop)
        w_k = w_k1
        w_l = w_l1

    test(prep, bestBilmodel, bestLinmodel, featset)

if __name__ == '__main__':
    import plac
    plac.call(main)

