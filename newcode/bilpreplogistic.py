from __future__ import division
from sklearn import preprocessing
from scipy.io import mmread
from time import time
import numpy as np
from scipy import linalg
from sklearn.decomposition import randomized_svd as rsvd
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


def dataextract(pp='in', wetype='skipdep'):
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

    trainMat = extract_reps(filename='/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/train.'+wetype+'.txt', wordlist=trainvocab)
    trainMat = np.matrix(np.hstack([np.matrix(trainMat), np.matrix(np.ones(len(trainMat))).T]))
    for eg in xrange(len(traindata)):
        trainV[eg] = trainMat[trainvocab.index(trainX[eg][0])]
        trainN[eg] = trainMat[trainvocab.index(trainX[eg][1])]
        trainP[eg] = trainMat[trainvocab.index(trainX[eg][2])]
        trainM[eg] = trainMat[trainvocab.index(trainX[eg][3])]

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
    devMat = extract_reps(filename='/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/dev.'+wetype+'.txt', wordlist=devvocab)
    devMat = np.matrix(np.hstack([np.matrix(devMat), np.matrix(np.ones(len(devMat))).T]))

    for eg in xrange(len(devdata)):
        devV[eg] = devMat[devvocab.index(devX[eg][0])]
        devN[eg] = devMat[devvocab.index(devX[eg][1])]
        devP[eg] = devMat[devvocab.index(devX[eg][2])]
        devM[eg] = devMat[devvocab.index(devX[eg][3])]

    return trainX, trainY, trainV, trainN, trainP, trainM, devX, devY, devV, devN, devP, devM


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

class Bilnear(object):

    def __init__(self, samples, Vdict, Ndict, Pdict, Mdict, Y, Winit='zeros'):
        self.samples = samples
        self.Vdict = Vdict
        self.Ndict = Ndict
        self.Pdict = Pdict
        self.Mdict = Mdict
        self.ll = None
        self.Y = Y
        self.dim1 = (Vdict.values()[0]).shape[1]
        self.dim2 = self.dim1*self.dim1
        self.nsamples = len(self.samples)
        self.Xi = {}
        self.grad = np.matrix(np.zeros((self.dim1, self.dim2), dtype=np.float))
        if Winit is 'random':
            self.Wmat = np.matrix(np.random.rand(self.dim1, self.dim2))
        elif Winit is 'zeros':
            self.Wmat = np.matrix(np.zeros((self.dim1, self.dim2), dtype=np.float))
        else:
            self.Wmat = Winit
        self.norm = np.linalg.norm(self.Wmat)

    def scale(self, v, n, m):
        return np.matrix(preprocessing.scale((m.T*(v-n))))

    def grad_init(self):
        self.grad = np.matrix(np.zeros((self.dim1, self.dim2), dtype=np.float))



    def preprocess(self, ptype='dev'):
        for s in xrange(self.nsamples):
            #if s > 4871:
            mod = np.kron(self.Pdict[s], self.Mdict[s])
            #   self.Xi[s] = [self.Vdict[s], self.Ndict[s], mod, (np.outer(self.Vdict[s].T, mod) - np.outer(self.Ndict[s].T, mod))] 
            self.Xi[s] = [self.Vdict[s], self.Ndict[s], mod] #, mod, ] 

    def predict(self,W, X):
        if logistic(np.dot(X[0],np.dot(W, X[2].T)) - np.dot(X[1],np.dot(W, X[2].T)))> 0.5:
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
            if self.predict(Wmat, self.Xi[i]) == self.Y[i]:
                n_correct += 1
        return n_correct * 1.0 / self.nsamples

    def log_likelihood(self):
        self.ll = 0
        for n in xrange(self.nsamples):
            self.ll +=  np.log(logistic(self.Y[n] *
                                        (np.dot(self.Xi[n][0],np.dot(self.Wmat, self.Xi[n][2].T)) - np.dot(self.Xi[n][1],np.dot(self.Wmat, self.Xi[n][2].T)))))

    def log_l(self, Wmat):
        ll = 0
        for n in xrange(self.nsamples):
            ll +=  np.log(logistic(self.Y[n] * (np.dot(self.Xi[n][0],np.dot(Wmat, self.Xi[n][2].T)) - np.dot(self.Xi[n][1],np.dot(Wmat, self.Xi[n][2].T)))))
        return ll

#   def gradient(self):
#       for n in range(self.nsamples):
#           self.grad +=  self.Y[n] * np.dot(self.Xpi[n].T, logistic(-self.Y[n] *
#                                                   np.trace(np.dot(self.Wmat,
#                                                                   self.Xpi[n]))))

    def log_l_grad(self, Wmat):
        grad = np.matrix(np.zeros((self.dim1, self.dim2), dtype=np.float))
        for n in range(self.nsamples):
            dif = np.outer(self.Xi[n][0].T, self.Xi[n][2]) - np.outer(self.Xi[n][1].T, self.Xi[n][2])
            grad = grad +  self.Y[n] * dif * logistic(-self.Y[n] * (np.dot(self.Xi[n][0],np.dot(Wmat, self.Xi[n][2].T)) - np.dot(self.Xi[n][1],np.dot(Wmat, self.Xi[n][2].T))))[0,0]
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


def dataextractTest(pp='into', wetype='skipdep'):
    '''
    Simple stuff,relatively:
    '''
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

    testMat = extract_reps(filename='/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/test.'+wetype+'.txt', wordlist=testvocab)
    testMat = np.matrix(np.hstack([np.matrix(testMat), np.matrix(np.ones(len(testMat))).T]))

    for eg in xrange(len(testdata)):
        testV[eg] = testMat[testvocab.index(testX[eg][0])]
        testN[eg] = testMat[testvocab.index(testX[eg][1])]
        testP[eg] = testMat[testvocab.index(testX[eg][2])]
        testM[eg] = testMat[testvocab.index(testX[eg][3])]

    return testX, testY, testV, testN, testP, testM


def dataextractTestNYC(pp='into', wetype='skipdep'):
    '''
    Simple stuff,relatively:
    '''
    testV = {}
    testN = {}
    testP = {}
    testM = {}

    if pp != 'all':
        testdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in
                    open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/transfer/nyctestset.txt') if d.strip().split()[3] != pp]
    else:
        testdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in
                    open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/transfer/nyctestset.txt')]

    testX = [list(t[0][i] for i in [0,1,2,3]) for t in testdata]
    testY = [1 if y[1] == 'v' else -1 for y in testdata]
    testvocab = [w.strip() for w in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/transfer/nyctestsetvocab.txt')]

    testMat = extract_reps(filename='/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/transfer/nyctestset.'+wetype+'.txt', wordlist=testvocab)
    testMat = np.matrix(np.hstack([np.matrix(testMat), np.matrix(np.ones(len(testMat))).T]))

    for eg in xrange(len(testdata)):
        testV[eg] = testMat[testvocab.index(testX[eg][0])]
        testN[eg] = testMat[testvocab.index(testX[eg][1])]
        testP[eg] = testMat[testvocab.index(testX[eg][2])]
        testM[eg] = testMat[testvocab.index(testX[eg][3])]

    return testX, testY, testV, testN, testP, testM


def dataextractTestWIKI(pp='into', wetype='skipdep'):
    '''
    Simple stuff,relatively:
    '''
    testV = {}
    testN = {}
    testP = {}
    testM = {}

    if pp != 'all':
        testdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in
                    open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/transfer/wkptest.txt') if d.strip().split()[3] != pp]
    else:
        testdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in
                    open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/transfer/wkptest.txt')]

    testX = [list(t[0][i] for i in [0,1,2,3]) for t in testdata]
    testY = [1 if y[1] == 'v' else -1 for y in testdata]
    testvocab = [w.strip() for w in open('/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/transfer/wkptestvocab.txt')]

    testMat = extract_reps(filename='/home/usuaris/pranava/acl2016/shorts/ppattach/newcode/datasets/transfer/wkptestvocab.'+wetype+'.txt', wordlist=testvocab)
    testMat = np.matrix(np.hstack([np.matrix(testMat), np.matrix(np.ones(len(testMat))).T]))

    for eg in xrange(len(testdata)):
        testV[eg] = testMat[testvocab.index(testX[eg][0])]
        testN[eg] = testMat[testvocab.index(testX[eg][1])]
        testP[eg] = testMat[testvocab.index(testX[eg][2])]
        testM[eg] = testMat[testvocab.index(testX[eg][3])]

    return testX, testY, testV, testN, testP, testM

class Fobos(object):
    def __init__(self, eta, tau):
        self.eta = eta
        self.tau = tau
        self.iteration = 1
        self.lr = self.eta / np.sqrt(self.iteration)

    def fobos_nn(self, w):
        nu = self.tau * self.lr
        u, s, vt = rsvd(w, w.shape[0])
        sdash = np.maximum(s - nu, 0)
        sdashtemp = np.diag(sdash)
        sdashzeros = np.zeros(u.shape, dtype=np.float)
        sdashzeros[:sdashtemp.shape[0], :sdashtemp.shape[1]] = sdashtemp
        return (np.matrix(u) * np.matrix(sdashzeros * np.matrix(vt))), s

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


def test(prep, model, wetyp):

    teX, teY, teV, teN, teP, teM = dataextractTest(pp=prep, wetype=wetyp)
    toperator = Bilnear(teX, teV, teN, teP, teM, teY)
    toperator.preprocess()
    testacc = toperator.accuracy(model)
    print 'Computing test accuracy, ... ' 
    print 'accuracy over test = ',testacc
    return testacc

def testNYC(prep, model, wetyp):

    teX, teY, teV, teN, teP, teM = dataextractTestNYC(pp=prep, wetype=wetyp)
    toperator = Bilnear(teX, teV, teN, teP, teM, teY)
    toperator.preprocess()
    testacc = toperator.accuracy(model)
    print 'Computing test accuracy, ... ' 
    print 'accuracy over test = ',testacc
    return testacc

def testWIKI(prep, model, wetyp):

    teX, teY, teV, teN, teP, teM = dataextractTestWIKI(pp=prep, wetype=wetyp)
    toperator = Bilnear(teX, teV, teN, teP, teM, teY)
    toperator.preprocess()
    testacc = toperator.accuracy(model)
    print 'Computing test accuracy, ... ' 
    print 'accuracy over test = ',testacc
    return testacc



def main(maxiter=10, tau=0.01, eta=0.01, prep='into', we='skipdep', model=None):
#   buffsize = 0
#   foutfile = open('output.txt', 'w', buffsize)
    trX, trY, trV, trN, trP, trM, deX, deY, deV, deN, deP, deM = dataextract(pp=prep, wetype=we)
    operator = Bilnear(trX, trV, trN, trP, trM, trY)
    if len(deY) != 0:
        doperator = Bilnear(deX, deV, deN, deP, deM, deY)
    else:
        doperator = operator
    optimizer = Fobos(float(eta), float(tau))
    operator.preprocess(ptype='train')
    doperator.preprocess()
    l = (trV.values()[0]).shape[1]
    m = l*l
    print 'Preposition =', prep, 'Number of Training Examples = ', len(trY), \
        ' Number of Dev Examples = ', len(deY), ' Dimensionality = ', l
    if model:
        w_k = np.load(model)
    else:
        w_k = np.matrix(np.zeros((l,m), dtype=np.float))

    print 'here', w_k.shape
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
            np.save('model'+str(maxiter)+str(tau)+str(eta)+prep+'.npy', bestmodel)

        print '%d cost=%.2f norm=%.2f tracc=%.2f devacc=%.2f time=%.2f' % (i+1,
        cost, norm, operator.accuracy(w_k), devacc, end_loop - start_loop)
        w_k = w_k1
    test(prep, bestmodel, wetyp=we)

if __name__ == '__main__':
    import plac
    plac.call(main)

