from __future__ import division
from sklearn import preprocessing
from scipy.io import mmread
from scipy import linalg
from time import time
import numpy as np
import numbers
import gnumpy as gnp
from scipy import linalg
#from sklearn.decomposition import randomized_svd as rsvd
from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator
import sys
sys.stdout.flush()


PATH = 'ppdata'

class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=10e-05, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X):
        X = as_float_array(X, copy=self.copy)
        self._mean = np.mean(X, axis=0)
        X -= self._mean
        sigma = np.dot(X.T,X) / X.shape[1]
        U, S, V = linalg.svd(sigma, full_matrices=0, compute_uv=1)
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
        else:
            fmat[i] = fdict['*UNKNOWN*'] 

    return fmat

def read_samples(filename, pp='all'):
    data = [d.strip().split() for d in open(filename)]

    if pp != 'all':
        data = filter(lambda x: x[0] == pp, data)


    X = [ (t[0], t[1], t[4:]) for t in data]
    # Y has the correct index (already shifted to 0-origin
    Y = [ int(t[3])-1 for t in data]

    return X, Y


def dataextract(pp='in', wetype='skipdep'):

    Ftrain = 'wsj.2-21.txt.dep.pp.full'
    Fdev = 'wsj.23.txt.dep.pp.full'

    trainX, trainY = read_samples(PATH + '/' + Ftrain, pp)


    trainVN = {}
    trainP = {}
    trainM = {}

#   trainvocab = [w.strip() for w in open(PATH + '/trainvocab.txt')]
    trainvocab = [w.strip() for w in open(PATH + '/wsj.2-21.txt.dep.pp.vocab')]
#   trainMat = extract_reps(filename=PATH + '/train.'+wetype+'.txt', wordlist=trainvocab)
    trainMat = extract_reps(filename=PATH + '/emb.skip.w2vtxt.d50.m1.w1.s0', wordlist=trainvocab)

    trainMat = np.array(np.hstack([np.matrix(trainMat), np.matrix(np.ones(len(trainMat))).T]))
    position = np.zeros(7)
    for eg in xrange(len(trainX)):

        cands = len(trainX[eg][2])
        trainVN[eg] = np.zeros((cands, (trainMat.shape[1]*7)))
        pos = 7 - cands
        for cand in xrange(cands): 
            pos1 = pos + cand
            postemp = position
            postemp[pos1] = 1.0
#           trainVN[eg][cand] = np.matrix(np.hstack([trainMat[trainvocab.index(trainX[eg][2][cand])],postemp]))
            trainVN[eg][cand] = np.matrix(np.kron(trainMat[trainvocab.index(trainX[eg][2][cand])], postemp))

        trainP[eg] = trainMat[trainvocab.index(trainX[eg][0])]
        trainM[eg] = trainMat[trainvocab.index(trainX[eg][1])]

    devVN = {}
    devP = {}
    devM = {}
    devX, devY = read_samples(PATH + '/' + Fdev, pp)

#   devvocab = [w.strip() for w in open(PATH + '/devvocab.txt')]
    devvocab = [w.strip() for w in open(PATH + '/wsj.23.txt.dep.pp.vocab')]
#   devMat = extract_reps(filename=PATH + '/dev.'+wetype+'.txt', wordlist=devvocab)
    devMat = extract_reps(filename=PATH + '/emb.skip.w2vtxt.d50.m1.w1.s0', wordlist=devvocab)
    devMat = np.array(np.hstack([np.matrix(devMat), np.matrix(np.ones(len(devMat))).T]))

    for eg in xrange(len(devX)):

        dcands = len(devX[eg][2])
        dpos = 7 - dcands
        devVN[eg] = np.zeros((dcands, (devMat.shape[1]*7)))
        for dcand in xrange(dcands):
            dpos1 = dpos + dcand
            dpostemp = position
            dpostemp[dpos1] = 1.0
            devVN[eg][dcand] = np.matrix(np.kron((devMat[devvocab.index(devX[eg][2][dcand])]), dpostemp))

        devP[eg] = devMat[devvocab.index(devX[eg][0])]
        devM[eg] = devMat[devvocab.index(devX[eg][1])]

    return trainX, trainY, trainVN, trainP, trainM, devX, devY, devVN, devP, devM


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

class Bilinear(object):

    def __init__(self, samples, VNdict, Pdict, Mdict, Y, Winit='random', mtype='concat'):
        self.samples = samples
        self.VNdict = VNdict
        self.Pdict = Pdict
        self.Mdict = Mdict
        self.ll = None
        self.Y = Y
        self.dim1 = (VNdict.values()[0]).shape[1]
        self.dim1temp = self.dim1 / 7
        if mtype == 'concat':
            self.dim2 = self.dim1temp + self.dim1temp
        elif mtype == 'average':
            self.dim2 = self.dim1temp
        else:
            self.dim2 = self.dim1temp * self.dim1temp
        self.nsamples = len(self.samples)
        self.Xi = {}
        self.gXi = {}
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


    def preprocess(self, ptype='dev', mtype='concat'):
        for s in xrange(self.nsamples):
            #if s > 4871:
            if mtype == 'concat':
                mod = np.concatenate((self.Pdict[s][0], self.Mdict[s][0]), axis=1)
            elif mtype == 'average':
                mod = (self.Pdict[s] + self.Mdict[s]) / 2.0
            else:
                mod = np.kron(self.Pdict[s], self.Mdict[s])
            self.Xi[s] = [self.VNdict[s], mod] 

    def gpredict(self,W, X):
        # scores has a score for each possible attachment head
        scores = gnp.dot(X[0], gnp.dot(W, X[1].T))
        # predicted head
        phead = scores.argmax(axis=0) 
        return phead

    def accuracy_gen(self, Wmat, X, Y):
        n_correct = 0
        for i,v in X:
            if self.predict(Wmat, v) == Y[i]:
                n_correct += 1
        return n_correct * 1.0 / len(X.keys())

    # DONE CHANGE
    def accuracy(self, Wmat):
        n_correct = 0
        for i in range(self.nsamples):
            if self.gpredict(Wmat, self.Xi[i]) == self.Y[i]:
                n_correct += 1
        return n_correct * 1.0 / self.nsamples

    def glog_l_new(self, Wmat):
        ll = 0
        n_correct = 0
        gWmat = gnp.garray(np.array(Wmat))
        for n in xrange(self.nsamples):
            S = gnp.dot(self.Xi[n][0], gnp.dot(Wmat, self.Xi[n][1].T))
            # update number of correct predictions
            #phead = S.T[0].argmax(axis=0)
            phead = S.argmax(axis=0)
            if phead == self.Y[n]: n_correct += 1
            # normalizer
            #logz = np.log(np.sum(np.exp(S.T[0].as_numpy_array())))
            logz = np.log(np.sum(np.exp(S.as_numpy_array())))
            # add log probability of the observed head
            ll +=  S[self.Y[n]] - logz
        return ll / self.nsamples, n_correct / self.nsamples

    # CHANGE THIS
    def glog_l_grad_stoch(self, Wmat, coin, st):
        grad = np.zeros((self.dim1, self.dim2), dtype=np.float)
        ggrad = gnp.garray(grad)
        gWmat = gnp.garray(Wmat)
        for s in range(st):
            n = coin[s]
            # scores
            #expS = np.exp((gnp.dot(self.Xi[n][0], gnp.dot(Wmat, self.Xi[n][1].T))).T[0].as_numpy_array()) 
            expS = np.exp((gnp.dot(self.Xi[n][0], gnp.dot(Wmat, self.Xi[n][1].T))).as_numpy_array()) 
            # normalizer
            z = np.sum(expS)
            # probability distribution
            P = expS / z

            num_heads = self.Xi[n][0].shape[0]
            for j in xrange(num_heads):
                coeff = 1 if j==self.Y[n] else 0
                ggrad = ggrad + (coeff-P[j])*(gnp.outer(gnp.garray(self.Xi[n][0][j].T), gnp.garray(self.Xi[n][1]))).as_numpy_array()
        return ggrad.as_numpy_array()

    # CHECK THIS
    def objective(self, Wmat, tau, norm):
        ll, tracc = self.glog_l_new(Wmat)
        return (-ll + tau*norm), ll, tracc

    def update(self, w_k, norm):
        self.Wmat = w_k
        self.norm = norm

    def output(self, Wmat, tau):
        grad = self.log_l_grad(Wmat)
        grad = grad - tau * Wmat
        return - grad

    # CHECK THIS
    def output_stoch(self, Wmat, tau, st):
        coin = np.array(range(self.nsamples))
        #np.random.shuffle(coin)
        grad = self.glog_l_grad_stoch(Wmat, coin, st)
        grad = (grad ) / st
        #grad = (grad - tau * Wmat) / st
#       grad = (grad - tau * Wmat) 
        return - grad

class Adagrad(object):
    def __init__(self, x0, stepsize, fudge_factor,):
        self.ss = stepsize
        self.x0 = x0
        self.ff = fudge_factor
        self.gti = np.zeros(np.shape(self.x0))

    def optimize(self, w_k, grad): 
        print grad.shape
        grad = np.array(grad)
        self.gti += grad ** 2
        adjusted_grad = grad / (self.ff + np.sqrt(self.gti))
        w_k -= self.ss * adjusted_grad
        return w_k, np.linalg.norm(w_k)


class Fobos(object):
    def __init__(self, eta, tau):
        self.eta = eta
        self.tau = tau
        self.iteration = 1
        self.lr = self.eta / np.sqrt(self.iteration)

    def fobos_nn(self, w):
        nu = self.tau * self.lr
        u, s, vt = linalg.svd(w, full_matrices=0, compute_uv=1)
        sdash = np.maximum(s - nu, 0)
        sdashzeros = np.diag(sdash)
#       sdashzeros = np.zeros(u.shape, dtype=np.float)
#       sdashzeros[:sdashtemp.shape[0], :sdashtemp.shape[1]] = sdashtemp
        return gnp.dot(gnp.garray(u), gnp.dot(gnp.garray(sdashzeros), gnp.garray(vt))).as_numpy_array(), s

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


def test(prep, model, wetyp, mtpe):
    devVN = {}
    devP = {}
    devM = {}
    devX, devY = read_samples(PATH + '/' + Fdev, pp)

#   devvocab = [w.strip() for w in open(PATH + '/devvocab.txt')]
    devvocab = [w.strip() for w in open(PATH + '/wsj.23.txt.dep.pp.vocab')]
#   devMat = extract_reps(filename=PATH + '/dev.'+wetype+'.txt', wordlist=devvocab)
    devMat = extract_reps(filename=PATH + '/emb.skip.w2vtxt.d50.m1.w1.s0', wordlist=devvocab)
    devMat = np.matrix(np.hstack([np.matrix(devMat), np.matrix(np.ones(len(devMat))).T]))

    for eg in xrange(len(devX)):

        dcands = len(devX[eg][2])
        devVN[eg] = np.zeros((dcands, devMat.shape[1]))
        for dcand in xrange(dcands):
            devVN[eg][dcand] = devMat[devvocab.index(devX[eg][2][dcand])]

        devP[eg] = devMat[devvocab.index(devX[eg][0])]
        devM[eg] = devMat[devvocab.index(devX[eg][1])]

    doperator = Bilinear(deX, deVN, deP, deM, deY, mtype=mtpe)
    doperator.preprocess(mtype=mtpe)
    w_k = np.load(model)
    print 'Computing test accuracy, ... '
    dacc = doperator.accuracy(model)
    print 'accuracy over test = ',dacc

#   teX, teY, teV, teN, teP, teM = dataextractTest(pp=prep, wetype=wetyp)
#   toperator = Bilinear(teX, teV, teN, teP, teM, teY)
#   toperator.preprocess(mtype=mtpe)
#   testacc = toperator.accuracy(model)
#   print 'Computing test accuracy, ... '
#   print 'accuracy over test = ',testacc
#   return testacc

def testNYC(prep, model, wetyp):

    teX, teY, teV, teN, teP, teM = dataextractTestNYC(pp=prep, wetype=wetyp)
    toperator = Bilinear(teX, teV, teN, teP, teM, teY)
    toperator.preprocess()
    testacc = toperator.accuracy(model)
    print 'Computing test accuracy, ... '
    print 'accuracy over test = ',testacc
    return testacc

def testWIKI(prep, model, wetyp):

    teX, teY, teV, teN, teP, teM = dataextractTestWIKI(pp=prep, wetype=wetyp)
    toperator = Bilinear(teX, teV, teN, teP, teM, teY)
    toperator.preprocess()
    testacc = toperator.accuracy(model)
    print 'Computing test accuracy, ... '
    print 'accuracy over test = ',testacc
    return testacc



def main(maxiter=10, tau=0.0001, eta=0.0001, prep='all', we='w2v100', st=100, model=None, mtpe='non'):
#   buffsize = 0
#   foutfile = open('output.txt', 'w', buffsize)
    trX, trY, trVN, trP, trM, deX, deY, deVN, deP, deM = dataextract(pp=prep, wetype=we)
    operator = Bilinear(trX, trVN, trP, trM, trY, mtype=mtpe)
    if model == 'None':
        model = None
    if len(deY) != 0:
        doperator = Bilinear(deX, deVN, deP, deM, deY, mtype=mtpe)
    else:
        doperator = operator
    optimizer = Fobos(float(eta), float(tau))
    operator.preprocess(mtype=mtpe)
    doperator.preprocess(mtype=mtpe)
    l = (trVN.values()[0]).shape[1]
    ltemp = l / 7 
    if mtpe == 'concat':
        m = ltemp + ltemp
    elif mtpe == 'average':
        m = (ltemp + ltemp) / 2.0
    else:
        m = ltemp * ltemp
    print 'Preposition =', prep, 'Number of Training Examples = ', len(trY), \
        ' Number of Dev Examples = ', len(deY), ' Dimensionality = ', l, 'Stochastic Mini-Batch = ', st,  \
        'M type = ', mtpe
    if model:
        w_k = np.load(model)
    else:
        w_k = np.matrix(np.zeros((l,m), dtype=np.float))
#   optimizer = Adagrad(w_k, stepsize=1e-2, fudge_factor=1e-6) 
#       w_k = np.matrix(np.random.rand(l,m ))
#   print 'here', w_k.shape
    norm = 0
    bestacc = 0.0
    for i in xrange(int(maxiter)):
        start_loop = time()
        operator.grad_init()
        cost, ll, tracc = operator.objective(w_k, float(tau), norm)
        if st:
            grad = operator.output_stoch(w_k, float(tau), int(st))
        else:
            grad = operator.output(w_k, float(tau))
        w_k1, norm = optimizer.optimize(w_k, grad)
        operator.update(w_k, norm)
        end_loop = time()
        devacc = doperator.accuracy(w_k)
        if devacc >= bestacc:
            print 'saving model file .... as dev acc is greater by ', devacc - bestacc
            bestacc = devacc
            bestmodel = w_k
            np.save('model'+str(maxiter)+str(tau)+str(eta)+prep+'st'+str(st)+'.npy', bestmodel)
        print '%d cost=%.4f objective=%.4f norm=%.4f tracc=%.4f devacc=%.4f time=%.2f' % (i+1,
        ll, cost, norm, tracc, devacc, end_loop - start_loop)
        w_k = w_k1
    np.save('model-fin'+str(maxiter)+str(tau)+str(eta)+prep+'st'+str(st)+'.npy', w_k)
#   test(prep, bestmodel, wetyp=we, mtpe=mtpe)
#   test(prep, w_k, wetyp=we, mtpe=mtpe)

if __name__ == '__main__':
    import plac
    plac.call(main)

