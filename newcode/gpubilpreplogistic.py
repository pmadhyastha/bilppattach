from __future__ import division
from sklearn import preprocessing
from scipy.io import mmread
from scipy import linalg
from time import time
import numpy as np
import numbers
#import fbpca as fb
import gnumpy as gnp
from scipy import linalg
#from sklearn.decomposition import randomized_svd as rsvd
from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator
import sys
sys.stdout.flush()


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u, v : ndarray
        u and v are the output of `linalg.svd` or
        `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
        so one can compute `np.dot(u * s, v)`.
    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, xrange(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[xrange(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def randomized_range_finder(A, size, n_iter=2,
                            power_iteration_normalizer='auto',
                            random_state=None):
    """Computes an orthonormal matrix whose range approximates the range of A.

    Parameters
    ----------
    A: 2D array
        The input data matrix.

    size: integer
        Size of the return array

    n_iter: integer
        Number of power iterations used to stabilize the result

    power_iteration_normalizer: 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter`<=2 and switches to LU otherwise.

        .. versionadded:: 0.18

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance

    Returns
    -------
    Q: 2D array
        A (size x size) projection matrix, the range of which
        approximates well the range of the input matrix A.

    Notes
    -----

    Follows Algorithm 4.3 of
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) http://arxiv.org/pdf/0909.4061

    An implementation of a randomized algorithm for principal component
    analysis
    A. Szlam et al. 2014
    """
    random_state = check_random_state(random_state)

    # Generating normal random vectors with shape: (A.shape[1], size)
    Q = random_state.normal(size=(A.shape[1], size))

    # Deal with "auto" mode
    if power_iteration_normalizer == 'auto':
        if n_iter <= 2:
            power_iteration_normalizer = 'none'
        else:
            power_iteration_normalizer = 'LU'

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for i in range(n_iter):
        if power_iteration_normalizer == 'none':
            Q = gnp.dot(gnp.garray(A), gnp.garray(Q)).as_numpy_array()
            Q = gnp.dot(gnp.garray(A.T),gnp.garray(Q)).as_numpy_array()
        elif power_iteration_normalizer == 'LU':
            Q, _ = linalg.lu(safe_sparse_dot(A, Q), permute_l=True)
            Q, _ = linalg.lu(safe_sparse_dot(A.T, Q), permute_l=True)
        elif power_iteration_normalizer == 'QR':
            Q, _ = linalg.qr(safe_sparse_dot(A, Q),  mode='economic')
            Q, _ = linalg.qr(safe_sparse_dot(A.T, Q),  mode='economic')

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = linalg.qr(gnp.dot(gnp.garray(A), gnp.garray(Q)).as_numpy_array(),  mode='economic')
    return Q



def randomized_svd(M, n_components, n_oversamples=10, n_iter=2,
                   power_iteration_normalizer='auto', transpose='auto',
                   flip_sign=True, random_state=0):
    """
    Grabbed from sklearn for gpu optimization
    
    Computes a truncated randomized SVD

    Parameters
    ----------
    M: ndarray or sparse matrix
        Matrix to decompose

    n_components: int
        Number of singular values and vectors to extract.

    n_oversamples: int (default is 10)
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.

    n_iter: int (default is 2)
        Number of power iterations (can be used to deal with very noisy
        problems).

        .. versionchanged:: 0.18

    power_iteration_normalizer: 'auto' (default), 'QR', 'LU', 'none'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter`<=2 and switches to LU otherwise.

        .. versionadded:: 0.18

    transpose: True, False or 'auto' (default)
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.

        .. versionchanged:: 0.18

    flip_sign: boolean, (True by default)
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance to make behavior

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components.

    References
    ----------
    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions
      Halko, et al., 2009 http://arxiv.org/abs/arXiv:0909.4061

    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert

    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    """
    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if transpose == 'auto':
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        M = M.T

    Q = randomized_range_finder(M, n_random, n_iter,
                                power_iteration_normalizer, random_state)

    # project M to the (k + p) dimensional space using the basis vectors
#   B = safe_sparse_dot(Q.T, M)
    B = gnp.dot(gnp.garray(Q.T), gnp.garray(M)).as_numpy_array()

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = linalg.svd(B, full_matrices=False)
    del B
#   U = np.dot(Q, Uhat)
    U = gnp.dot(gnp.garray(Q), gnp.garray(Uhat)).as_numpy_array()

    if flip_sign:
        if not transpose:
            U, V = svd_flip(U, V)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, V = svd_flip(U, V, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return V[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], V[:n_components, :]

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

class Bilinear(object):

    def __init__(self, samples, Vdict, Ndict, Pdict, Mdict, Y, Winit='zeros', mtype='concat'):
        self.samples = samples
        self.Vdict = Vdict
        self.Ndict = Ndict
        self.Pdict = Pdict
        self.Mdict = Mdict
        self.ll = None
        self.Y = Y
        self.dim1 = (Vdict.values()[0]).shape[1]
        if mtype == 'concat':
            self.dim2 = self.dim1 + self.dim1
        elif mtype == 'average': 
            self.dim2 = self.dim1 
        else:
            self.dim2 = self.dim1 * self.dim1
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
            #   self.Xi[s] = [self.Vdict[s], self.Ndict[s], mod, (np.outer(self.Vdict[s].T, mod) - np.outer(self.Ndict[s].T, mod))] 
            self.Xi[s] = [self.Vdict[s], self.Ndict[s], mod] #, mod, ] 
#           print s
#           self.gXi[s] = [gnp.garray(np.array(self.Vdict[s])), gnp.garray(np.array(self.Ndict[s])), gnp.garray(np.array(mod))] #, mod, ] 

    def predict(self,W, X):
        if logistic(np.dot(X[0],np.dot(W, X[2].T)) - np.dot(X[1],np.dot(W, X[2].T)))> 0.5:
            p = 1
        else:
            p = -1
        return p

    def gpredict(self,W, X):
        if logistic((gnp.dot(X[0],gnp.dot(W, X[2].T)) - gnp.dot(X[1],gnp.dot(W, X[2].T))).as_numpy_array()[0])> 0.5:
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
            if self.gpredict(Wmat, self.Xi[i]) == self.Y[i]:
                n_correct += 1
        return n_correct * 1.0 / self.nsamples

#   def log_likelihood(self):
#       self.ll = 0
#       for n in xrange(self.nsamples):
#           self.ll +=  np.log(logistic(self.Y[n] *
#                                       (gnp.dot(self.gXi[n][0],gnp.dot(self.Wmat, self.gXi[n][2].T)) - gnp.dot(self.gXi[n][1],gnp.dot(self.Wmat, self.gXi[n][2].T)))))

    def log_l(self, Wmat):
        ll = 0
        for n in xrange(self.nsamples):
            ll +=  np.log(logistic(self.Y[n] * (np.dot(self.Xi[n][0],np.dot(Wmat, self.Xi[n][2].T)) - np.dot(self.Xi[n][1],np.dot(Wmat, self.Xi[n][2].T)))))
        return ll



    def log_l_new(self, Wmat):
        ll = 0
        n_correct = 0
        for n in xrange(self.nsamples):
            internals = (np.dot(self.Xi[n][0],np.dot(Wmat, self.Xi[n][2].T)) - np.dot(self.Xi[n][1],np.dot(Wmat, self.Xi[n][2].T)))

            if logistic(internals) > 0.5 and self.Y[n] == 1:
                n_correct += 1
            elif logistic(internals) < 0.5 and self.Y[n] == -1:
                n_correct += 1

            ll +=  np.log(logistic(self.Y[n] * internals))

        return ll, n_correct / self.nsamples


    def glog_l_new(self, Wmat):
        ll = 0
        n_correct = 0
        gWmat = gnp.garray(np.array(Wmat))
        for n in xrange(self.nsamples):
            #print self.Xi[n][0].shape, self.Xi[n][1].shape, self.Xi[n][2].shape, gWmat.shape
            internals = (gnp.dot(gnp.garray(self.Xi[n][0]),gnp.dot(gWmat, gnp.garray(self.Xi[n][2].T))) - gnp.dot(gnp.garray(self.Xi[n][1]),gnp.dot(gWmat, gnp.garray(self.Xi[n][2].T)))).as_numpy_array()[0]

            if logistic(internals) > 0.5 and self.Y[n] == 1:
                n_correct += 1
            elif logistic(internals) < 0.5 and self.Y[n] == -1:
                n_correct += 1

            ll +=  np.log(logistic(self.Y[n] * internals))

        return ll, n_correct / self.nsamples

#   def gradient(self):
#       for n in range(self.nsamples):
#           self.grad +=  self.Y[n] * np.dot(self.Xpi[n].T, logistic(-self.Y[n] *
#                                                   np.trace(np.dot(self.Wmat,
#                                                                   self.Xpi[n]))))

    def log_l_grad(self, Wmat):
        grad = np.matrix(np.zeros((self.dim1, self.dim2), dtype=np.float))
        ggrad = gnp.garray(grad)
        for n in range(self.nsamples):
            dif = np.outer(self.Xi[n][0].T, self.Xi[n][2]) - np.outer(self.Xi[n][1].T, self.Xi[n][2])
            grad = grad +  self.Y[n] * dif * logistic(-self.Y[n] * (np.dot(self.Xi[n][0],np.dot(Wmat, self.Xi[n][2].T)) - np.dot(self.Xi[n][1],np.dot(Wmat, self.Xi[n][2].T))))[0,0]
        return grad

    def log_l_grad_stoch(self, Wmat, coin, st):
        grad = np.matrix(np.zeros((self.dim1, self.dim2), dtype=np.float))
        for s in range(st):
            n = coin[s]
            dif = np.outer(self.Xi[n][0].T, self.Xi[n][2]) - np.outer(self.Xi[n][1].T, self.Xi[n][2])
            grad = grad +  self.Y[n] * dif * logistic(-self.Y[n] * (np.dot(self.Xi[n][0],np.dot(Wmat, self.Xi[n][2].T)) - np.dot(self.Xi[n][1],np.dot(Wmat, self.Xi[n][2].T))))[0,0]
        return grad


    def glog_l_grad_stoch(self, Wmat, coin, st):
        grad = np.zeros((self.dim1, self.dim2), dtype=np.float)
        ggrad = gnp.garray(grad)
        gWmat = gnp.garray(Wmat)
        for s in range(st):
            n = coin[s]
            dif = gnp.outer(gnp.garray(self.Xi[n][0].T), gnp.garray(self.Xi[n][2])) - gnp.outer(gnp.garray(self.Xi[n][1].T), gnp.garray(self.Xi[n][2]))
            ggrad = ggrad +  self.Y[n] * dif * logistic(-self.Y[n] * (gnp.dot(gnp.garray(self.Xi[n][0]),gnp.dot(gWmat, gnp.garray(self.Xi[n][2].T))) - gnp.dot(gnp.garray(self.Xi[n][1]),gnp.dot(gWmat, gnp.garray(self.Xi[n][2].T)))).as_numpy_array())[0,0]
        return ggrad.as_numpy_array()


    def objective(self, Wmat, tau, norm):
        ll, tracc = self.glog_l_new(Wmat)
        return -(ll - tau*norm), ll, tracc

#   def logl(self):
#       self.log_likelihood()
#       return self.ll

    def update(self, w_k, norm):
        self.Wmat = w_k
        self.norm = norm

    def output(self, Wmat, tau):
        grad = self.log_l_grad(Wmat)
        grad = grad - tau * Wmat
        return - grad

    def output_stoch(self, Wmat, tau, st):
        coin = np.array(range(self.nsamples))
        np.random.shuffle(coin)
        grad = self.glog_l_grad_stoch(Wmat, coin, st)
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
        u, s, vt = randomized_svd(w, w.shape[0])
        sdash = np.maximum(s - nu, 0)
        sdashtemp = np.diag(sdash)
        sdashzeros = np.zeros(u.shape, dtype=np.float)
        sdashzeros[:sdashtemp.shape[0], :sdashtemp.shape[1]] = sdashtemp
        return gnp.dot(gnp.garray(u), gnp.dot(gnp.garray(sdashzeros), gnp.garray(vt))).as_numpy_array(), s
#       return (np.matrix(u) * np.matrix(sdashzeros * np.matrix(vt))), s

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


def test(prep, model, wetyp, mtpe):

    teX, teY, teV, teN, teP, teM = dataextractTest(pp=prep, wetype=wetyp)
    toperator = Bilinear(teX, teV, teN, teP, teM, teY)
    toperator.preprocess(mtype=mtpe)
    testacc = toperator.accuracy(model)
    print 'Computing test accuracy, ... ' 
    print 'accuracy over test = ',testacc
    return testacc

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
    trX, trY, trV, trN, trP, trM, deX, deY, deV, deN, deP, deM = dataextract(pp=prep, wetype=we)
    operator = Bilinear(trX, trV, trN, trP, trM, trY, mtype=mtpe)
    if len(deY) != 0:
        doperator = Bilinear(deX, deV, deN, deP, deM, deY, mtype=mtpe)
    else:
        doperator = operator
    optimizer = Fobos(float(eta), float(tau))
    operator.preprocess(mtype=mtpe)
    doperator.preprocess(mtype=mtpe)
    l = (trV.values()[0]).shape[1]
    if mtpe == 'concat':
        m = l + l
    elif mtpe == 'average': 
        m = (l + l) / 2.0 
    else:
        m = l * l
    print 'Preposition =', prep, 'Number of Training Examples = ', len(trY), \
        ' Number of Dev Examples = ', len(deY), ' Dimensionality = ', l, 'Stochastic Mini-Batch = ', st,  \
        'M type = ', mtpe
    if model:
        w_k = np.load(model)
    else:
        w_k = np.matrix(np.zeros((l,m), dtype=np.float))
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
    test(prep, bestmodel, wetyp=we, mtpe=mtpe)
    test(prep, w_k, wetyp=we, mtpe=mtpe)

if __name__ == '__main__':
    import plac
    plac.call(main)

