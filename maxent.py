#!/usr/bin/python

from __future__ import print_function, unicode_literals, division
import numpy
import math
import nltk
from collections import defaultdict


def calculate_empirical_fcount(train_toks, encoding):
    fcount = numpy.zeros(encoding.length(), 'd')
    for tok, label in train_toks:
        for (index, val) in encoding.encode(tok, label):
            fcount[index] += val
#    print (numpy.sum(fcount))
    return fcount

def proximal_op(matrix, nu):
        return np.sign(matrix) * np.maximum(np.abs(matrix) - nu, 0.)


def proximal_l2(matrix, nu):
    return ((1./(1.+nu)) * matrix)


def calculate_estimated_fcount(classifier, train_toks, encoding):
    fcount = numpy.zeros(encoding.length(), 'd')
    for tok, label in train_toks:
        pdist = classifier.prob_classify(tok)
        for label in pdist.samples():
            prob = pdist.prob(label)
            for (fid, fval) in encoding.encode(tok, label):
#                print (fval, 'fval')
                fcount[fid] += prob*fval

    for tok, label in train_toks:
        pdist = classifier.prob_classify(tok)
    print(pdist.prob('n'), pdist.prob('v'))

    return fcount


def log_likelihood(classifier, gold):
    results = classifier.batch_prob_classify([fs for (fs,l) in gold])
    ll = [pdist.prob(l) for ((fs, l), pdist) in zip(gold, results)]
    return math.log(float(sum(ll))/len(ll))


def accuracy(classifier, gold):
    results = classifier.batch_classify([fs for (fs, l) in gold])
    correct = [l == r for ((fs,l),r) in zip(gold, results)]
    if correct:
        return float(sum(correct))/len(correct)
    else:
        return 0

class Maxent(object):

    def __init__(self, encoding, weights, logarthmic=True):

        self._encoding = encoding
        self._weights = weights
        self._logarithmic = logarthmic
        assert encoding.length() == len(weights)

    def labels(self):
        return self._encoding.labels()

    def set_weights(self, new_weights):

        self._weights = new_weights
        assert (self._encoding.length() == len(new_weights))

    def weights(self):
        return self._weights

    def classify(self, featureset):
        return self.prob_classify(featureset).max()

    def batch_classify(self, featuresets):
        return [self.classify(fs) for fs in featuresets]

    def batch_prob_classify(self, featuresets):
        return [self.prob_classify(fs) for fs in featuresets]

    def prob_classify(self, featureset):
        prob_dict = {}
        for label in self._encoding.labels():
            feature_vector = self._encoding.encode(featureset, label)
            if self._logarithmic:
                total = 0.0
                for (f_id, f_val) in feature_vector:
                    total += self._weights[f_id] * f_val
#                    print (label, total)
                prob_dict[label] = total
#        return DictProbDist(prob_dict, log=self._logarithmic, normalize=True)
        return nltk.DictionaryProbDist(prob_dict, log=self._logarithmic, normalize=True)


    def bin_classify(self, headmat, modmat, weightmat):
        return 'NotImplemented'

    @classmethod
    def train(cls, train_toks, algorithm=None, trace=3, encoding=None, labels=None, max_iter=10, LC=100, tau=1.0, norm=None, devset=None, eta=1):
        if algorithm == 'gis':
            return train_maxent_classifier_with_gis(train_toks, encoding, labels, max_iter, devset, eta)
        if algorithm == 'gd':
            return train_maxent_classifier_with_gd(train_toks, encoding, labels, max_iter, LC, tau, norm, devset, eta)

class BinaryMaxentFeatureEncoding(object):
    def __init__(self, labels, mapping):

        self._labels = list(labels)
        self._mapping = mapping
        self._length = len(mapping)

    def encode(self, featureset, label):
        encoding = []
        for fname, fval in featureset.items():
            if (fname, fval, label) in self._mapping:
                encoding.append((self._mapping[fname, fval, label], 1))
        return encoding

    def mapping(self):
        return self._mapping
    def labels(self):
        return self._labels

    def length(self):
        return self._length

    @classmethod
    def train(cls, train_toks, count_cutoff=0,labels=None ):
        mapping = {}
        seen_labels = set()
        count = defaultdict(int)
        for (tok, label) in train_toks:
            if labels and label not in labels:
                raise ValueError('val err...')
            seen_labels.add(label)
            for (fname, fval) in tok.items():
                count[fname, fval] += 1
                if count[fname, fval] >= count_cutoff:
                    if (fname, fval, label) not in mapping:
                        mapping[fname, fval, label] = len(mapping)
        if labels is None:
            labels = seen_labels
        return cls(labels, mapping)

class GISEncoding(BinaryMaxentFeatureEncoding):

    def __init__(self, labels, mapping, C=None):
        BinaryMaxentFeatureEncoding.__init__(self, labels, mapping)
        if C is None:
            C = len(set(fname for (fname, fval, label) in mapping)) + 1
        self._C = C

    @property
    def C(self):
        return self._C

    def encode(self, featureset, label):
        encoding = BinaryMaxentFeatureEncoding.encode(self, featureset, label)
        base_length = BinaryMaxentFeatureEncoding.length(self)
        total = sum(v for (f,v) in encoding)
        if total >= self._C:
            raise ValueError('err in C')
        encoding.append( (base_length, self._C-total))
        return encoding

    def length(self):
        return BinaryMaxentFeatureEncoding.length(self) + 1

def train_maxent_classifier_with_gis(train_toks, encoding, labels, max_iter, devset):
    if encoding == None:
        encoding = GISEncoding.train(train_toks, labels=labels)

    Cinv = 1.0/encoding.C
    empirical_fcount = calculate_empirical_fcount(train_toks, encoding)
    unattested = set(numpy.nonzero(empirical_fcount==0)[0])
    weights = numpy.zeros(len(empirical_fcount), 'd')
    for fid in unattested: weights[fid] = numpy.NINF
    classifier = Maxent(encoding, weights)

    log_empirical_fcount = numpy.log2(empirical_fcount)

    print (' ----- Training (%d iterations)'% max_iter )
    print ('-------------------------------------------------------')
    itr = 0
    tr = []
    trl = []
    dr = []
    try:
        while True:
            itr += 1
            ll = -log_likelihood(classifier, train_toks)
            acc = accuracy(classifier, train_toks)
            dacc = accuracy(classifier, devset)
            print ('%9d   %14.5f  %9.3f, %9.3f' %(itr, ll, acc, dacc))
            tr.append(acc)
            trl.append(ll)
            dr.append(dacc)

            estimated_fcount = calculate_estimated_fcount(classifier, train_toks, encoding)
            for fid in unattested: estimated_fcount[fid] += 1
            log_estimated_fcount = numpy.log2(estimated_fcount)
            weights = classifier.weights()
            weights -= -(log_empirical_fcount - log_estimated_fcount) * Cinv
            classifier.set_weights(weights)
            if itr >= max_iter:
                break
    except:
        raise

    return classifier, tr, trl, dr


def train_maxent_classifier_with_gd(train_toks, encoding, labels, max_iter, LC, tau,  norm, devset, eta):

    encoding = BinaryMaxentFeatureEncoding.train(train_toks, labels=labels)
    empirical_fcount = calculate_empirical_fcount(train_toks, encoding)
    weights = numpy.zeros(len(empirical_fcount), 'd')
    weights_x = numpy.zeros(len(empirical_fcount), 'd')

    classifier = Maxent(encoding, weights)
    mapping = encoding.mapping()
    print ('------Training (%d iterations----------)' % max_iter)
    print ('----------------------------------------------------------')
    tr = []
    trl = []
    dr = []
    itr = 0
    lam_k = 1
    try:
        while True:
            itr += 1
            ll = -log_likelihood(classifier, train_toks)
            acc = accuracy(classifier, train_toks)
            estimated_fcount = calculate_estimated_fcount(classifier, train_toks, encoding)
            grad = -(empirical_fcount - estimated_fcount) / len(train_toks)
            dacc = accuracy(classifier, devset)
            weights = classifier.weights()
            lam_kp1 = float(1 + numpy.sqrt(1 + 4*(lam_k**2 ))) / 2
            normsum = (numpy.linalg.norm(weights, ord=2)**2)
            if norm == None:
                weights -= (eta * grad) / numpy.sqrt(itr)
                print ('%9d   %14.5f  %14.5f  %9.3f  %9.3f' %(itr, normsum, ll, acc, dacc))
                tr.append(acc)
                trl.append(ll)
                dr.append(dacc)

            elif norm == 'l1':

                normsum = (numpy.linalg.norm(weights, ord=1)**2)
                nu = 1 / LC
                temp_wy = weights - grad / LC
                weights_xp1 = proximal_op(temp_wvy, nu)
                lr = (lam_k - 1) / lam_kp1
                weights_yp1 = weights_xp1 + lr * (weights_xp1 - weights_x)
                weights_x = weights_xp1
                weights = weights_yp1
                print ('%9d   %14.5f  %9.3f, %9.3f' %(itr, ll+(tau*normsum), acc, dacc))
                tr.append(acc)
                trl.append(ll)
                dr.append(dacc)

            elif norm == 'l2':
                if devset == None:
                    raise ValueError('no devset provided')
                    break
                weights -= eta*(grad + numpy.multiply(tau, weights))/numpy.sqrt(itr)
                print ('%9d   %14.5f  %9.3f  %9.3f' %(itr, ll, acc, dacc))
                tr.append(acc)
                trl.append(ll)
                dr.append(dacc)


            elif norm == 'l2p':

                normsum = (numpy.linalg.norm(weights, ord=2)**2)
                nu = 1 / LC
                temp_wy = weights - (tau * grad) / LC
                weights_xp2 = proximal_l2(temp_wvy, nu)
                lr = (lam_k - 1) / lam_kp1
                weights_yp1 = weights_xp1 + lr * (weights_xp1 - weights_x)
                weights_x = weights_xp1
                weights = weights_yp1
                print ('%9d   %14.5f  %9.3f, %9.3f' %(itr, ll+(tau*normsum), acc, dacc))
                tr.append(acc)
                trl.append(ll)
                dr.append(dacc)


            classifier.set_weights(weights)
            if itr >= max_iter:
                break
    except:
        raise
    return classifier, tr, trl, dr, empirical_fcount, mapping
