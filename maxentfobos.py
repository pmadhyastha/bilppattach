#!/usr/bin/python

from __future__ import print_function, division
import traceback
import numpy as np
import  sys, traceback
from collections import defaultdict


def calculate_empirical_fcount(train_toks, encoding):

    fcount_n = np.zeros(encoding.length_n(), 'd')
    fcount_v = np.zeros(encoding.length_v(), 'd')

    for tok, label in train_toks:
        if label == 'n':
            for (index, val) in encoding.encode(tok, label):
                fcount_n[index] += val

        if label == 'v':
            for (index, val) in encoding.encode(tok, label):
                fcount_v[index] += val

    return fcount_n, fcount_v


def calculate_estimated_fcount(classifier, train_toks, encoding):
    fcount_n = np.zeros(encoding.length_n(), 'd')
    fcount_v = np.zeros(encoding.length_v(), 'd')

    for tok, l in train_toks:
        pdist = classifier.prob_classify(tok)

        for label in pdist.samples():
            prob = pdist.prob(label)
            if label == 'n':
                for (fid, fval) in encoding.encode(tok, label):
                    fcount_n[fid] += prob * fval
            elif label == 'v':
                for (fid, fval) in encoding.encode(tok, label):
                    fcount_v[fid] += prob * fval

    return fcount_n, fcount_v


class probdist(object):

    def __init__(self, prob_dict):
        self._prob_dict = prob_dict.copy()

    def prob(self, label):
        return (np.exp(self._prob_dict[label]) /
                np.sum([np.exp(val) for val in self._prob_dict.values()]))

    def max(self):
#        if self._prob_dict['n'] > self._prob_dict['v']:
#            return 'n'
#        elif self._prob_dict['v'] > self._prob_dict['n']:
#            return 'v'
#        elif self._prob_dict['v']  == self._prob_dict['n']:
#            return 'e'

        if not hasattr(self, '_max'):
            self._max = max((p, v) for (v, p) in self._prob_dict.items())[1]
        return self._max

    def samples(self):
        return self._prob_dict.keys()


def log_likelihood(classifier, gold):
    results = classifier.batch_prob_classify([fs for (fs, l) in gold])
    ll = [pdist.prob(l) for ((fs, l), pdist) in zip(gold, results)]
    return float(np.sum(np.log(ll))) / len(ll)


def accuracy(classifier, gold):
    results = classifier.batch_classify([fs for (fs, l) in gold])
    correct = [l == r for ((fs, l), r) in zip(gold, results)]
    equal = [r == 'e' for ((fs, l), r) in zip(gold, results)]

    if sum(equal):
        print ('equal results = ', float(sum(equal) * 100) / len(equal), '%')

    if correct:
        return float(sum(correct))/len(correct)

    else:
        return 0


class Maxent(object):

    def __init__(self, encoding, weights_n, weights_v, logarthmic=True):

        self._encoding = encoding
        self._weights_n = weights_n
        self._weights_v = weights_v
        self._logarithmic = logarthmic
        assert encoding.length_n() == len(weights_n)
        assert encoding.length_v() == len(weights_v)

    def labels(self):
        return self._encoding.labels()

    def set_weights(self, new_weights_n, new_weights_v):

        self._weights_n = new_weights_n
        self._weights_v = new_weights_v

        assert (self._encoding.length_n() == len(new_weights_n))
        assert (self._encoding.length_v() == len(new_weights_v))

    def weights_n(self):
        return self._weights_n

    def weights_v(self):
        return self._weights_v

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
            total = 0.0
            if label == 'n':
                for (f_id, f_val) in feature_vector:
                    total += self._weights_n[f_id] * f_val
            elif label == 'v':
                for (f_id, f_val) in feature_vector:
                    total += self._weights_v[f_id] * f_val

            prob_dict[label] = total

        return probdist(prob_dict)

    def bin_classify(self, headmat, modmat, weightmat):
        return 'NotImplemented'

    @classmethod
    def train(cls, train_toks, algorithm=None, encoding=None, labels=None,
              max_iter=10, LC=100, tau=1.0, norm=None, devset=None, eta=1):
        if algorithm == 'gd':
            return train_maxent_classifier_with_gd(train_toks, encoding,
                                                   labels, max_iter, LC,
                                                   tau, norm, devset, eta)


class BinaryMaxentFeatureEncoding(object):
    def __init__(self, labels, mapping_n, mapping_v):

        self._labels = list(labels)
        self._mapping_n = mapping_n
        self._mapping_v = mapping_v
        self._length_n = len(mapping_n)
        self._length_v = len(mapping_v)

    def mapping_n(self):
        return self._mapping_n

    def mapping_v(self):
        return self._mapping_v

    def encode(self, featureset, label):
        encoding = []
        if label == 'n':
#            print ('here')
            for fname, fval in featureset.items():
                if (fname, fval) in self._mapping_n:
                    encoding.append((self._mapping_n[fname, fval], 1))

        if label == 'v':
            for fname, fval in featureset.items():
                if (fname, fval) in self._mapping_v:
                    encoding.append((self._mapping_v[fname, fval], 1))

        return encoding

    def labels(self):
        return self._labels

    def length_n(self):
        return self._length_n

    def length_v(self):
        return self._length_v

    @classmethod
    def train(cls, train_toks, count_cutoff=0, labels=None):
        mapping_n = {}
        mapping_v = {}
        seen_labels = set()
        count = defaultdict(int)
        for (tok, label) in train_toks:
            if labels and label not in labels:
                raise ValueError('val err...')
            seen_labels.add(label)
            for (fname, fval) in tok.items():
                count[fname, fval] += 1
                if count[fname, fval] >= count_cutoff:
                    if label == 'n':
                        if (fname, fval) not in mapping_n:
                            mapping_n[fname, fval] = len(mapping_n)
                    elif label == 'v':
                        if (fname, fval) not in mapping_v:
                            mapping_v[fname, fval] = len(mapping_v)

        if labels is None:
            labels = seen_labels
        return cls(labels, mapping_n, mapping_v)


def proximal_op(matrix, nu):
        return np.sign(matrix) * np.maximum(np.abs(matrix) - nu, 0.)


def proximal_l2(matrix, nu):
    return ((1./(1.+nu)) * matrix)


def train_maxent_classifier_with_gd(train_toks, encoding, labels,
                                    max_iter, LC, tau,  norm, devset, eta):

    encoding = BinaryMaxentFeatureEncoding.train(train_toks, labels=labels)

    emp_n, emp_v = calculate_empirical_fcount(train_toks, encoding)
    dprev = 0
    bestwts = []
    weights_n = np.zeros(len(emp_n), 'd')
    weights_v = np.zeros(len(emp_v), 'd')

    weights_xn = np.zeros(len(emp_n), 'd')
    weights_xv = np.zeros(len(emp_v), 'd')

    classifier = Maxent(encoding, weights_n, weights_v)

    mapping_n = encoding.mapping_n()
    mapping_v = encoding.mapping_v()

    print ('------Training (%d iterations----------)' % max_iter)
    print ('----------------------------------------------------------')
    norm_list = []
    tr = []
    trl = []
    dr = []
    ob = []
    itr = 0

    lam_k = 1
    try:
        while True:
            itr += 1

            ll = -log_likelihood(classifier, train_toks)
            acc = accuracy(classifier, train_toks)

            est_n, est_v = calculate_estimated_fcount(classifier,
                                                      train_toks, encoding)

            grad_n = -(emp_n - est_n) / len(train_toks)
            grad_v = -(emp_v - est_v) / len(train_toks)

            dacc = accuracy(classifier, devset)

            weights_n = classifier.weights_n()
            weights_v = classifier.weights_v()

            lam_kp1 = float(1 + np.sqrt(1 + 4 * (lam_k**2))) / 2

            norm_n2 = (np.linalg.norm(weights_n, ord=2)**2)
            norm_v2 = (np.linalg.norm(weights_v, ord=2)**2)

            norm_n2p = (np.linalg.norm(weights_n, ord=2))
            norm_v2p = (np.linalg.norm(weights_v, ord=2))

            normsum_l2 = (norm_n2 + norm_v2)
            normsum_l2p = (norm_n2p + norm_v2p)

            if norm is None:
                weights_n -= (eta * grad_n) / np.sqrt(itr)
                weights_v -= (eta * grad_v) / np.sqrt(itr)
                print ('%9d   %14.5f   %14.5f  %9.3f  %9.3f'
                       % (itr, ll, normsum_l2p, acc, dacc))
                tr.append(acc)
                norm_list.append(normsum_l2p)
                trl.append(ll)
                dr.append(dacc)
                ob.append(ll)

            elif norm == 'l1':

                norm_n1 = (np.linalg.norm(weights_n, ord=1))
                norm_v1 = (np.linalg.norm(weights_v, ord=1))

                normsum_l1 = (norm_n1 + norm_v1)
                nu = tau / LC

                temp_wvy = weights_v - grad_v / LC
                temp_wny = weights_n - grad_n / LC

#                weights_xvp1 = np.where(temp_wvy > 0,
#                                        np.maximum(temp_wvy-nu, 0),
#                                        np.minimum(temp_wvy+nu, 0))

#                weights_xnp1 = np.where(temp_wny > 0,
#                                         np.maximum(temp_wny-nu, 0),
#                                         np.minimum(temp_wny+nu, 0))

                weights_xvp1 = proximal_op(temp_wvy, nu)
                weights_xnp1 = proximal_op(temp_wny, nu)

                lr = (lam_k - 1) / lam_kp1

                weights_yvp1 = weights_xvp1 + lr * (weights_xvp1 - weights_xv)
                weights_ynp1 = weights_xnp1 + lr * (weights_xnp1 - weights_xn)

                weights_xn = weights_xnp1
                weights_xv = weights_xvp1

                weights_n = weights_ynp1
                weights_v = weights_yvp1

                obj = ll + (tau * normsum_l1)

                print ('%9d   %14.5f %14.5f  %14.5f    %9.3f, %9.3f'
                       % (itr, ll, obj, normsum_l1, acc, dacc))
                tr.append(acc)
                norm_list.append(normsum_l1)
                trl.append(ll)
                dr.append(dacc)
                ob.append(obj)

            elif norm == 'l2proximal':

                nu = tau / LC

                temp_wvy = weights_v - grad_v / LC
                temp_wny = weights_n - grad_n / LC

                weights_xvp1 = proximal_l2(temp_wvy, nu)
                weights_xnp1 = proximal_l2(temp_wny, nu)

                lr = (lam_k - 1) / lam_kp1

                weights_yvp1 = weights_xvp1 + lr * (weights_xvp1 - weights_xv)
                weights_ynp1 = weights_xnp1 + lr * (weights_xnp1 - weights_xn)

                weights_xn = weights_xnp1
                weights_xv = weights_xvp1

                weights_n = weights_ynp1
                weights_v = weights_yvp1

                obj = ll + (tau * normsum_l2p)

                print ('%9d   %14.5f %14.5f  %14.5f    %9.3f, %9.3f'
                       % (itr, ll, obj, normsum_l2p, acc, dacc))
                tr.append(acc)
                norm_list.append(normsum_l2p)
                trl.append(ll)
                dr.append(dacc)
                ob.append(obj)


            elif norm == 'l2f':
                Eta = LC * ( 1 / itr)

                obj = ll + ((tau / 2) * normsum_l2)

                weights_n -= ((np.multiply(Eta, grad_n)) / (1 + (tau * Eta)))
                weights_v -= ((np.multiply(Eta, grad_v)) / (1 + (tau * Eta)))

                print ('%9d   %14.5f %14.5f  %14.5f  %9.3f  %9.3f'
                       % (itr, ll, obj, normsum_l2, acc, dacc))

                tr.append(acc)
                norm_list.append(normsum_l2)
                trl.append(ll)
                dr.append(dacc)
                ob.append(obj)

            elif norm == 'l2':

                if devset is None:
                    raise ValueError('no devset provided')

                obj = ll + (tau/2 * normsum_l2)
                weights_n -= eta*(grad_n + np.dot(tau, weights_n))/np.sqrt(itr)
                weights_v -= eta*(grad_v + np.dot(tau, weights_v))/np.sqrt(itr)

                print ('%9d   %14.5f %14.5f  %14.5f  %9.3f  %9.3f'
                       % (itr, ll, obj, normsum_l2, acc, dacc))

                tr.append(acc)
                norm_list.append(normsum_l2)
                trl.append(ll)
                dr.append(dacc)
                ob.append(obj)
            prev_wts = [classifier.weights_n(), classifier.weights_v()]
            classifier.set_weights(weights_n, weights_v)
            if dprev < dacc:
                dprev = dacc
                bestwts = prev_wts

            if itr >= max_iter:
                break
    except:
#        raise
        traceback.print_exc()

    return classifier, tr, ob, dr, bestwts, dprev , norm_list
