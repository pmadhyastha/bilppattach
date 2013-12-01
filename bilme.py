#!/usr/bin/python
from __future__ import division
import numpy as np
import scipy.misc as sm

def extract_pptype(train_toks, pptype):
    extracted = []
    for (tok, label) in train_toks:
        if tok[2] == pptype:
            extracted.append((tok, label))
    return extracted

def accuracy(encoding, classifier, gold, pptype):
    encoded = encoding.encode(gold, pptype)
    score_n = np.exp(np.array(np.dot(encoded[1], np.dot(classifier.weight_n(), encoded[2].transpose()))))
    score_v = np.exp(np.array(np.dot(encoded[0], np.dot(classifier.weight_v(), encoded[2].transpose()))))
    total =  0
    correct = 0
    aclist = [(lambda x,y: 'n' if x > y else 'v')(x,y) for (x,y) in zip(np.diag(score_n), np.diag(score_v))]
    for (tok, tag) in gold:
        if tok[2] == pptype:
            if aclist[total] == tag:
                correct += 1
            total += 1
    return float(correct)/total


def magic_function(encoding, train_toks, classifier):
    V, N, M = encoding.encode(train_toks)
    Np = (np.dot(N, np.dot(classifier.weight_n(), M.transpose())))
    Vp = (np.dot(V, np.dot(classifier.weight_v(), M.transpose())))
    aclist = [(lambda x,y: 'n' if x > y else 'v')(x,y) for (x,y) in zip(np.diag(Np), np.diag(Vp))]
    total = 0
    correct = 0
    Z = np.exp(Np) + np.exp(Vp)
    nprob = sm.logsumexp(Vp)/Z
    vprob = sm.logsumexp(Vp)/Z
    ll = []
    for (tok, tag) in train_toks:
        if tag == 'n':
            ll.append(np.exp(np.log(np.exp(np.diag(Np)[total])) - np.log(np.exp(np.diag(Vp)[total]) + np.exp(np.diag(Np)[total]))))
        elif tag == 'v':
            ll.append(np.exp(np.log(np.exp(np.diag(Vp)[total])) - np.log(np.exp(np.diag(Vp)[total]) + np.exp(np.diag(Np)[total]))))
        if aclist[total] == tag:
            correct += 1
        total += 1
    acc = float(correct)/total
    empn, empv = classifier.getemp()
    nfeat = np.dot(np.exp(nprob), empn)
    vfeat = np.dot(np.exp(vprob), empv)
    grad_n = (nfeat - empn)
    grad_v = (vfeat - empv)
    return acc, -float(sum(ll))/len(ll), grad_n, grad_v

def calculate_bil_empirical_fcount(train_toks, encoding, label):
    encoded = encoding.encode(train_toks)
    if label == 'n':
        fcount = np.dot(encoded[1].transpose(), encoded[2])
    elif label == 'v':
        fcount = np.dot(encoded[0].transpose(), encoded[2])
    return fcount

def bil_prob(v, n, m, w_n, w_v, label):

    Z = np.exp(np.dot(v, np.dot(w_v, m.transpose()))[0,0]) + np.exp(np.dot(n, np.dot(w_n, m.transpose()))[0,0])
    if label == 'n':
        prob = (np.exp(np.dot(n, np.dot(w_n, m.transpose()))[0,0])) / Z
    elif label == 'v':
        prob = (np.exp(np.dot(v, np.dot(w_v, m.transpose()))[0,0])) / Z
    else:
        raise ValueError('err in bil_prob')
    return prob

def calculate_bil_estimated_fcount(classifier, train_toks, encoding, label):
    encoded = encoding.encode(train_toks)
    prob = bil_prob(encoded[0], encoded[1], encoded[2], classifier.weight_n(), classifier.weight_v(), label)
    if label == 'n':
        fcount = np.dot(prob, np.dot(encoded[1].transpose(), encoded[2]))
    elif label == 'v':
        fcount = np.dot(prob, np.dot(encoded[0].transpose(), encoded[2]))

    return fcount

def loglikelihood(encoding, classifier, gold):
    ll = []
    for (tok, label) in gold:
        sencoded = encoding.sencode(tok)
        ll.append(bil_prob(sencoded[0], sencoded[1], sencoded[2], classifier.weight_n(), classifier.weight_v(), label))
    return np.log(float(sum(ll))/len(ll))


class BilMaxent(object):

    def __init__(self, encoding, weight_n, weight_v, empirical_nfcount=None, empirical_vfcount=None):
        self._weight_n = weight_n
        self._weight_v = weight_v
        self._encoding = encoding
        self._empirical_nfcount = empirical_nfcount
        self._empirical_vfcount = empirical_vfcount
        assert weight_v.shape == weight_n.shape

    def compute_emp_fcount(self, train_toks):
        V, N, M = self._encoding.encode(train_toks)
        self._empirical_nfcount = np.dot(N.transpose(), M)
        self._empirical_vfcount = np.dot(V.transpose(), M)

    def getemp(self):
        return self._empirical_nfcount, self._empirical_vfcount

    def set_weights(self, new_weight_n, new_weight_v):
        assert new_weight_n.shape == new_weight_v.shape
        self._weight_n = new_weight_n
        self._weight_v = new_weight_v

    def weight_n(self):
        return self._weight_n

    def weight_v(self):
        return self._weight_v

    @classmethod
    def bil_train(cls, train_toks, encoding, algorithm='gd', max_iter=10, LC=0.01, tau=0.01, penalty=None, devset=None, devencode=None, pptype='for'):
        if algorithm == 'gd':
            return train_bilin_maxent_classifier_with_gd(train_toks, encoding, max_iter, LC, tau, penalty, devset, devencode, pptype)


class BilMaxentFeatEncoding(object):
    def __init__(self, phi_h, phi_m, map_h, map_m, pptype, train_toks):
        self._phi_h = phi_h
        self._phi_m = phi_m
        self._map_h = list(map_h)
        self._map_m = list(map_m)
        self._pptype = pptype
        self._train_toks = extract_pptype(train_toks, pptype)

    def shape(self):
        return self._phi_h.shape[1], self._phi_m.shape[1]

    def extokens(self):
        return self._train_toks


    def encode(self, train_toks):
        n_list = []
        v_list = []
        m_list = []
        encoding = []
        for (tok, label) in train_toks:
            v_list.append(self._map_h.index(tok[0]))
            n_list.append(self._map_h.index(tok[1]))
            m_list.append(self._map_m.index(tok[3]))

        phi_v = np.matrix(self._phi_h.todense())[v_list]
        encoding.append(phi_v)

        phi_n = np.matrix(self._phi_h.todense())[n_list]
        encoding.append(phi_n)

        phi_m = np.matrix(self._phi_m.todense())[m_list]
        encoding.append(phi_m)

        return encoding

    def sencode(self, toks):

        n_list = []
        v_list = []
        m_list = []
        sencoding = []

        v_list.append(self._map_h.index(toks[0]))
        n_list.append(self._map_h.index(toks[1]))
        m_list.append(self._map_m.index(toks[3]))

        phi_v = np.matrix(self._phi_h.todense())[v_list]
        sencoding.append(phi_v)

        phi_n = np.matrix(self._phi_h.todense())[n_list]
        sencoding.append(phi_n)

        phi_m = np.matrix(self._phi_m.todense())[m_list]
        sencoding.append(phi_m)

        return sencoding

def train_bilin_maxent_classifier_with_gd(train_toks, encoding, max_iter, LC, tau, penalty, devset, devencode, pptype):
    trac = []
    trll = []
#    deac = []
#    dell = []
#    encoded = encoding.encode(train_toks, pptype)


    r,c = encoding.shape()
    weight_ny = np.zeros([r,c], 'd')
    weight_vy = np.zeros([r,c], 'd')
    weight_nx = np.zeros([r,c], 'd')
    weight_vx = np.zeros([r,c], 'd')
    classifier = BilMaxent(encoding, weight_ny, weight_vy)
    classifier.compute_emp_fcount(train_toks)
    print ('------Training (%d iterations----------)' % max_iter)
    print ('----------------------------------------------------------')

#    empirical_fcount_v = calculate_bil_empirical_fcount(train_toks, encoding, label='v', pptype=pptype)
#    empirical_fcount_n = calculate_bil_empirical_fcount(train_toks, encoding, label='n', pptype=pptype)

    iter = 0
#    try:
    while True:
        iter += 1
#        ll = -(loglikelihood(encoding, classifier, train_toks))
        acc, ll, grad_n, grad_v = magic_function(encoding, train_toks, classifier)
        weight_ny = classifier.weight_n()
        weight_vy = classifier.weight_v()
        if penalty == None:
            tau = tau/np.sqrt(iter)
            weight_ny -= tau * grad_n
            weight_vy -= tau * grad_v
            trac.append(acc)
            trll.append(ll)
            print ('%9d   %14.5f  %9.3f' %(iter, ll, acc))

        elif penalty == 'l1':
            nu = 1 / LC
            print nu
            lam_k = 1
#                devencoded = devencode.encode(devset, pptype)
            temp_wy_n = weight_ny - (tau * grad_n) / LC
            temp_wy_v = weight_vy -(tau * grad_v) / LC

            weight_nxp1 = np.maximum(temp_wy_n - nu, 0)
            weight_vxp1 = np.maximum(temp_wy_v - nu, 0)

            lam_kp1 = float(1 + np.sqrt(1 + 4*(lam_k**2 ))) / 2
            lr = (lam_k - 1) / lam_kp1

            weight_nyp1 = weight_nxp1 + lr * (weight_nxp1 - weight_nx)
            weight_vyp1 = weight_vxp1 + lr * (weight_vxp1 - weight_vx)

            print ('%9d   %14.5f  %9.3f' %(iter, ll, acc))
            weight_nx = weight_nxp1
            weight_ny = weight_nyp1

            weight_vx = weight_vxp1
            weight_vy = weight_vyp1


        elif penalty == 'nn':
            nu = 1 / LC
            lam_k = 1
#                devencoded = devencode.encode(devset, pptype)

            temp_wy_n = weight_ny - (tau * grad_n) / LC
            temp_wy_v = weight_vy -(tau * grad_v) / LC

            nU, nS, nVt = np.linalg.svd(temp_wy_n)
            vU, vS, vVt = np.linalg.svd(temp_wy_v)

            nS = np.maximum(nS - nu, 0)
            vS = np.maximum(vS - nu, 0)

            weight_nxp1 = np.dot(nU, np.dot(np.diag(nS), nVt))
            weight_vxp1 = np.dot(vU, np.dot(np.diag(vS), vVt))

            lam_kp1 = float(1 + np.sqrt(1 + 4*(lam_k**2 ))) / 2
            lr = (lam_k - 1) / lam_kp1

            weight_nyp1 = weight_nxp1 + lr * (weight_nxp1 - weight_nx)
            weight_vyp1 = weight_vxp1 + lr * (weight_vxp1 - weight_vx)

            weight_nx = weight_nxp1
            weight_ny = weight_nyp1

            print ('%9d   %14.5f  %9.3f' %(iter, ll, acc))
            weight_vx = weight_vxp1
            weight_vy = weight_vyp1

        elif penalty == 'l2':
            print 'hi'
        classifier.set_weights(weight_ny, weight_vy)

        if iter >= max_iter:
            break
#except:
#        raise ValueError('try, raise, except error')

    return classifier, trac, trll


