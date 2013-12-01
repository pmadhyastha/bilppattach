#!/usr/bin/python

from __future__ import print_function, unicode_literals, division
#import sklearn.preprocessing as skp
import numpy as np
from time import time
np.seterr(all='raise')


#def extract_rep(matrix, cols=0):
#    if cols is 0:
#        cols = matrix.shape[1]
#    retain_cols = np.array(matrix.tocsc().sum(axis=0).tolist()[0]).argsort()[::-1][:cols]

#    return retain_cols, skp.normalize(matrix.tocsc()[:, retain_cols].tocsr(), norm='l1', axis=1)


def data_extract(train_tokens, pptype):
    '''
    Extraction of data, with a specific format:
    [((Verb, Noun1, Prep, Noun2),Label=(v|n))]
    '''
    train_tok = []
    for (tok, label) in train_tokens:
        if pptype == None:
            train_tok.append((tok, label))
        elif tok[2] == pptype:
            train_tok.append((tok, label))
    return train_tok


def word_features(l):
    '''
    Extracting features from tokens
    '''
    return dict(quad=l[0]+l[1]+l[2]+l[3], tri1=l[0]+l[1]+l[2], \
    tri2=l[1]+l[2]+l[3], bi1=l[0]+l[1], bi2=l[1]+l[2], bi3=l[2]+l[3], \
    bi4=l[0]+l[2], bi5=l[0]+l[3], bi6=l[1]+l[3], uni1=l[0], uni2=l[1],\
    uni3=l[2], uni4=l[3])


def accuracy(encoding_l, encoding_b, classifier, gold):
    '''
    Computing accuracy given some gold data using the learned classifier
    '''
    r,c = encoding_l.shape()
    ln = (classifier.weight_ln())
    lv = (classifier.weight_lv())
    bn = np.matrix(classifier.weight_bn())
    bv = np.matrix(classifier.weight_bv())
#    I = np.matrix(np.eye(r,c))
    if encoding_l == None:
        encoding_l = encoding_b
    score = []
    total = 0
    equal = 0

    for (tok, label) in gold:
        total += 1
        noun = 0
        verb = 0
        v, n, m = encoding_b.bil_u_encode(tok)
        featureset = encoding_l.ext_featstruct(tok)
        fvec_n = encoding_l.lin_encode(featureset, 'n')
        fvec_v = encoding_l.lin_encode(featureset, 'v')
        for (f_id, f_val) in fvec_n:
            noun += ln[f_id] * f_val
        for (f_id, f_val) in fvec_v:
            verb += lv[f_id] * f_val
#        print (noun, verb)
#            print ('1 ', noun, verb)
#        noun += (n*(I*m.T))[0,0]
#        verb += (v*(I*m.T))[0,0]

        noun += (n*(bn*m.T))[0,0]
        verb += (v*(bv*m.T))[0,0]
#        print (np.exp(noun), np.exp(verb))
#        Z = np.exp(noun) + np.exp(verb)
        if np.exp(noun) > np.exp(verb) and label == 'n':
            score.append(1)
        elif np.exp(noun) < np.exp(verb) and label == 'v':
            score.append(1)
        elif np.exp(noun) == np.exp(verb):
            equal += 1
    if equal > 0:
        print ('number of equal scores = ', equal)
    return float(np.sum(score)) / total

class ComboMaxent(object):
    '''
    Main classifier class
    '''
    def __init__(self, encoding, weight_bn, weight_bv, weight_ln, weight_lv):
        self._encoding = encoding
        self._weight_bn = weight_bn
        self._weight_bv = weight_bv
        self._weight_ln = weight_ln
        self._weight_lv = weight_lv
        assert encoding.length_n() == len(weight_ln)
        assert encoding.length_v() == len(weight_lv)
        self._gradN_l = None
        self._gradN_b = None
        self._gradV_l = None
        self._gradV_b = None

    def set_weights(self, new_weight_bn,new_weight_bv, new_weight_ln, new_weight_lv):
        self._weight_bn = new_weight_bn
        self._weight_bv = new_weight_bv
        self._weight_ln = new_weight_ln
        self._weight_lv = new_weight_lv

    def get_grad(self, weight, label1=None, label2=None):
#        print (label, extra)
        if label1 == 'n' and label2 == 'l':
            return self._gradN_l
        elif label1 == 'n' and label2 == 'b':
            return self._gradN_b
        elif label1 == 'v' and label2 == 'l':
            return self._gradV_l
        elif label1 == 'v' and label2 == 'b':
            return self._gradV_b

    def get_neglogl(self, weight, label1=None, label2=None):
        return self._logl

    def gradients(self):
        '''
        Function computes gradient, loglikelihood and accuracy.
        Here,
        Gradient = (empirical features/inner-product - estimated features/inner-product)
        Log-likelihood = log(sum(P(h=(n|v)|m; for every sample(n|v)))

        '''
        ln = self._weight_ln
        lv = self._weight_lv
        bn = self._weight_bn
        bv = self._weight_bv

        emp_bv, emp_bn, emp_lv, emp_ln, bil_inn = self._encoding.ext_emps()
        est_bn = np.matrix(np.zeros(self._encoding.shape(), 'd'))
        est_ln = (np.zeros(self._encoding.length_n(), 'd'))
        est_bv = np.matrix(np.zeros(self._encoding.shape(), 'd'))
        est_lv = (np.zeros(self._encoding.length_v(), 'd'))
        tot_samples = len(self._encoding.train_toks())
        ll = []
        correct = 0

        for tok, label in self._encoding.train_toks():
            prob_nl = 0
            prob_vl = 0
            v, n, m = self._encoding.bil_encode([(tok, label)])
            featureset = word_features(tok)

            for l in ('n','v'):
                if l == 'n':
                    feature_vector_n = self._encoding.lin_encode(featureset, l)
                    for (f_id, f_val) in feature_vector_n:
                        prob_nl += ln[f_id] * f_val
                    prob_bnl = (n * (bn * m.T))[0,0]
                    prob_n = np.exp(prob_nl + prob_bnl)
                if l == 'v':
                    feature_vector_v = self._encoding.lin_encode(featureset, l)
                    for (f_id, f_val) in feature_vector_v:
                        prob_vl += lv[f_id] * f_val
                    prob_bvl = (v * (bv * m.T))[0,0]
                    prob_v = np.exp(prob_vl + prob_bvl)

            Z = prob_n + prob_v
            probN = float(prob_n) / Z
            probV=  float(prob_v) / Z
#            print (probN, probV)
            if label == 'n':
                ll.append(probN)
                if probN > probV:
                    correct += 1
            else:
                ll.append(probV)
                if probV > probN:
                    correct += 1

            est_bn += probN * bil_inn[tok[1]+'_'+tok[3]]
            est_bv += probV * bil_inn[tok[0]+'_'+tok[3]]
            for (index, val) in feature_vector_n:
                est_ln[index] += probN * val
            for (index, val) in feature_vector_v:
                est_lv[index] += probV * val
        ####Computing for negative log likelihood minimization!!! #####
        gradN_l = -(emp_ln - est_ln)/tot_samples
        gradN_b = -(emp_bn - est_bn)/tot_samples
        gradV_l = -(emp_lv - est_lv)/tot_samples
        gradV_b = -(emp_bv - est_bv)/tot_samples
        logl = -float(np.sum(np.log(ll))) / tot_samples
        acc = float(correct) / tot_samples

        self._gradN_l = gradN_l
        self._gradN_b = gradN_b
        self._gradV_l = gradV_l
        self._gradV_b = gradV_b
        self._logl = logl

        return (gradN_l, gradN_b, gradV_l, gradV_b, logl, acc)

    def weight_bn(self):
        return self._weight_bn

    def weight_bv(self):
        return self._weight_bv

    def weight_ln(self):
        return self._weight_ln

    def weight_lv(self):
        return self._weight_lv

    @classmethod
    def combo_train(cls, train_toks, encoding, algorithm='gd', max_iter=10, tau_l=1, tau_b=1, LC_l=1000, LC_b=1000, l_penalty=None, b_penalty=None, pptype='for', devencode=None, devset=None, eta=1):
        if algorithm == 'gd':
            return train_combo_maxent_classifier_with_gd(train_toks, encoding, algorithm, max_iter, tau_l, tau_b, LC_l, LC_b, l_penalty, b_penalty, pptype, devencode, devset, eta)


class ComboMaxentFeatEncoding(object):

    def __init__(self, train_toks, phi_h, phi_m, map_h, map_m, labels, mapping_n, mapping_v, featuresets, pptype, emps=None):

        self._train_toks = data_extract(train_toks, pptype)
        self._phi_h = np.matrix(phi_h.todense())
        self._phi_m = np.matrix(phi_m.todense())
        self._map_h = list(map_h)
        self._map_m = list(map_m)
        self._labels = list(labels)
        self._mapping_n = mapping_n
        self._mapping_v = mapping_v
        self._length_n = len(mapping_n)
        self._length_v = len(mapping_v)
        self._labels = labels
        self._featuresets = featuresets
        self._emps = emps

    def featdata(self):
        return self._featuresets

    def ext_featstruct(self, tok):
        return word_features(tok)

    def ext_emps(self):
        if self._emps == None:
            self.compute_emps()
        return self._emps

    def compute_emps(self):
        trn = self._train_toks
        V, N, M = self.bil_encode(trn)
        emp_nfcount = N.T*M
        emp_vfcount = V.T*M

        fcount_v = (np.zeros(self._length_v, np.float64))
        fcount_n = (np.zeros(self._length_n, np.float64))
        featuresets = [(word_features(x), c) for x,c in self._train_toks]
        verb = 'v'
        noun = 'n'
        for tok, l in featuresets:
            for (index, val) in self.lin_encode(tok, verb):
                fcount_v[index] += val
            for (index, val) in self.lin_encode(tok, noun):
                fcount_n[index] += val
        fcount_bil = {}
        for tok, label in self._train_toks:
            v, n, m = self.bil_encode([(tok, label)])
            vm = tok[0]+'_'+tok[3]
            nm = tok[1]+'_'+tok[3]
            if vm not in fcount_bil:
                fcount_bil[vm] = v.T*m
            if nm not in fcount_bil:
                fcount_bil[nm] = n.T*m
        self._emps = (emp_vfcount, emp_nfcount, fcount_v, fcount_n, fcount_bil)

    def train_toks(self):
        return self._train_toks
    def bil_encode(self, train_toks):
        n_list = []
        v_list = []
        m_list = []
        encoding = []
        for (tok, label) in train_toks:
            v_list.append(self._map_h.index(tok[0]))
            n_list.append(self._map_h.index(tok[1]))
            m_list.append(self._map_m.index(tok[3]))

        phi_v = self._phi_h[v_list]
        encoding.append(phi_v)

        phi_n = self._phi_h[n_list]
        encoding.append(phi_n)

        phi_m = self._phi_m[m_list]
        encoding.append(phi_m)

        return encoding

    def bil_u_encode(self, tok):
        return (self._phi_h[self._map_h.index(tok[0])],\
                self._phi_h[self._map_h.index(tok[1])],\
                self._phi_m[self._map_m.index(tok[3])])

    def lin_encode(self, featureset, label):
        encoding = []
        if label == 'n':
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

    def shape(self):
        return self._phi_m.shape[1], self._phi_m.shape[1]

    @classmethod
    def train(cls, train_toks, phi_h, phi_m, map_h, map_m, pptype, labels=None, cols=1000):
        mapping_n = {}
        mapping_v = {}
        seen_labels = set()
        featuresets = []
#        cols_h, phi_h = extract_rep(phi_h, cols)
#        cols_m, phi_m = extract_rep(phi_m, cols)
        if pptype == None:
            for (tok, label) in train_toks:
                featureset = word_features(tok)
                featuresets.append((featureset, label))
                seen_labels.add(label)

                if label == 'n':
                    for (fname, fval) in featureset.items():
                        if (fname, fval) not in mapping_n:
                            mapping_n[fname, fval] = len(mapping_n)
                if label == 'v':
                    for (fname, fval) in featureset.items():
                        if (fname, fval) not in mapping_v:
                            mapping_v[fname, fval] = len(mapping_v)

        else:
            for (tok, label) in train_toks:
                if tok[2] == pptype:
                    featureset = word_features(tok)
                    featuresets.append((featureset, label))
                    seen_labels.add(label)
                    if label == 'n':
                        for (fname, fval) in featureset.items():
                            if (fname, fval) not in mapping_n:
                                mapping_n[fname, fval] = len(mapping_n)
                    if label == 'v':
                        for (fname, fval) in featureset.items():
                            if (fname, fval) not in mapping_v:
                                mapping_v[fname, fval] = len(mapping_v)

        if labels is None:
            labels = seen_labels
        return cls(train_toks, phi_h, phi_m, map_h, map_m, labels, mapping_n, mapping_v, featuresets, pptype)


def train_combo_maxent_classifier_with_gd(train_toks, encoding, algorithm, max_iter, tau_l, tau_b, LC_l, LC_b, l_penalty, b_penalty, pptype, devencode, devset, eta):

    trac = []
    trll = []
    devac = []
    if encoding == None:
        raise ValueError('Build an embedding and pass!!')

    weight_bn = np.matrix(np.zeros(encoding.shape()))
    weight_bv = np.matrix(np.zeros(encoding.shape()))
    weight_ln = np.array(np.zeros(encoding.length_n()))
    weight_lv = np.array(np.zeros(encoding.length_v()))
    classifier = ComboMaxent(encoding, weight_bn, weight_bv, weight_ln, weight_lv)
    weight_bnx = np.matrix(np.zeros(encoding.shape()))
    weight_bvx = np.matrix(np.zeros(encoding.shape()))
    weight_lnx = np.array(np.zeros(encoding.length_n()))
    weight_lvx = np.array(np.zeros(encoding.length_v()))
    r,c = encoding.shape()
#    empirical_fcount_bv = calculate_bil_empirical_fcount(train_toks, encoding, label='v')
#    empirical_fcount_bn = calculate_bil_empirical_fcount(train_toks, encoding, label='n')
#    empirical_fcount_lv = calculate_lin_empirical_fcount(train_toks, encoding, label='v')
#    empirical_fcount_ln = calculate_lin_empirical_fcount(train_toks, encoding, label='n')
#
    print ('-------------------------------------Training for %d iterations------------------------------------' % max_iter)
    print ('---------------------------------------------------------------------------------------------------')
    print ('     Iteration         -LogLik          Norms(bn, ln, bv, lv)      Accuracy      Time    DevelAccuracy')
    print ('---------------------------------------------------------------------------------------------------')
#    wscale_bn = 1
#    wscale_ln = 1
#    wscale_bv = 1
#    wscale_lv = 1

    bnS = np.zeros(encoding.shape())
    bvS = np.zeros(encoding.shape())

    t1 = time()
    itr = 0
    lam_k = 1
    while True:
        itr += 1
        lam_kp1 = float(1 + np.sqrt(1 + 4*(lam_k**2 ))) / 2
        grad_ln, grad_bn, grad_lv, grad_bv, ll, acc = classifier.gradients()
        devacc = accuracy(encoding, devencode, classifier, devset)

#        print (grad_ln, grad_lv)

#        if devencode and devset:
#            devacc = accuracy(encoding, devencode, classifier, devset)
#        else:
#            devacc = 0

        weight_bny = classifier.weight_bn()
        weight_bvy = classifier.weight_bv()
        weight_lvy = classifier.weight_lv()
        weight_lny = classifier.weight_ln()

        if l_penalty==None and  b_penalty==None:
            t2 = time()
            print ('|%9d     |%14.7f    | (%2.3f, %2.3f, %2.3f, %2.3f) |%9.3f  |%9.3f    | %9.3f  |'
                   %(itr, ll, np.linalg.norm(weight_bny, ord=2),
                     np.linalg.norm(weight_lny, ord=2),
                     np.linalg.norm(weight_bvy, ord=2),
                     np.linalg.norm(weight_lvy, ord=2), acc, t2-t1, devacc), )
            t1 = time()
            weight_bny -= eta * ((tau_b * grad_bn) / np.sqrt(itr))
            weight_bvy -= eta * ((tau_b * grad_bv) / np.sqrt(itr))
            weight_lvy -= eta * ((tau_l * grad_lv) / np.sqrt(itr))
            weight_lny -= eta * ((tau_l * grad_ln) / np.sqrt(itr))
            trac.append(acc)
            trll.append(ll)
            devac.append(devacc)

        if l_penalty=='l1' and b_penalty=='l1':

            t2 = time()
            print ('|%9d     |%14.7f    | (%2.3f, %2.3f, %2.3f, %2.3f) |%9.3f  |%9.3f    | %9.3f    |'
                   %(itr, ll, np.linalg.norm(weight_bny, ord=1),
                     np.linalg.norm(weight_lny, ord=1),
                     np.linalg.norm(weight_bvy, ord=1),
                     np.linalg.norm(weight_lvy, ord=1), acc, t2-t1, devacc))
            t1 = time()

            nu_l = 1 / LC_l
            nu_b = 1 / LC_b

            temp_by_n = weight_bny - (tau_b * grad_bn) / LC_b
            temp_ly_n = weight_lny - (tau_l * grad_ln) / LC_l
            temp_by_v = weight_bvy - (tau_b * grad_bv) / LC_b
            temp_ly_v = weight_lvy - (tau_l * grad_lv) / LC_l

            weight_bnxp1 = np.where(temp_by_n > 0, np.maximum(temp_by_n - nu_b, 0), np.minimum(temp_by_n + nu_b, 0))
            weight_lnxp1 = np.where(temp_ly_n > 0, np.maximum(temp_ly_n - nu_l, 0), np.minimum(temp_ly_n + nu_l, 0))
            weight_bvxp1 = np.where(temp_by_v > 0, np.maximum(temp_by_v - nu_b, 0), np.minimum(temp_by_n + nu_b, 0))
            weight_lvxp1 = np.where(temp_ly_v > 0, np.maximum(temp_ly_v - nu_l, 0), np.minimum(temp_ly_v + nu_l, 0))

            lr = (lam_k - 1) / lam_kp1

            weight_bnyp1 = weight_bnxp1 + lr * (weight_bnxp1 - weight_bnx)
            weight_lnyp1 = weight_lnxp1 + lr * (weight_lnxp1 - weight_lnx)
            weight_bvyp1 = weight_bvxp1 + lr * (weight_bvxp1 - weight_bvx)
            weight_lvyp1 = weight_lvxp1 + lr * (weight_lvxp1 - weight_lvx)
            trac.append(acc)
            trll.append(ll)
            devac.append(devacc)

            weight_bnx = weight_bnxp1
            weight_lnx = weight_lnxp1
            weight_bny = weight_bnyp1
            weight_lny = weight_lnyp1

            weight_bvx = weight_bvxp1
            weight_lvx = weight_lvxp1
            weight_bvy = weight_bvyp1
            weight_lvy = weight_lvyp1


        if l_penalty=='l1' and b_penalty=='nn':
            t2 = time()
            print ('|%9d     |%14.7f    | (%2.3f, %2.3f, %2.3f, %2.3f) |%9.3f  |%9.3f    | %9.3f    |'
                   %(itr, ll, np.sum(bnS),
                     np.linalg.norm(weight_lny, ord=1),
                     np.sum(bvS),
                     np.linalg.norm(weight_lvy, ord=1), acc, t2-t1, devacc))
            t1 = time()

            nu_b = 1 / LC_b
            nu_l = 1 / LC_l

            temp_by_n = weight_bny - (tau_b * grad_bn) / LC_b
            temp_ly_n = weight_lny - (tau_l * grad_ln) / LC_l
            temp_by_v = weight_bvy - (tau_b * grad_bv) / LC_b
            temp_ly_v = weight_lvy - (tau_l * grad_lv) / LC_l

            bnU, bnS, bnVt = np.linalg.svd(temp_by_n)
            bvU, bvS, bvVt = np.linalg.svd(temp_by_v)

            bnS = np.maximum(bnS - nu_b, 0)
            bvS = np.maximum(bvS - nu_b, 0)

            weight_bnxp1 = np.dot(bnU, np.dot(np.diag(bnS), bnVt))
            weight_bvxp1 = np.dot(bvU, np.dot(np.diag(bvS), bvVt))
            weight_lnxp1 = np.where(temp_ly_n > 0, np.maximum(temp_ly_n - nu_l, 0), np.minimum(temp_ly_n + nu_l, 0))
            weight_lvxp1 = np.where(temp_ly_v > 0, np.maximum(temp_ly_v - nu_l, 0), np.minimum(temp_ly_v + nu_l, 0))

            lr = (lam_k - 1) / lam_kp1
            weight_bnyp1 = weight_bnxp1 + lr * (weight_bnxp1 - weight_bnx)
            weight_lnyp1 = weight_lnxp1 + lr * (weight_lnxp1 - weight_lnx)
            weight_bvyp1 = weight_bvxp1 + lr * (weight_bvxp1 - weight_bvx)
            weight_lvyp1 = weight_lvxp1 + lr * (weight_lvxp1 - weight_lvx)

            weight_bnx = weight_bnxp1
            weight_lnx = weight_lnxp1
            weight_bny = weight_bnyp1
            weight_lny = weight_lnyp1

            weight_bvx = weight_bvxp1
            weight_lvx = weight_lvxp1
            weight_bvy = weight_bvyp1
            weight_lvy = weight_lvyp1
            trac.append(acc)
            trll.append(ll)
            devac.append(devacc)

        if l_penalty=='l2' and b_penalty=='l2':

            t2 = time()
            print ('|%9d     |%14.7f    | (%2.3f, %2.3f, %2.3f, %2.3f) |%9.3f  |%9.3f    | %9.3f  |'
                   %(itr, ll, np.linalg.norm(weight_bny, ord=2),
                     np.linalg.norm(weight_lny, ord=2),
                     np.linalg.norm(weight_bvy, ord=2),
                     np.linalg.norm(weight_lvy, ord=2), acc, t2-t1, devacc), )
            t1 = time()

            weight_bny -= eta * (grad_bn + np.dot(tau_b, weight_bny)) / np.sqrt(itr)
            weight_bvy -= eta * (grad_bv + np.dot(tau_b, weight_bvy)) / np.sqrt(itr)
            weight_lvy -= eta * (grad_lv + np.dot(tau_l, weight_lvy)) / np.sqrt(itr)
            weight_lny -= eta * (grad_ln + np.dot(tau_l, weight_lny))  / np.sqrt(itr)

            ll += (tau_b*((np.linalg.norm(weight_bny)**2) + (np.linalg.norm(weight_bvy)**2)) + tau_l*((np.linalg.norm(weight_lny)**2) + (np.linalg.norm(weight_lvy)**2)))
            trac.append(acc)
            trll.append(ll)
            devac.append(devacc)

        if l_penalty=='l2p' and b_penalty=='l2p':

            t2 = time()
            print ('|%9d     |%14.7f    | (%2.3f, %2.3f, %2.3f, %2.3f) |%9.3f  |%9.3f    |'
                   %(itr, ll, np.linalg.norm(weight_bny, ord=2),
                     np.linalg.norm(weight_lny, ord=2),
                     np.linalg.norm(weight_bvy, ord=2),
                     np.linalg.norm(weight_lvy, ord=2), acc, t2-t1))
            t1 = time()

            weight_bny -= (grad_bn) / np.sqrt(itr)
            weight_bvy -= (grad_bv) / np.sqrt(itr)
            weight_lvy -= (grad_lv) / np.sqrt(itr)
            weight_lny -= (grad_ln)  / np.sqrt(itr)

            weight_bvy = eta * ((weight_bvy / np.linalg.norm(weight_bvy, ord=2)) * tau_b)
            weight_bny = eta * ((weight_bny / np.linalg.norm(weight_bny, ord=2)) * tau_b)
            weight_lvy = eta * ((weight_lvy / np.linalg.norm(weight_lvy, ord=2)) * tau_l)
            weight_lny = eta * ((weight_lny / np.linalg.norm(weight_lny, ord=2)) * tau_l)

            trac.append(acc)
            trll.append(ll)


        if l_penalty=='l2' and b_penalty=='nn':
            t2 = time()
            print ('|%9d     |%14.7f    | (%2.3f, %2.3f, %2.3f, %2.3f) |%9.3f  |%9.3f    |'
                   %(itr, ll, np.sum(bnS),
                     np.linalg.norm(weight_lny, ord=2),
                     np.sum(bvS),
                     np.linalg.norm(weight_lvy, ord=2), acc, t2-t1))
            t1 = time()

            nu_b = 1 / LC_b

            temp_by_n = weight_bny - (tau_b * grad_bn) / LC_b
            temp_by_v = weight_bvy - (tau_b * grad_bv) / LC_b

            bnU, bnS, bnVt = np.linalg.svd(temp_by_n)
            bvU, bvS, bvVt = np.linalg.svd(temp_by_v)

            bnS = np.maximum(bnS - nu_b, 0)
            bvS = np.maximum(bvS - nu_b, 0)

            weight_bnxp1 = np.dot(bnU, np.dot(np.diag(bnS), bnVt))
            weight_bvxp1 = np.dot(bvU, np.dot(np.diag(bvS), bvVt))
            lr = (lam_k - 1) / lam_kp1
            weight_bnyp1 = weight_bnxp1 + lr * (weight_bnxp1 - weight_bnx)
            weight_bvyp1 = weight_bvxp1 + lr * (weight_bvxp1 - weight_bvx)

            weight_bnx = weight_bnxp1
            weight_bny = weight_bnyp1
            weight_bvx = weight_bvxp1
            weight_bvy = weight_bvyp1
            weight_lvy -= (grad_lv + np.dot(tau_l, weight_lvy)) / np.sqrt(itr)
            weight_lny -= (grad_ln + np.dot(tau_l, weight_lny))  / np.sqrt(itr)

            trac.append(acc)
            trll.append(ll)
            devac.append(devacc)

        if l_penalty=='l2' and b_penalty=='l1':

            t2 = time()
            print ('|%9d     |%14.7f    | (%2.3f, %2.3f, %2.3f, %2.3f) |%9.3f  |%9.3f    |  %9.3f |'
                   %(itr, ll, np.linalg.norm(weight_bny, ord=1),
                     np.linalg.norm(weight_lny, ord=2),
                     np.linalg.norm(weight_bvy, ord=1),
                     np.linalg.norm(weight_lvy, ord=2), acc, t2-t1, devacc))
            t1 = time()

            nu_b = 1 / LC_b
            temp_by_n = weight_bny - (tau_b * grad_bn) / LC_b
            temp_by_v = weight_bvy - (tau_b * grad_bv) / LC_b

            weight_bnxp1 = np.where(temp_by_n > 0, np.maximum(temp_by_n - nu_b, 0), np.minimum(temp_by_n + nu_b, 0))
            weight_bvxp1 = np.where(temp_by_v > 0, np.maximum(temp_by_v - nu_b, 0), np.minimum(temp_by_n + nu_b, 0))

            lr = (lam_k - 1) / lam_kp1

            weight_bnyp1 = weight_bnxp1 + lr * (weight_bnxp1 - weight_bnx)
            weight_bvyp1 = weight_bvxp1 + lr * (weight_bvxp1 - weight_bvx)
            trac.append(acc)
            trll.append(ll)
            devac.append(devacc)

            weight_bnx = weight_bnxp1
            weight_bny = weight_bnyp1

            weight_bvx = weight_bvxp1
            weight_bvy = weight_bvyp1
            weight_lvy -= (grad_lv + np.dot(tau_l, weight_lvy)) / np.sqrt(itr)
            weight_lny -= (grad_ln + np.dot(tau_l, weight_lny))  / np.sqrt(itr)

        if l_penalty=='l1' and b_penalty=='l2':

            t2 = time()
            print ('|%9d     |%14.7f    | (%2.3f, %2.3f, %2.3f, %2.3f) |%9.3f  |%9.3f    |'
                   %(itr, ll, np.linalg.norm(weight_bny, ord=2),
                     np.linalg.norm(weight_lny, ord=1),
                     np.linalg.norm(weight_bvy, ord=2),
                     np.linalg.norm(weight_lvy, ord=1), acc, t2-t1))
            t1 = time()

            nu_l = 1 / LC_l

            temp_ly_n = weight_lny - (tau_l * grad_ln) / LC_l
            temp_ly_v = weight_lvy - (tau_l * grad_lv) / LC_l

            weight_lnxp1 = np.where(temp_ly_n > 0, np.maximum(temp_ly_n - nu_l, 0), np.minimum(temp_ly_n + nu_l, 0))
            weight_lvxp1 = np.where(temp_ly_v > 0, np.maximum(temp_ly_v - nu_l, 0), np.minimum(temp_ly_v + nu_l, 0))

            lr = (lam_k - 1) / lam_kp1

            weight_lnyp1 = weight_lnxp1 + lr * (weight_lnxp1 - weight_lnx)
            weight_lvyp1 = weight_lvxp1 + lr * (weight_lvxp1 - weight_lvx)
            trac.append(acc)
            trll.append(ll)

            weight_lnx = weight_lnxp1
            weight_lny = weight_lnyp1

            weight_lvx = weight_lvxp1
            weight_lvy = weight_lvyp1
            weight_bny -= (grad_bn + np.dot(tau_b, weight_bny)) / np.sqrt(itr)
            weight_bvy -= (grad_bv + np.dot(tau_b, weight_bvy)) / np.sqrt(itr)



        classifier.set_weights(weight_bny, weight_bvy, weight_lny, weight_lvy)
        lam_k = lam_kp1
        if itr >= max_iter:
            break

#    except:
#        raise ValueError('try, raise, except error')
    return classifier, trac, trll, devac

