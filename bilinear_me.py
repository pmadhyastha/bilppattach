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

def proximal_op(matrix, nu):
        return np.sign(matrix) * np.maximum(np.abs(matrix) - nu, 0.)

def proximal_l2(matrix, nu):
    return ((1./(1.+nu)) * matrix)



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


def accuracy(encoding, classifier, gold):
    '''
    Computing accuracy given some gold data using the learned classifier
    '''
    r,c = encoding.shape()
    bn = np.matrix(classifier.weight_bn())
    bv = np.matrix(classifier.weight_bv())
#    I = np.matrix(np.eye(r,c))
    score = []
    total = 0
    equal = 0
    eqstuff = []

    for (tok, label) in gold:
        total += 1
        noun = 0
        verb = 0
        v, n, m = encoding.bil_u_encode(tok)
#        noun += (n*(I*m.T))[0,0]
#        verb += (v*(I*m.T))[0,0]

#        noun += (n*(bn*m.T))[0,0]
#        verb += (v*(bv*m.T))[0,0]

        noun += (np.dot(n, np.dot(bn, m.transpose())))[0,0]
        verb += (np.dot(v, np.dot(bv, m.transpose())))[0,0]

#        print (np.exp(noun), np.exp(verb))
#        Z = np.exp(noun) + np.exp(verb)

        if np.exp(noun) > np.exp(verb) and label == 'n':
            score.append(1)
        elif np.exp(noun) < np.exp(verb) and label == 'v':
            score.append(1)
        elif np.exp(noun) == np.exp(verb):
            eqstuff.append([tok, label, np.sum(n), np.sum(v), np.sum(m), np.exp(noun), np.exp(verb) ])
            equal += 1

#    if equal > 0:
#        print ('number of equal scores = ', equal)
#    if len(eqstuff) < 20:
#        print (eqstuff)

    return float(np.sum(score)) / total


class BilinearMaxent(object):
    '''
    Main classifier class
    '''
    def __init__(self, encoding, weight_bn, weight_bv):
        self._encoding = encoding
        self._weight_bn = weight_bn
        self._weight_bv = weight_bv
        self._gradN_b = None
        self._gradV_b = None

    def set_weights(self, new_weight_bn,new_weight_bv):
        self._weight_bn = new_weight_bn
        self._weight_bv = new_weight_bv

    def get_grad(self, weight, label1=None, label2=None):
#        print (label, extra)
        if label1 == 'n' and label2 == 'b':
            return self._gradN_b
        elif label1 == 'v' and label2 == 'b':
            return self._gradV_b

    def get_neglogl(self):
        return self._logl

    def gradients(self):
        '''
        Function computes gradient, loglikelihood and accuracy.
        Here,
        Gradient = (empirical features/inner-product - estimated features/inner-product)
        Log-likelihood = log(sum(P(h=(n|v)|m; for every sample(n|v)))
        '''

        bn = self._weight_bn
        bv = self._weight_bv

        emp_bv, emp_bn, bil_inn = self._encoding.ext_emps()

        est_bn = np.matrix(np.zeros(self._encoding.shape(), 'd'))
        est_bv = np.matrix(np.zeros(self._encoding.shape(), 'd'))

        tot_samples = len(self._encoding.train_toks())

        ll = []

        correct = 0

        for tok, label in self._encoding.train_toks():
            v, n, m = self._encoding.bil_encode([(tok, label)])

#            score_n = np.exp((n * (bn * m.transpose()))[0,0])
#            score_v = np.exp((v * (bv * m.transpose()))[0,0])

            score_n = np.exp(np.dot(n, np.dot(bn, m.transpose()))[0,0])
            score_v = np.exp(np.dot(v, np.dot(bv, m.transpose()))[0,0])

            Z = (score_n + score_v)

            probN = float(score_n) / Z
            probV =  float(score_v) / Z
#            print (probN, probV)

            if label == 'n':
                ll.append(probN)

                if probN > probV:
                    correct += 1
            else:
                ll.append(probV)

                if probV > probN:
                    correct += 1

            est_bn += np.dot(probN, bil_inn[tok[1]+'_'+tok[3]])
            est_bv += np.dot(probV, bil_inn[tok[0]+'_'+tok[3]])

        ####Computing for negative log likelihood minimization!!! #####

        gradN_b = -(emp_bn - est_bn)/tot_samples
        gradV_b = -(emp_bv - est_bv)/tot_samples
        logl = - float(np.sum(np.log(ll))) / tot_samples
        acc = float(correct) / tot_samples

        self._gradN_b = gradN_b
        self._gradV_b = gradV_b
        self._logl = logl

        return (gradN_b, gradV_b, logl, acc)

    def weight_bn(self):
        return self._weight_bn

    def weight_bv(self):
        return self._weight_bv

    @classmethod
    def train(cls, train_toks, encoding, algorithm='gd', max_iter=10, tau=0.01, LC=1000, penalty=None, pptype='for', devencode=None, devset=None, eta=1):

        if algorithm == 'gd':
            return train_bilinear_maxent_classifier_with_gd(train_toks, encoding, algorithm, max_iter, tau, LC, penalty, pptype, devencode, devset, eta)


class BilinearMaxentFeatEncoding(object):

    def __init__(self, train_toks, phi_h, phi_m, map_h, map_m, labels, pptype, emps=None):

        self._train_toks = data_extract(train_toks, pptype)
        self._phi_h = np.matrix(phi_h.todense())
        self._phi_m = np.matrix(phi_m.todense())
        self._map_h = list(map_h)
        self._map_m = list(map_m)
        self._labels = list(labels)
        self._emps = emps

    def ext_emps(self):

        if self._emps == None:
            self.compute_emps()
        return self._emps

    def compute_emps(self):

        trn = self._train_toks
        emp_nfcount = np.matrix(np.zeros(self.shape()))
        emp_vfcount = np.matrix(np.zeros(self.shape()))
        fcount_bil = {}

        for tok, label in self._train_toks:
            v, n, m = self.bil_encode([(tok, label)])
            vm = tok[0]+'_'+tok[3]
            nm = tok[1]+'_'+tok[3]
            score_vm = v.T*m
            score_nm = n.T*m

            if vm not in fcount_bil:
                fcount_bil[vm] = score_vm
            if label == 'v':
                emp_vfcount += score_vm
            if nm not in fcount_bil:
                fcount_bil[nm] = score_nm
            if label == 'n':
                emp_nfcount += score_nm

        self._emps = (emp_vfcount, emp_nfcount, fcount_bil)

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

    def labels(self):
        return self._labels

    def shape(self):
        return self._phi_m.shape[1], self._phi_m.shape[1]

    @classmethod
    def train(cls, train_toks, phi_h, phi_m, map_h, map_m, pptype, labels=None, cols=1000):

#        cols_h, phi_h = extract_rep(phi_h, cols)
#        cols_m, phi_m = extract_rep(phi_m, cols)

        if labels is None:
            labels = set(['n', 'v'])
        return cls(train_toks, phi_h, phi_m, map_h, map_m, labels, pptype)

def train_bilinear_maxent_classifier_with_gd(train_toks, encoding, algorithm, max_iter, tau, LC, penalty, pptype, devencode, devset, eta):

    trac = []
    trll = []
    devac = []

    if encoding == None:
        raise ValueError('Build an embedding and pass!!')

    weight_bn = np.matrix(np.zeros(encoding.shape()))
    weight_bv = np.matrix(np.zeros(encoding.shape()))
    classifier = BilinearMaxent(encoding, weight_bn, weight_bv)
    weight_bnx = np.matrix(np.zeros(encoding.shape()))
    weight_bvx = np.matrix(np.zeros(encoding.shape()))
    r,c = encoding.shape()
#    empirical_fcount_bv = calculate_bil_empirical_fcount(train_toks, encoding, label='v')
#    empirical_fcount_bn = calculate_bil_empirical_fcount(train_toks, encoding, label='n')
#
    print ('-------------------------------------Training for %d iterations------------------------------------' % max_iter)
    print ('---------------------------------------------------------------------------------------------------')
    print ('     Iteration         -LogLik          Norms(bn, ln, bv, lv)      Accuracy      Time')
    print ('---------------------------------------------------------------------------------------------------')
#    wscale_bn = 1
#    wscale_bv = 1

    bnS = np.zeros(encoding.shape())
    bvS = np.zeros(encoding.shape())

    t1 = time()
    itr = 0
    lam_k = 1
    while True:
        itr += 1
        lam_kp1 = float(1 + np.sqrt(1 + 4*(lam_k**2 ))) / 2
        grad_bn, grad_bv, ll, acc = classifier.gradients()

#        print (grad_ln, grad_lv)

        if devencode and devset:
            devacc = accuracy(devencode, classifier, devset)
        else:
            devacc = 0

        weight_bny = classifier.weight_bn()
        weight_bvy = classifier.weight_bv()

        if penalty==None:

            bn_norm = np.linalg.norm(weight_bny, ord=2)
            bv_norm = np.linalg.norm(weight_bvy, ord=2)

            sum_norm = (tau * bn_norm) + (tau * bv_norm)

            objective = ll #+ sum_norm

            t2 = time()

            print ('|%9d     |%14.7f    | (%2.3f, %2.3f) |%9.3f  |%9.3f    | %9.3f  |'
                   %(itr, objective, bn_norm, bv_norm, acc, t2-t1, devacc), )

            t1 = time()
            weight_bny -= np.dot(eta, (np.dot(tau, grad_bn) / np.sqrt(itr)))
            weight_bvy -= np.dot(eta, (np.dot(tau, grad_bv) / np.sqrt(itr)))
            trac.append(acc)
            trll.append(objective)
            devac.append(devacc)

        if penalty=='l1':

            bn_norm = np.linalg.norm(weight_bny, ord=1)
            bv_norm = np.linalg.norm(weight_bvy, ord=1)

            objective = ll + sum_norm

            t2 = time()

            print ('|%9d     |%14.7f    | (%2.3f, %2.3f) |%9.3f  |%9.3f    | %9.3f  |'
                   %(itr, objective, bn_norm, bv_norm, acc, t2-t1, devacc), )

            t1 = time()

            nu = tau / LC

            temp_by_n = weight_bny - grad_bn / LC
            temp_by_v = weight_bvy - grad_bv / LC

            weight_bnxp1 = proximal_op(temp_by_n, nu)
            weight_bvxp1 = proximal_op(temp_by_v, nu)

            lr = (lam_k - 1) / lam_kp1

            weight_bnyp1 = weight_bnxp1 + lr * (weight_bnxp1 - weight_bnx)
            weight_bvyp1 = weight_bvxp1 + lr * (weight_bvxp1 - weight_bvx)
            trac.append(acc)
            trll.append(objective)
            devac.append(devacc)

            weight_bnx = weight_bnxp1
            weight_bny = weight_bnyp1

            weight_bvx = weight_bvxp1
            weight_bvy = weight_bvyp1


        if penalty=='nn':

            bn_norm = np.sum(bnS)
            bv_norm = np.sum(bvS)

            sum_norm = (tau * bn_norm) + (tau * bv_norm)

            objective = ll + sum_norm

            t2 = time()

            print ('|%9d     |%14.7f    | (%2.3f, %2.3f) |%9.3f  |%9.3f    | %9.3f  |'
                   %(itr, objective, bn_norm, bv_norm, acc, t2-t1, devacc), )

            t1 = time()

            nu = tau / LC

            temp_by_n = weight_bny - grad_bn / LC
            temp_by_v = weight_bvy - grad_bv / LC

            bnU, bnS, bnVt = np.linalg.svd(temp_by_n)
            bvU, bvS, bvVt = np.linalg.svd(temp_by_v)

            bnS = np.maximum(bnS - nu, 0)
            bvS = np.maximum(bvS - nu, 0)

            weight_bnxp1 = np.dot(bnU, np.dot(np.diag(bnS), bnVt))
            weight_bvxp1 = np.dot(bvU, np.dot(np.diag(bvS), bvVt))

            lr = (lam_k - 1) / lam_kp1
            weight_bnyp1 = weight_bnxp1 + lr * (weight_bnxp1 - weight_bnx)
            weight_bvyp1 = weight_bvxp1 + lr * (weight_bvxp1 - weight_bvx)

            weight_bnx = weight_bnxp1
            weight_bny = weight_bnyp1

            weight_bvx = weight_bvxp1
            weight_bvy = weight_bvyp1
            trac.append(acc)
            trll.append(objective)
            devac.append(devacc)

        if penalty=='l2':

            bn_norm = np.linalg.norm(weight_bny, ord=2)
            bv_norm = np.linalg.norm(weight_bvy, ord=2)

            sum_norm = (tau * bn_norm) + (tau * bv_norm)

            objective = ll + sum_norm

            t2 = time()

            print ('|%9d     |%14.7f    | (%2.3f, %2.3f) |%9.3f  |%9.3f    | %9.3f  |'
                   %(itr, objective, bn_norm, bv_norm, acc, t2-t1, devacc), )

            t1 = time()

            weight_bny -= eta * (grad_bn + np.dot(tau, weight_bny)) / np.sqrt(itr)
            weight_bvy -= eta * (grad_bv + np.dot(tau, weight_bvy)) / np.sqrt(itr)
            trac.append(acc)
            trll.append(objective)
            devac.append(devacc)

        if penalty=='l2p':

            bn_norm = np.linalg.norm(weight_bny, ord=2)
            bv_norm = np.linalg.norm(weight_bvy, ord=2)

            sum_norm = (tau * bn_norm) + (tau * bv_norm)

            objective = ll + sum_norm

            t2 = time()

            print ('|%9d     |%14.7f    | (%2.3f, %2.3f) |%9.3f  |%9.3f    | %9.3f  |'
                   %(itr, objective, bn_norm, bv_norm, acc, t2-t1, devacc), )

            t1 = time()

            nu = tau / LC

            temp_by_n = weight_bny - grad_bn / LC
            temp_by_v = weight_bvy - grad_bv / LC

            weight_bnxp1 = proximal_l2(temp_by_n, nu)
            weight_bvxp1 = proximal_l2(temp_by_v, nu)

            lr = (lam_k - 1) / lam_kp1

            weight_bnyp1 = weight_bnxp1 + lr * (weight_bnxp1 - weight_bnx)
            weight_bvyp1 = weight_bvxp1 + lr * (weight_bvxp1 - weight_bvx)
            trac.append(acc)
            trll.append(objective)
            devac.append(devacc)

            weight_bnx = weight_bnxp1
            weight_bny = weight_bnyp1

            weight_bvx = weight_bvxp1
            weight_bvy = weight_bvyp1


        classifier.set_weights(weight_bny, weight_bvy)
        lam_k = lam_kp1
        if itr >= max_iter:
            break

#    except:
#        raise ValueError('try, raise, except error')
    return classifier, trac, trll, devac

