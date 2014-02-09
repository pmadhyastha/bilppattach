

from __future__ import print_function, unicode_literals, division
#import sklearn.preprocessing as skp
import numpy as np
from time import time
np.seterr(all='raise')


def proximal_op(matrix, nu):
        return np.sign(matrix) * np.maximum(np.abs(matrix) - nu, 0.0)

def proximal_l2(matrix, nu):
    return ((1.0/(1.0+nu)) * matrix)

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

    if encoding_l == None:
        encoding_l = encoding_b

    score = []
    total = 0
    equal = 0

    for (tok, label) in gold:
        prob = {}
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

        prob['n'] = np.exp((n*(bn*m.T))[0,0] + noun)
        prob['v'] = np.exp((v*(bv*m.T))[0,0] + verb)

        if label == max((p, v) for (v, p) in prob.items())[1]:
            score.append(1)

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
        print (len(weight_lv), encoding.length_v())
        print (len(weight_ln), encoding.length_n())
        assert encoding.length_n() == len(weight_ln)
        assert encoding.length_v() == len(weight_lv)
        self._gradN_l = None
        self._gradN_b = None
        self._gradV_l = None
        self._gradV_b = None

    def set_weights(self, new_weight_bn, new_weight_bv):
        self._weight_bn = new_weight_bn
        self._weight_bv = new_weight_bv

    def set_lin_weights(self, new_weight_ln, new_weight_lv):
        self._weight_ln = new_weight_ln
        self._weight_lv = new_weight_lv

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

        emp_bv, emp_bn, emp_lv, emp_ln, bil_inn, meanbv, meanbn, meanlv, meanln = self._encoding.ext_emps()

        est_bn = np.matrix(np.zeros(self._encoding.shape(), 'd'))
        est_bv = np.matrix(np.zeros(self._encoding.shape(), 'd'))

        tot_samples = len(self._encoding.train_toks())
        ll = []

        correct = 0

        for tok, label in self._encoding.train_toks():

            score_nl = 0
            score_vl = 0

            v, n, m = self._encoding.bil_encode([(tok, label)])
            featureset = word_features(tok)

            for l in ('n','v'):

                if l == 'n':

                    feature_vector_n = self._encoding.lin_encode(featureset, l)
                    for (f_id, f_val) in feature_vector_n:
                        score_nl += ln[f_id] * f_val

                    score_bnl = (n * (bn * m.T))[0,0]

                    score_n = np.exp(score_nl + score_bnl)

                if l == 'v':

                    feature_vector_v = self._encoding.lin_encode(featureset, l)

                    for (f_id, f_val) in feature_vector_v:
                        score_vl += lv[f_id] * f_val

                    score_bvl = (v * (bv * m.T))[0,0]

                    score_v = np.exp(score_vl + score_bvl)

            Z = score_n + score_v
            probN = float(score_n) / Z
            probV = float(score_v) / Z

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

        ####Computing for negative log likelihood minimization!!! #####
        gradN_b = -(emp_bn - (est_bn / meanbn)) / tot_samples
        gradV_b = -(emp_bv - (est_bv / meanbv)) / tot_samples

        logl = -float(np.sum(np.log(ll))) / tot_samples
        acc = float(correct) / tot_samples

        self._gradN_b = gradN_b
        self._gradV_b = gradV_b
        self._logl = logl

        return (gradN_b, gradV_b, logl, acc)

    def weight_bn(self):
        return self._weight_bn

    def weight_bv(self):
        return self._weight_bv

    def weight_ln(self):
        return self._weight_ln

    def weight_lv(self):
        return self._weight_lv

    @classmethod
    def combo_train(cls, train_toks, encoding, algorithm='gd', max_iter=10, tau=10, LC=0.5, penalty=None, pptype='for', devencode=None, devset=None, eta=1, fln=None, flv=None):
        if algorithm == 'gd':
            return train_combo_maxent_classifier_with_gd(train_toks, encoding, algorithm, max_iter, tau, LC, penalty, pptype, devencode, devset, eta, fln, flv)


class ComboMaxentFeatEncoding(object):

    def __init__(self, train_toks, phi_h, phi_m, map_h, map_m, labels, mapping_n, mapping_v, featuresets, pptype, fix=1, emps=None,):

        self._train_toks = data_extract(train_toks, pptype)
        self._phi_h = np.matrix(phi_h.todense())
        self._phi_m = np.matrix(phi_m.todense())
        self._fix = fix
        self._map_h = list(map_h)
        self._map_m = list(map_m)
        self._labels = list(labels)
        self._mapping_n = mapping_n
        self._mapping_v = mapping_v
        self._length_n = len(mapping_n)
        self._length_v = len(mapping_v)
        self._shape = self._phi_h.shape[1], self._phi_m.shape[1]
        self._labels = labels
        self._featuresets = featuresets
        self._emps = emps
        print ('emps', emps)
    def mapping_n(self):
        return self._mapping_n

    def mapping_v(self):
        return self._mapping_v

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
        emp_nfcount = np.matrix(np.zeros(self._shape))
        emp_vfcount = np.matrix(np.zeros(self._shape))

        fix_ln = []
        fix_lv = []
        fix_bn = []
        fix_bv = []

        fcount_v = (np.zeros(self._length_v, np.float64))
        fcount_n = (np.zeros(self._length_n, np.float64))

        featuresets = [(word_features(x), c) for x,c in self._train_toks]

        for tok, label in featuresets:
            if label == 'v':
                for (index, val) in self.lin_encode(tok, label):
                    fcount_v[index] += val
                if self._fix == 1 or self._fix == 2:
                    fix_lv.append(np.linalg.norm(fcount_v))
                if self._fix == 3:
                    fix_lv.append(1)


            if label == 'n':
                for (index, val) in self.lin_encode(tok, label):
                    fcount_n[index] += val
                if self._fix == 1 or self._fix == 2:
                    fix_ln.append(np.linalg.norm(fcount_n))
                if self._fix == 3:
                    fix_ln.append(1)

        fcount_bil = {}

        for tok, label in self._train_toks:

            v, n, m = self.bil_encode([(tok, label)])
            vm = tok[0]+'_'+tok[3]
            nm = tok[1]+'_'+tok[3]

            vm_feat = v.T*m
            nm_feat = n.T*m

            if label == 'v':
                emp_vfcount += vm_feat
                if self._fix == 1 or self._fix == 3:
                    fix_bv.append(np.linalg.norm(emp_vfcount))
                if self._fix == 2:
                    fix_bv.append(1)

            if vm not in fcount_bil:
                fcount_bil[vm] = vm_feat

            if label == 'n':
                emp_nfcount += nm_feat
                if self._fix == 1 or self._fix == 3:
                    fix_bn.append(np.linalg.norm(emp_nfcount))
                if self._fix == 2:
                    fix_bn.append(1)

            if nm not in fcount_bil:
                fcount_bil[nm] = nm_feat

        self._emps = ((emp_vfcount/np.mean(fix_bv)), (emp_nfcount/np.mean(fix_bn)), (fcount_v/np.mean(fix_lv)), (fcount_n/np.mean(fix_ln)), fcount_bil, np.mean(fix_bv), np.mean(fix_bn), np.mean(fix_lv), np.mean(fix_ln))

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
        return self._shape

    @classmethod
    def train(cls, train_toks, phi_h, phi_m, map_h, map_m, pptype, labels=None, cols=1000, fix=1):
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

                for (fname, fval) in featureset.items():
                    if label == 'n':
                        if (fname, fval) not in mapping_n:
                            mapping_n[fname, fval] = len(mapping_n)
                    elif label == 'v':
                        if (fname, fval) not in mapping_v:
                            mapping_v[fname, fval] = len(mapping_v)

        else:
            for (tok, label) in train_toks:
                if tok[2] == pptype:
                    featureset = word_features(tok)
                    featuresets.append((featureset, label))
                    seen_labels.add(label)
                    for (fname, fval) in featureset.items():
                        if label == 'n':
                            print ('n - here')
                            if (fname, fval) not in mapping_n:
                                mapping_n[fname, fval] = len(mapping_n)
                        elif label == 'v':
                            print ('v - here')
                            if (fname, fval) not in mapping_v:
                                mapping_v[fname, fval] = len(mapping_v)

        if labels is None:
            labels = seen_labels
        return cls(train_toks, phi_h, phi_m, map_h, map_m, labels, mapping_n, mapping_v, featuresets, pptype, fix)


def train_combo_maxent_classifier_with_gd(train_toks, encoding, algorithm, max_iter, tau, LC, penalty, pptype, devencode, devset, eta, fln, flv):

    trac = []
    trll = []
    devac = []
    if encoding == None:
        raise ValueError('Build an embedding and pass!!')

    weight_bn = np.matrix(np.zeros(encoding.shape()))
    weight_bv = np.matrix(np.zeros(encoding.shape()))
    weight_ln = np.array(np.loadtxt(fln))
    weight_lv = np.array(np.loadtxt(flv))
    classifier = ComboMaxent(encoding, weight_bn, weight_bv, weight_ln, weight_lv)
    weight_bnx = np.matrix(np.zeros(encoding.shape()))
    weight_bvx = np.matrix(np.zeros(encoding.shape()))
    r,c = encoding.shape()

    print ('-------------------------------------Training for %d iterations------------------------------------' % max_iter)
    print ('---------------------------------------------------------------------------------------------------')
    print ('     Iteration         Objective          Norms(bn, bv)      Accuracy      Time    DevelAccuracy')
    print ('---------------------------------------------------------------------------------------------------')
#    wscale_bn = 1
#    wscale_bv = 1

    bnS = np.zeros(encoding.shape())
    bvS = np.zeros(encoding.shape())

    bestdevacc = 0
    bestwts = []
    t1 = time()
    itr = 0
    lam_k = 1
    while True:
        itr += 1
        lam_kp1 = float(1 + np.sqrt(1 + 4*(lam_k**2 ))) / 2
        grad_bn, grad_bv, ll, acc = classifier.gradients()
        devacc = accuracy(encoding, devencode, classifier, devset)

#        print (grad_ln, grad_lv)

#        if devencode and devset:
#            devacc = accuracy(encoding, devencode, classifier, devset)
#        else:
#            devacc = 0

        weight_bny = classifier.weight_bn()
        weight_bvy = classifier.weight_bv()

        if penalty==None:

            bn_norm = np.linalg.norm(weight_bny, ord=2)
            bv_norm = np.linalg.norm(weight_bvy, ord=2)
            combo_norm = (bn_norm + bv_norm)
            objective = ll
            t2 = time()

            print ('|%9d     |%14.7f    | (%2.3f, %2.3f, %2.3f, %2.3f) |%9.3f  |%9.3f    | %9.3f  |'
                   %(itr, objective, bn_norm,
                     bv_norm, acc, t2-t1, devacc), )
            t1 = time()
            weight_bny -= eta * ((tau * grad_bn) / np.sqrt(itr))
            weight_bvy -= eta * ((tau * grad_bv) / np.sqrt(itr))

            trac.append(acc)
            trll.append(objective)
            devac.append(devacc)

        if penalty=='l2p':

            bn_norm = np.linalg.norm(weight_bny, ord=2)
            bv_norm = np.linalg.norm(weight_bvy, ord=2)

            combo_norm = (tau * bn_norm) + (tau * bv_norm)

            objective = ll + combo_norm

            t2 = time()

            print ('|%9d     |%14.7f    | (%2.3f, %2.3f) |%9.3f  |%9.3f    | %9.3f  |'
                   %(itr, objective, bn_norm,
                     bv_norm, acc, t2-t1, devacc), )

            t1 = time()

            nu_b = tau / LC

            temp_by_n = weight_bny - grad_bn / LC
            temp_by_v = weight_bvy - grad_bv / LC

            weight_bnxp1 = proximal_l2(temp_by_n, nu_b)
            weight_bvxp1 = proximal_l2(temp_by_v, nu_b)

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

            combo_norm = (tau * bn_norm) + (tau * bv_norm)

            objective = ll + combo_norm

            t2 = time()

            print ('|%9d     |%14.7f    | (%2.3f, %2.3f) |%9.3f  |%9.3f    | %9.3f  |'
                   %(itr, objective, bn_norm,
                     bv_norm, acc, t2-t1, devacc), )

            t1 = time()

            nu_b = tau / LC

            temp_by_n = weight_bny - grad_bn / LC
            temp_by_v = weight_bvy - grad_bv / LC

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

            trac.append(acc)
            trll.append(objective)
            devac.append(devacc)

        if penalty=='l1':

            bn_norm = np.linalg.norm(weight_bny, ord=1)
            bv_norm = np.linalg.norm(weight_bvy, ord=1)

            combo_norm = (tau * bn_norm) + (tau * bv_norm)
            objective = ll + combo_norm

            t2 = time()

            print ('|%9d     |%14.7f    | (%2.3f, %2.3f) |%9.3f  |%9.3f    | %9.3f  |'
                   %(itr, objective, bn_norm,
                     bv_norm, acc, t2-t1, devacc), )

            t1 = time()

            nu_b = tau / LC
            temp_by_n = weight_bny - grad_bn / LC
            temp_by_v = weight_bvy - grad_bv / LC

            weight_bnxp1 =  proximal_op(temp_by_n, nu_b)
            weight_bvxp1 =  proximal_op(temp_by_v, nu_b)

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

        prev_wt = [classifier.weight_bn(), classifier.weight_bv()]
        classifier.set_weights(weight_bny, weight_bvy)

        if bestdevacc < devacc:
            bestdevacc = devacc
            bestwts = prev_wt

        lam_k = lam_kp1
        if itr >= max_iter:

            break

#    except:
#        raise ValueError('try, raise, except error')
    return classifier, trac, trll, devac, bestdevacc, bestwts

