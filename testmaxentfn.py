from __future__ import division
import numpy as np
import scipy.optimize as so
import nltk


class ProbDist(object):

    def __init__(self, prob_dict):
        self._prob_dict = prob_dict.copy()

    def prob(self, label):
        return (np.exp(self._prob_dict[label]) /
                np.sum([np.exp(val) for val in self._prob_dict.values()]))

    def max(self):
        if not hasattr(self, '_max'):
            self._max = max((p, v) for (v, p) in self._prob_dict.items())[1]
        return self._max

    def samples(self):
        return self._prob_dict.keys()


class Classification(object):

    def __init__(self, traintokens, develtokens, encoding, weights, tau=0.1, itr=50, tp=0):

        self.traintokens = traintokens
        self.develtokens = develtokens
        self.cnt = 0
        self.tau = tau
        self.itr = itr
        self.encoding = encoding
        self.weights = weights
        self.tp = tp
        self.savenorm = []
        self.saveobj = []
        self.savell = []
        self.savedevacc = []
        self.emp_fcount = np.zeros(self.encoding.length(), 'd')
        for tok, label in self.traintokens:
            for (index, val) in self.encoding.encode(tok, label):
                self.emp_fcount[index] += val



    def weights(self):
        return self.weights

    def classify(self, featureset):
        return self.prob_classify(featureset).max()

    def batch_classify(self, featuresets):
        return [self.classify(fs) for fs in featuresets]

    def batch_prob_classify(self, featuresets):
        return [self.prob_classify(fs) for fs in featuresets]

    def prob_classify(self, featureset):
        prob_dict = {}
        for label in self.encoding.labels():
            feature_vector = self.encoding.encode(featureset, label)
            total = 0.0
            for (f_id, f_val) in feature_vector:
                total += self.weights[f_id] * f_val
            prob_dict[label] = total
        return ProbDist(prob_dict)

    def loglik(self, wts):
        self.weights = wts
        results = self.batch_prob_classify([fs for (fs,l) in self.traintokens])
        ll = [pdist.prob(l) for ((fs, l), pdist) in zip(self.traintokens, results)]
        neglog = -1 * (np.log(float(sum(ll))/len(ll)))
        return neglog

    def objective(self, wts):
        ll = self.loglik(wts)
        objective = ll + ((self.tau / 2) * ((np.linalg.norm(self.weights, ord=2))**2))
        return objective

    def calculate_estimated_fcount(self):
        fcount = np.zeros(self.encoding.length(), 'd')
        for tok, label in self.traintokens:
            pdist = self.prob_classify(tok)
            for label in pdist.samples():
                prob = pdist.prob(label)
                for (fid, fval) in self.encoding.encode(tok, label):
                    fcount[fid] += prob*fval
        for tok, label in self.traintokens:
            pdist = self.prob_classify(tok)
        print(pdist.prob('n'), pdist.prob('v'))
        return fcount

    def accuracy(self, dataset):
        results = self.batch_classify([fs for (fs, l) in dataset])
        correct = [l == r for ((fs,l),r) in zip(dataset, results)]
        if correct:
            return float(sum(correct))/len(correct)
        else:
            return 0

    def gradient(self, wts):
        self.weights = wts
        return -1 * (self.emp_fcount - self.calculate_estimated_fcount())

    def callback(self, wts):
        self.weights = wts
        self.cnt += 1
        tracc = self.accuracy(self.traintokens)
        devacc = self.accuracy(self.develtokens)
        self.savedevacc.append(devacc)
        self.saveobj.append(self.objective(wts))
        self.savenorm.append(np.linalg.norm(wts, ord=2))
        print self.cnt, tracc, devacc

    def train(self):
        if self.tp == 0:
            self.weights = so.fmin_bfgs(self.objective, self.weights, fprime=self.gradient, disp=True, full_output=1, retall=1, maxiter=self.itr, callback=self.callback)
        elif self.tp == 1:
            self.weights = so.fmin_cg(self.objective, self.weights, fprime=self.gradient, disp=True, full_output=1, retall=1, maxiter=self.itr, callback=self.callback)
        elif self.tp == 2:
            self.weights = so.fmin_ncg(self.objective, self.weights, fprime=self.gradient, disp=True, full_output=1, retall=1, maxiter=self.itr, callback=self.callback)
        elif self.tp == 3:
            self.weights = so.fmin_l_bfgs_b(self.objective, self.weights, fprime=self.gradient, disp=True, maxiter=self.itr, callback=self.callback)
        elif self.tp == 4:
            self.weights = so.fmin(self.objective, self.weights, disp=True, maxiter=self.itr, callback=self.callback)
        elif self.tp == 5:
            self.weights = so.fmin_powell(self.objective, self.weights, maxiter=self.itr, disp=True, full_output=1, retall=1, callback=self.callback)
        elif self.tp == 6:
            self.weights = so.fmin_tnc(self.objective, self.weights, fprime=self.gradient, callback=self.callback)

    def saveall(self):
        np.savetxt('devacc_type_'+self.tp+'_iter_'+self.cnt+'.txt', self.savedevacc, fmt='%f')
        np.savetxt('norm_type_'+self.tp+'_iter_'+self.cnt+'.txt', self.savenorm, fmt='%f')
        np.savetxt('obj_type_'+self.tp+'_iter_'+self.cnt+'.txt', self.savenorm, fmt='%f')
