from __future__ import division
import numpy as np
import scipy.sparse as ss
import sys
import fix_combo_me as co
import bilmenew as bilme
import maxentnew as maxent
import scipy.io as sio

ppt = sys.argv[1]
samples = sys.argv[2]
l_tau = sys.argv[3]
l_lc = sys.argv[4]
l_regtype = sys.argv[5]

b_tau = sys.argv[6]
b_lc = sys.argv[7]
b_regtype = sys.argv[8]

gamma = float(sys.argv[9])

def load(ppt, samples, l_tau, l_lc, l_regtype, b_tau, b_lc, b_regtype):

    ln = np.loadtxt('lin-models/bestlinwtln'+l_regtype+samples+'tau'+l_tau+'lc'+l_lc+ppt+'.txt')
    lv = np.loadtxt('lin-models/bestlinwtlv'+l_regtype+samples+'tau'+l_tau+'lc'+l_lc+ppt+'.txt')
    bv = np.loadtxt('bil-models/bestbilwtbn'+b_regtype+samples+'tau'+b_tau+'eta'+b_lc+ppt+'.txt')
    bn = np.loadtxt('bil-models/bestbilwtbv'+b_regtype+samples+'tau'+b_tau+'eta'+b_lc+ppt+'.txt')

    traindata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in open('clean/cleantrain.txt')]
    devdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in open('clean/cleandev.txt')]
    testdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in open('clean/cleantest.txt')]

    traindata = traindata[:int(samples)]

    phih = sio.mmread('clean/trh1k.mtx')
    phim = sio.mmread('clean/trm1k.mtx')
    phidh = sio.mmread('clean/devh1k.mtx')
    phidm = sio.mmread('clean/devm1k.mtx')
    maph = np.loadtxt('clean/forhead.txt', dtype=str)
    mapm = np.loadtxt('clean/formod.txt', dtype=str)
    mapdh = np.loadtxt('clean/devheads.txt', dtype=str)
    mapdm = np.loadtxt('clean/devmods.txt', dtype=str)


    trainingdat = bilme.BilinearMaxentFeatEncoding.train(traindata, phih, phim, maph, mapm, pptype=ppt)
    traintoks = trainingdat.train_toks()
    traintokens = [(co.word_features(t),l) for t,l in trainingdat.train_toks()]
    devencode = bilme.BilinearMaxentFeatEncoding.train(devdata, phidh, phidm, mapdh, mapdm, pptype=ppt)
    devtoks = devencode.train_toks()
    devtokens = [(co.word_features(t),l) for t,l in devencode.train_toks()]

    data = [devtoks, devtokens]

    trlinencoding = maxent.BinaryMaxentFeatureEncoding.train(traintokens)

    return trlinencoding, devencode, [ln, lv], [bn, bv], data
#    if b_regtype == l1:
#        lcb = 'lc'
#    else:
#        lcb = 'eta'


def calculate_accuracy(lencoding, bencoding, lweights, bweights, data, gamma=0.5):
    '''

    Computes accuracy when the weights are selected!

    '''

    gold = data[0]
    featureset = data[1]

    bn = np.matrix(bweights[0])
    bv = np.matrix(bweights[1])

    ln = np.array(lweights[0])
    lv = np.array(lweights[1])

    count_samples = 0
    score = []

    for (tok, label) in gold:
        scoreln = 0
        scorelv = 0
        prob = {}
        feature_vector_n = lencoding.encode(featureset[count_samples][0], 'n')
        feature_vector_v = lencoding.encode(featureset[count_samples][0], 'v')
        v, n, m = bencoding.bil_u_encode(tok)
        scorebn = (n*(bn*m.T))[0,0]
        scorebv = (v*(bv*m.T))[0,0]

        for (f_id, f_val) in feature_vector_n:
            scoreln += ln[f_id] * f_val

        for (f_id, f_val) in feature_vector_v:
            scorelv += lv[f_id] * f_val

        exp_bn = np.exp(scorebn) / (np.exp(scorebn) + np.exp(scorebv))
        exp_bv = np.exp(scorebv) / (np.exp(scorebn) + np.exp(scorebv))
        exp_ln = np.exp(scoreln) / (np.exp(scoreln) + np.exp(scorelv))
        exp_lv = np.exp(scorelv) / (np.exp(scoreln) + np.exp(scorelv))

        prob['n'] = gamma * exp_ln + (1 - gamma) * exp_bn
        prob['v'] = gamma * exp_lv + (1 - gamma) * exp_bv

        if label == max((p,v) for (v,p) in prob.items())[1]:
            score.append(1)

        count_samples += 1

    return float(np.sum(score)) / count_samples


le, be, lw, bw, d = load(ppt, samples, l_tau, l_lc, l_regtype, b_tau, b_lc, b_regtype)

precision = calculate_accuracy(le, be, lw, bw, d, gamma)
print precision
