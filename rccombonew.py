#!/usr/bin/python

import scipy.io as sio
import numpy as np
#import combo_me as co
import fix_combo_me as co
#import maxent_new as maxent
import bilinear_me as bilme
import sys

samples = int(sys.argv[1])
inp = str(sys.argv[2])
eta = float(sys.argv[3])
tau_l = float(sys.argv[4])
tau_b = float(sys.argv[5])
numbers = int(sys.argv[6])
ppt = str(sys.argv[7])
fx = int(sys.argv[8])
#ppt=None
print 'pptype = ', ppt
print 'fix type = ', fx
#if float(sys.argv[5]):
#    eta2 = float(sys.argv[5])
#    eta = np.random.uniform(float(sys.argv[3]), float(sys.argv[5]), 10)
#else:
#    eta = eta1

traindata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in open('clean/cleantrain.txt')]
devdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in open('clean/cleandev.txt')]
testdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in open('clean/cleantest.txt')]
traindata = traindata[:samples]
phih = sio.mmread('clean/trh1k.mtx')
phim = sio.mmread('clean/trm1k.mtx')
phidh = sio.mmread('clean/devh1k.mtx')
phidm = sio.mmread('clean/devm1k.mtx')
maph = np.loadtxt('clean/forhead.txt', dtype=str)
mapm = np.loadtxt('clean/formod.txt', dtype=str)
mapdh = np.loadtxt('clean/devheads.txt', dtype=str)
mapdm = np.loadtxt('clean/devmods.txt', dtype=str)


encoding = co.ComboMaxentFeatEncoding.train(traindata, phih, phim, maph, mapm, pptype=ppt, fix=fx)
traintoks = encoding.train_toks()
traintokens = [(co.word_features(t),l) for t,l in encoding.train_toks()]
print 'type = ', inp, 'total samples = ', samples
print "total training examples for the pptype - None ", len(traintoks)

devencode = co.ComboMaxentFeatEncoding.train(devdata, phidh, phidm, mapdh, mapdm, pptype=ppt, fix=fx)
devtoks = devencode.train_toks()
devtokens = [(co.word_features(t),l) for t,l in devencode.train_toks()]
print "total development examples for the pptype - None ", len(devtoks)

def decreasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def increasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def minlog(L):
    val, idx = min((val, idx) for (idx, val) in enumerate(L))
    return val

if inp == 'None':
    print 'calling function type ', inp, ' With tau_l and tau_b = ', tau_l, tau_b, ' and eta = ', eta

    cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=numbers, eta=eta, devset=devtoks, devencode=devencode, tau_l=tau_l, tau_b=tau_b)
    np.savetxt('combo-models/comlog'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[2], fmt='%f')
    np.savetxt('combo-models/comtracc'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[1], fmt='%f')
    np.savetxt('combo-models/comdevacc'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[3], fmt='%f')
    sio.mmwrite('combo-models/comwtbn'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', cl[0].weight_bn())
    sio.mmwrite('combo-models/comwtbv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', cl[0].weight_bv())
    np.savetxt('combo-models/comwtln'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[0].weight_ln(), fmt='%f')
    np.savetxt('combo-models/comwtlv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[0].weight_lv(), fmt='%f')
    print '-------------Norm sums = ====================', np.sum(cl[0].weight_bn()), np.sum(cl[0].weight_bv())
    print '------------- BEST DEVACC SCORE ================== ========== ', cl[4], ' -------------'
    print '-------------Bilinear norm sum = ====================', np.sum(cl[5][0]), np.sum(cl[5][1])
    sio.mmwrite('combo-models/bestwtbn'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', ((cl[5])[0]))
    sio.mmwrite('combo-models/bestwtbv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', ((cl[5])[1]))
    np.savetxt('combo-models/bestwtln'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', ((cl[5])[2]), fmt='%f')
    np.savetxt('combo-models/bestwtlv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', ((cl[5])[3]), fmt='%f')
    np.savetxt('combo-models/combonorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[6], fmt='%f')
    np.savetxt('combo-models/linnorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[7], fmt='%f')
    np.savetxt('combo-models/bilnorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[8], fmt='%f')


if inp == 'l2pl2p':
    print 'calling function type ', inp, ' With tau_l and tau_b = ', tau_l, tau_b, ' and lc = ', eta

    cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=numbers, devset=devtoks, devencode=devencode, tau_l=tau_l, tau_b=tau_b, l_penalty='l2p', b_penalty='l2p', LC_l=eta, LC_b=eta)
    np.savetxt('combo-models/comlog'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[2], fmt='%f')
    np.savetxt('combo-models/comtracc'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[1], fmt='%f')
    np.savetxt('combo-models/comdevacc'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[3], fmt='%f')
    sio.mmwrite('combo-models/comwtbn'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.mtx', cl[0].weight_bn())
    sio.mmwrite('combo-models/comwtbv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.mtx', cl[0].weight_bv())
    np.savetxt('combo-models/comwtln'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[0].weight_ln(), fmt='%f')
    np.savetxt('combo-models/comwtlv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[0].weight_lv(), fmt='%f')
    print '-------------Norm sums = ====================', np.sum(cl[0].weight_bn()), np.sum(cl[0].weight_bv())
    print '------------- BEST DEVACC SCORE ================== ========== ', cl[4], ' -------------'
    print '-------------Bilinear norm sum = ====================', np.sum(cl[5][0]), np.sum(cl[5][1])
    sio.mmwrite('combo-models/bestwtbn'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', ((cl[5])[0]))
    sio.mmwrite('combo-models/bestwtbv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', ((cl[5])[1]))
    np.savetxt('combo-models/bestwtln'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', ((cl[5])[2]), fmt='%f')
    np.savetxt('combo-models/bestwtlv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', ((cl[5])[3]), fmt='%f')
    np.savetxt('combo-models/combonorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[6], fmt='%f')
    np.savetxt('combo-models/linnorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[7], fmt='%f')
    np.savetxt('combo-models/bilnorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[8], fmt='%f')



if inp == 'l2pl1':
    print 'calling function type ', inp, ' With tau_l and tau_b = ', tau_l, tau_b, ' and lc = ', eta

    cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=numbers, devset=devtoks, devencode=devencode, tau_l=tau_l, tau_b=tau_b, l_penalty='l2p', b_penalty='l1', LC_l=eta, LC_b=eta)
    np.savetxt('combo-models/comlog'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[2], fmt='%f')
    np.savetxt('combo-models/comtracc'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[1], fmt='%f')
    np.savetxt('combo-models/comdevacc'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[3], fmt='%f')
    sio.mmwrite('combo-models/comwtbn'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.mtx', cl[0].weight_bn())
    sio.mmwrite('combo-models/comwtbv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.mtx', cl[0].weight_bv())
    np.savetxt('combo-models/comwtln'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[0].weight_ln(), fmt='%f')
    np.savetxt('combo-models/comwtlv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[0].weight_lv(), fmt='%f')
    print '-------------Norm sums = ====================', np.sum(cl[0].weight_bn()), np.sum(cl[0].weight_bv())
    print '------------- BEST DEVACC SCORE ================== ========== ', cl[4], ' -------------'
    print '-------------Bilinear norm sum = ====================', np.sum(cl[5][0]), np.sum(cl[5][1])
    sio.mmwrite('combo-models/bestwtbn'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', ((cl[5])[0]))
    sio.mmwrite('combo-models/bestwtbv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', ((cl[5])[1]))
    np.savetxt('combo-models/bestwtln'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', ((cl[5])[2]), fmt='%f')
    np.savetxt('combo-models/bestwtlv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', ((cl[5])[3]), fmt='%f')
    np.savetxt('combo-models/combonorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[6], fmt='%f')
    np.savetxt('combo-models/linnorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[7], fmt='%f')
    np.savetxt('combo-models/bilnorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[8], fmt='%f')




if inp == 'l2pnn':
    print 'calling function type ', inp, ' With tau_l and tau_b = ', tau_l, tau_b, ' and lc = ', eta

    cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=numbers, devset=devtoks, devencode=devencode, tau_l=tau_l, tau_b=tau_b, l_penalty='l2p', b_penalty='nn', LC_l=eta, LC_b=eta)
    np.savetxt('combo-models/comlog'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[2], fmt='%f')
    np.savetxt('combo-models/comtracc'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[1], fmt='%f')
    np.savetxt('combo-models/comdevacc'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[3], fmt='%f')
    sio.mmwrite('combo-models/comwtbn'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.mtx', cl[0].weight_bn())
    sio.mmwrite('combo-models/comwtbv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.mtx', cl[0].weight_bv())
    np.savetxt('combo-models/comwtln'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[0].weight_ln(), fmt='%f')
    np.savetxt('combo-models/comwtlv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[0].weight_lv(), fmt='%f')
    print '-------------Norm sums = ====================', np.sum(cl[0].weight_bn()), np.sum(cl[0].weight_bv())
    print '------------- BEST DEVACC SCORE ================== ========== ', cl[4], ' -------------'
    print '-------------Bilinear norm sum = ====================', np.sum(cl[5][0]), np.sum(cl[5][1])
    sio.mmwrite('combo-models/bestwtbn'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', ((cl[5])[0]))
    sio.mmwrite('combo-models/bestwtbv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', ((cl[5])[1]))
    np.savetxt('combo-models/bestwtln'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', ((cl[5])[2]), fmt='%f')
    np.savetxt('combo-models/bestwtlv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', ((cl[5])[3]), fmt='%f')
    np.savetxt('combo-models/combonorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[6], fmt='%f')
    np.savetxt('combo-models/linnorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[7], fmt='%f')
    np.savetxt('combo-models/bilnorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[8], fmt='%f')




if inp == 'l1nn':
    print 'calling function type ', inp, ' With tau_l and tau_b = ', tau_l, tau_b, ' and lc = ', eta

    cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=numbers, devset=devtoks, devencode=devencode, tau_l=tau_l, tau_b=tau_b, l_penalty='l1', b_penalty='nn', LC_l=eta, LC_b=eta)
    np.savetxt('combo-models/comlog'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[2], fmt='%f')
    np.savetxt('combo-models/comtracc'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[1], fmt='%f')
    np.savetxt('combo-models/comdevacc'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[3], fmt='%f')
    sio.mmwrite('combo-models/comwtbn'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.mtx', cl[0].weight_bn())
    sio.mmwrite('combo-models/comwtbv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.mtx', cl[0].weight_bv())
    np.savetxt('combo-models/comwtln'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[0].weight_ln(), fmt='%f')
    np.savetxt('combo-models/comwtlv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[0].weight_lv(), fmt='%f')
    print '------------- BEST DEVACC SCORE ================== ========== ', cl[4], ' -------------'
    print '-------------Bilinear norm sum = ====================', np.sum(cl[5][0]), np.sum(cl[5][1])
    sio.mmwrite('combo-models/bestwtbn'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', ((cl[5])[0]))
    sio.mmwrite('combo-models/bestwtbv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', ((cl[5])[1]))
    np.savetxt('combo-models/bestwtln'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', ((cl[5])[2]), fmt='%f')
    np.savetxt('combo-models/bestwtlv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', ((cl[5])[3]), fmt='%f')
    np.savetxt('combo-models/combonorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[6], fmt='%f')
    np.savetxt('combo-models/linnorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[7], fmt='%f')
    np.savetxt('combo-models/bilnorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[8], fmt='%f')



if inp == 'l1l1':
    print 'calling function type ', inp, ' With tau_l and tau_b = ', tau_l, tau_b, ' and lc = ', eta

    cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=numbers, devset=devtoks, devencode=devencode, tau_l=tau_l, tau_b=tau_b, l_penalty='l1', b_penalty='l1', LC_l=eta, LC_b=eta)
    np.savetxt('combo-models/comlog'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[2], fmt='%f')
    np.savetxt('combo-models/comtracc'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[1], fmt='%f')
    np.savetxt('combo-models/comdevacc'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[3], fmt='%f')
    sio.mmwrite('combo-models/comwtbn'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.mtx', cl[0].weight_bn())
    sio.mmwrite('combo-models/comwtbv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.mtx', cl[0].weight_bv())
    np.savetxt('combo-models/comwtln'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[0].weight_ln(), fmt='%f')
    np.savetxt('combo-models/comwtlv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[0].weight_lv(), fmt='%f')
    print '------------- BEST DEVACC SCORE ================== ========== ', cl[4], ' -------------'
    print '-------------Bilinear norm sum = ====================', np.sum(cl[5][0]), np.sum(cl[5][1])
    sio.mmwrite('combo-models/bestwtbn'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', ((cl[5])[0]))
    sio.mmwrite('combo-models/bestwtbv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', ((cl[5])[1]))
    np.savetxt('combo-models/bestwtln'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', ((cl[5])[2]), fmt='%f')
    np.savetxt('combo-models/bestwtlv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', ((cl[5])[3]), fmt='%f')
    np.savetxt('combo-models/combonorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[6], fmt='%f')
    np.savetxt('combo-models/linnorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[7], fmt='%f')
    np.savetxt('combo-models/bilnorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[8], fmt='%f')




if inp == 'l1l2p':
    print 'calling function type ', inp, ' With tau_l and tau_b = ', tau_l, tau_b, ' and lc = ', eta

    cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=numbers, devset=devtoks, devencode=devencode, tau_l=tau_l, tau_b=tau_b, l_penalty='l1', b_penalty='l2p', LC_l=eta, LC_b=eta)
    np.savetxt('combo-models/comlog'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[2], fmt='%f')
    np.savetxt('combo-models/comtracc'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[1], fmt='%f')
    np.savetxt('combo-models/comdevacc'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[3], fmt='%f')
    sio.mmwrite('combo-models/comwtbn'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.mtx', cl[0].weight_bn())
    sio.mmwrite('combo-models/comwtbv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.mtx', cl[0].weight_bv())
    np.savetxt('combo-models/comwtln'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[0].weight_ln(), fmt='%f')
    np.savetxt('combo-models/comwtlv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'lc'+str(eta)+ppt+'.txt', cl[0].weight_lv(), fmt='%f')
    print '-------------Norm sums = ====================', np.sum(cl[0].weight_bn()), np.sum(cl[0].weight_bv())
    print '------------- BEST DEVACC SCORE ================== ========== ', cl[4], ' -------------'
    print '-------------Bilinear norm sums = ====================', np.sum(cl[5][0]), np.sum(cl[5][1])
    sio.mmwrite('combo-models/bestwtbn'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', ((cl[5])[0]))
    sio.mmwrite('combo-models/bestwtbv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.mtx', ((cl[5])[1]))
    np.savetxt('combo-models/bestwtln'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', ((cl[5])[2]), fmt='%f')
    np.savetxt('combo-models/bestwtlv'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', ((cl[5])[3]), fmt='%f')
    np.savetxt('combo-models/combonorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[6], fmt='%f')
    np.savetxt('combo-models/linnorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[7], fmt='%f')
    np.savetxt('combo-models/bilnorm'+inp+str(samples)+'cl'+str(tau_l)+'cb'+str(tau_b)+'eta'+str(eta)+ppt+'.txt', cl[8], fmt='%f')



