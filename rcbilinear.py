#!/usr/bin/python

import scipy.io as sio
import numpy as np
import combo_me as co
#import maxent_new as maxent
import bilinear_me as bilme
import sys

samples = int(sys.argv[1])
inp = str(sys.argv[2])
eta = float(sys.argv[3])
tau = float(sys.argv[4])
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


encoding = bilme.BilinearMaxentFeatEncoding.train(traindata, phih, phim, maph, mapm, pptype='for')
traintoks = encoding.train_toks()
traintokens = [(co.word_features(t),l) for t,l in encoding.train_toks()]
print 'type = ', inp, 'total samples = ', samples
print "total training examples for the pptype - None ", len(traintoks)
devencode = bilme.BilinearMaxentFeatEncoding.train(devdata, phidh, phidm, mapdh, mapdm, pptype='for')
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
    print 'calling function type ', inp, ' WindowsErrorth tau = ', tau, ' and eta = ', eta
    cl = bilme.BilinearMaxent.train(traintoks, encoding, max_iter=100, eta=eta, devset=devtoks, devencode=devencode, tau=tau)
    np.savetxt('bdevacc'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+'.txt', cl[3], fmt='%f')
    np.savetxt('bobjective'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+'.txt', cl[2], fmt='%f')
    np.savetxt('btracc'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+'.txt', cl[1], fmt='%f')
    np.savetxt('bilwtbn'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+'.txt', cl[0].weight_bn(), fmt='%f')
    np.savetxt('bilwtbv'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+'.txt', cl[0].weight_bv(), fmt='%f')

elif inp == 'l2':
    print 'calling function type ', inp, ' and tau = ', tau, ' and eta = ', eta
    cl = bilme.BilinearMaxent.train(traintoks, encoding, max_iter=30, eta=eta, devset=devtoks, devencode=devencode, tau=tau, penalty='l2')
    np.savetxt('bdevacc'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+'.txt', cl[3], fmt='%f')
    np.savetxt('bobjective'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+'.txt', cl[2], fmt='%f')
    np.savetxt('btracc'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+'.txt', cl[1], fmt='%f')
    np.savetxt('bilwtbn'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+'.txt', cl[0].weight_bn(), fmt='%f')
    np.savetxt('bilwtbv'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+'.txt', cl[0].weight_bv(), fmt='%f')

elif inp == 'l1':
    print 'calling function type ', inp, ' and tau = ', tau, ' and eta = ', eta
    cl = bilme.BilinearMaxent.train(traintoks, encoding, max_iter=30, LC=eta, devset=devtoks, devencode=devencode, tau=tau, penalty='l1')
    np.savetxt('bdevacc'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[3], fmt='%f')
    np.savetxt('bobjective'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[2], fmt='%f')
    np.savetxt('btracc'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[1], fmt='%f')
    np.savetxt('bilwtbn'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[0].weight_bn(), fmt='%f')
    np.savetxt('bilwtbv'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[0].weight_bv(), fmt='%f')

elif inp == 'l2p':
    print 'calling function type ', inp, ' and tau = ', tau, ' and eta = ', eta
    cl = bilme.BilinearMaxent.train(traintoks, encoding, max_iter=30, LC=eta, devset=devtoks, devencode=devencode, tau=tau, penalty='l2p')
    np.savetxt('bdevacc'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[3], fmt='%f')
    np.savetxt('bobjective'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[2], fmt='%f')
    np.savetxt('btracc'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[1], fmt='%f')
    np.savetxt('bilwtbn'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[0].weight_bn(), fmt='%f')
    np.savetxt('bilwtbv'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[0].weight_bv(), fmt='%f')


elif inp == 'nn':
    print 'calling function type ', inp, ' and tau = ', tau, ' and eta = ', eta
    cl = bilme.BilinearMaxent.train(traintoks, encoding, max_iter=500, LC=eta, devset=devtoks, devencode=devencode, tau=tau, penalty='nn')
    np.savetxt('bdevacc'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[3], fmt='%f')
    np.savetxt('bobjective'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[2], fmt='%f')
    np.savetxt('btracc'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[1], fmt='%f')
    np.savetxt('bilwtbn'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[0].weight_bn(), fmt='%f')
    np.savetxt('bilwtbv'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+'.txt', cl[0].weight_bv(), fmt='%f')

