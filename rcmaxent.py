#!/usr/bin/python

import scipy.io as sio
import numpy as np
import combo_me as co
#import maxent_new as maxent
import maxentnew as maxent
import sys

samples = int(sys.argv[1])
inp = str(sys.argv[2])
eta = float(sys.argv[3])
tau = float(sys.argv[4])
numbers = int(sys.argv[5])
ppt = str(sys.argv[6])


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


encoding = co.ComboMaxentFeatEncoding.train(traindata, phih, phim, maph, mapm, pptype=ppt)
traintoks = encoding.train_toks()
traintokens = [(co.word_features(t),l) for t,l in encoding.train_toks()]
print 'type = ', inp, 'total samples = ', samples
print "total training examples for the pptype - ", ppt, len(traintoks)
devencode = co.ComboMaxentFeatEncoding.train(devdata, phidh, phidm, mapdh, mapdm, pptype=ppt)
devtoks = devencode.train_toks()
devtokens = [(co.word_features(t),l) for t,l in devencode.train_toks()]
print "total development examples for the pptype - ", ppt, len(devtoks)

def decreasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def increasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def minlog(L):
    val, idx = min((val, idx) for (idx, val) in enumerate(L))
    return val
if inp == 'None':
    print 'calling function type ', inp, ' with tau = ', tau, ' and eta = ', eta
    cl = maxent.Maxent.train(train_toks=traintokens, algorithm='gd', max_iter=100, eta=eta, devset=devtokens, tau=tau)
    np.savetxt('lin-models/devacc'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+str(ppt)+'.txt', cl[3], fmt='%f')
    np.savetxt('lin-models/lobjective'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+str(ppt)+'.txt', cl[2], fmt='%f')
    np.savetxt('lin-models/tracc'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+str(ppt)+'.txt', cl[1], fmt='%f')
    np.savetxt('lin-models/wtln'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+str(ppt)+'.txt', cl[0].weights_n(), fmt='%f')
    np.savetxt('lin-models/wtlv'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+str(ppt)+'.txt', cl[0].weights_v(), fmt='%f')

elif inp == 'l1':
    print 'calling function type ', inp, ' with tau = ', tau, ' and LC = ', eta
    cl = maxent.Maxent.train(train_toks=traintokens, algorithm='gd', max_iter=100, devset=devtokens, tau=tau, norm='l1', LC=eta)
    np.savetxt('lin-models/devacc'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+str(ppt)+'.txt', cl[3], fmt='%f')
    np.savetxt('lin-models/lobjective'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+str(ppt)+'.txt', cl[2], fmt='%f')
    np.savetxt('lin-models/tracc'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+str(ppt)+'.txt', cl[1], fmt='%f')
    np.savetxt('lin-models/wtln'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+str(ppt)+'.txt', cl[0].weights_n(), fmt='%f')
    np.savetxt('lin-models/wtlv'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+str(ppt)+'.txt', cl[0].weights_v(), fmt='%f')
    print '------------- BEST DEVACC SCORE ================== ========== ', cl[5], ' -------------'
    np.savetxt('lin-models/bestlinwtln'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+str(ppt)+'.txt', cl[4][0], fmt='%f')
    np.savetxt('lin-models/bestlinwtlv'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+str(ppt)+'.txt', cl[4][1], fmt='%f')

elif inp == 'l2proximal':
    print 'calling function type ', inp, ' with tau = ', tau, ' and LC = ', eta
    cl = maxent.Maxent.train(train_toks=traintokens, algorithm='gd', max_iter=100, devset=devtokens, tau=tau, norm='l2proximal', LC=eta)
    np.savetxt('lin-models/devacc'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+str(ppt)+'.txt', cl[3], fmt='%f')
    np.savetxt('lin-models/lobjective'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+str(ppt)+'.txt', cl[2], fmt='%f')
    np.savetxt('lin-models/tracc'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+str(ppt)+'.txt', cl[1], fmt='%f')
    np.savetxt('lin-models/wtln'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+str(ppt)+'.txt', cl[0].weights_n(), fmt='%f')
    np.savetxt('lin-models/wtlv'+inp+str(samples)+'tau'+str(tau)+'lc'+str(eta)+str(ppt)+'.txt', cl[0].weights_v(), fmt='%f')

    print '------------- BEST DEVACC SCORE ================== ========== ', cl[5], ' -------------'
    np.savetxt('lin-models/bestlinwtln'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+str(ppt)+'.txt', cl[4][0], fmt='%f')
    np.savetxt('lin-models/bestlinwtlv'+inp+str(samples)+'tau'+str(tau)+'eta'+str(eta)+str(ppt)+'.txt', cl[4][1], fmt='%f')

