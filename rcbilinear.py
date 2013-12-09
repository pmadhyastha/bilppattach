#!/usr/bin/python

import scipy.io as sio
import numpy as np
import combo_me as co
#import maxent_new as maxent
import bilinear_me as bilme
import sys

samples = int(sys.argv[1])
inp = str(sys.argv[2])
#cb = float(sys.argv[3])

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


encoding = bilme.BilinearMaxentFeatEncoding.train(traindata, phih, phim, maph, mapm, pptype=None)
traintoks = encoding.train_toks()
traintokens = [(co.word_features(t),l) for t,l in encoding.train_toks()]
print 'type = ', inp, 'total samples = ', samples
print "total training examples for the pptype - None ", len(traintoks)
devencode = bilme.BilinearMaxentFeatEncoding.train(devdata, phidh, phidm, mapdh, mapdm, pptype=None)
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
tau_vals = [1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.000000001, 0.0000000001, 0.00000000001]
eta = [90000000, 100000000]
if inp == 'None':
    fmins = {}
    tau = 1
    for e in eta:
        main_string = str(e)+','+str(tau)
        try:
            print 'calling function type ', inp, ' with tau = ', tau, ' and eta = ', e
            cl = bilme.BilinearMaxent.train(train_toks=traintokens, algorithm='gd', max_iter=40, eta=e, devset=devtokens, tau=tau)
            np.savetxt('lintracc25'+inp+str(samples)+'tau'+str(tau)+'eta'+str(e)+'.txt', cl[1], fmt='%f')
            np.savetxt('linlog25'+inp+str(samples)+'tau'+str(tau)+'eta'+str(e)+'.txt', cl[2], fmt='%f')
            np.savetxt('lindevacc25'+inp+str(samples)+'tau'+str(tau)+'eta'+str(e)+'.txt', cl[3], fmt='%f')
            fmins[main_string] = minlog(cl[2])
        except:
            print 'problem with eta = ', e, ' tau = ', tau

    mstr = sorted(fmins, key=fmins.get)
    final = mstr
    print final
    for i in range(len(final)):
        try:
            val = (final[i]).split(',')
            cl = maxent.Maxent.train(train_toks=traintokens, algorithm='gd', max_iter=100, eta=float(val[0]), devset=devtokens, tau=float(val[1]))
            np.savetxt('devacc'+inp+str(samples)+'tau'+val[1]+'eta'+val[0]+'.txt', cl[3], fmt='%f')
            np.savetxt('neglogl'+inp+str(samples)+'tau'+val[1]+'eta'+val[0]+'.txt', cl[2], fmt='%f')
            np.savetxt('tracc'+inp+str(samples)+'tau'+val[1]+'eta'+val[0]+'.txt', cl[1], fmt='%f')
            np.savetxt('wtln'+inp+str(samples)+'tau'+val[1]+'eta'+val[0]+'.txt', cl[0].weights_n(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'tau'+val[1]+'eta'+val[0]+'.txt', cl[0].weights_v(), fmt='%f')
        except:
            pass

elif inp == 'l2':
    fmins = {}
    for tau in tau_vals:
        for e in eta:
            main_string = str(e)+','+str(tau)
            try:
                print 'calling function type ', inp, ' with tau = ', tau, ' and eta = ', e
                cl = maxent.Maxent.train(train_toks=traintokens, algorithm='gd', max_iter=40, eta=e, devset=devtokens, tau=tau, norm='l2')
                np.savetxt('lintracc25'+inp+str(samples)+'tau'+str(tau)+'eta'+str(e)+'.txt', cl[1], fmt='%f')
                np.savetxt('linlog25'+inp+str(samples)+'tau'+str(tau)+'eta'+str(e)+'.txt', cl[2], fmt='%f')
                np.savetxt('lindevacc25'+inp+str(samples)+'tau'+str(tau)+'eta'+str(e)+'.txt', cl[3], fmt='%f')
                fmins[main_string] = minlog(cl[2])
            except:
                print 'problem with eta = ', e, ' tau = ', tau

    mstr = sorted(fmins, key=fmins.get)
    final = mstr
    print final
    for i in range(len(final)):
        try:
            val = (final[i]).split(',')
            cl = maxent.Maxent.train(train_toks=traintokens, algorithm='gd', max_iter=100, eta=float(val[0]), devset=devtokens, tau=float(val[1]), norm='l2')
            np.savetxt('devacc'+inp+str(samples)+'tau'+val[1]+'eta'+val[0]+'.txt', cl[3], fmt='%f')
            np.savetxt('neglogl'+inp+str(samples)+'tau'+val[1]+'eta'+val[0]+'.txt', cl[2], fmt='%f')
            np.savetxt('tracc'+inp+str(samples)+'tau'+val[1]+'eta'+val[0]+'.txt', cl[1], fmt='%f')
            np.savetxt('wtln'+inp+str(samples)+'tau'+val[1]+'eta'+val[0]+'.txt', cl[0].weights_n(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'tau'+val[1]+'eta'+val[0]+'.txt', cl[0].weights_v(), fmt='%f')

        except:
            pass

elif inp == 'l1':
    lc = 0.05
    fmins = {}
    for tau in tau_vals:
        main_string = str(lc)+','+str(tau)
        try:
            print 'calling function type ', inp, ' with tau = ', tau, ' and LC = ', lc
            cl = maxent.Maxent.train(train_toks=traintokens, algorithm='gd', max_iter=40, devset=devtokens, tau=tau, norm='l1', LC=lc)
            np.savetxt('lintracc25'+inp+str(samples)+'tau'+str(tau)+'lc'+str(lc)+'.txt', cl[1], fmt='%f')
            np.savetxt('linlog25'+inp+str(samples)+'tau'+str(tau)+'lc'+str(lc)+'.txt', cl[2], fmt='%f')
            np.savetxt('lindevacc25'+inp+str(samples)+'tau'+str(tau)+'lc'+str(lc)+'.txt', cl[3], fmt='%f')
            fmins[main_string] = minlog(cl[2])
        except:
            print 'problem with LC = ', lc, ' tau = ', tau

    mstr = sorted(fmins, key=fmins.get)
    final = mstr
    print final
    for i in range(len(final)):
        try:
            val = (final[i]).split(',')
            cl = maxent.Maxent.train(train_toks=traintokens, algorithm='gd', max_iter=100, LC=float(val[0]), devset=devtokens, tau=float(val[1]), norm='l1')
            np.savetxt('devacc'+inp+str(samples)+'tau'+val[1]+'lc'+val[0]+'.txt', cl[3], fmt='%f')
            np.savetxt('neglogl'+inp+str(samples)+'tau'+val[1]+'lc'+val[0]+'.txt', cl[2], fmt='%f')
            np.savetxt('tracc'+inp+str(samples)+'tau'+val[1]+'lc'+val[0]+'.txt', cl[1], fmt='%f')
            np.savetxt('wtln'+inp+str(samples)+'tau'+val[1]+'lc'+val[0]+'.txt', cl[0].weights_n(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'tau'+val[1]+'lc'+val[0]+'.txt', cl[0].weights_v(), fmt='%f')

        except:
            pass

elif inp == 'l2proximal':
    lc = 0.05
    fmins = {}
    for tau in tau_vals:
        main_string = str(lc)+','+str(tau)
        try:
            print 'calling function type ', inp, ' with tau = ', tau, ' and LC = ', lc
            cl = maxent.Maxent.train(train_toks=traintokens, algorithm='gd', max_iter=40, devset=devtokens, tau=tau, norm='l2proximal', LC=lc)
            np.savetxt('lintracc25'+inp+str(samples)+'tau'+str(tau)+'lc'+str(lc)+'.txt', cl[1], fmt='%f')
            np.savetxt('linlog25'+inp+str(samples)+'tau'+str(tau)+'lc'+str(lc)+'.txt', cl[2], fmt='%f')
            np.savetxt('lindevacc25'+inp+str(samples)+'tau'+str(tau)+'lc'+str(lc)+'.txt', cl[3], fmt='%f')
            fmins[main_string] = minlog(cl[2])
        except:
            print 'problem with LC = ', lc, ' tau = ', tau

    mstr = sorted(fmins, key=fmins.get)
    final = mstr
    print final
    for i in range(len(final)):
        try:
            val = (final[i]).split(',')
            cl = maxent.Maxent.train(train_toks=traintokens, algorithm='gd', max_iter=100, LC=float(val[0]), devset=devtokens, tau=float(val[1]), norm='l2proximal')
            np.savetxt('devacc'+inp+str(samples)+'tau'+val[1]+'lc'+val[0]+'.txt', cl[3], fmt='%f')
            np.savetxt('neglogl'+inp+str(samples)+'tau'+val[1]+'lc'+val[0]+'.txt', cl[2], fmt='%f')
            np.savetxt('tracc'+inp+str(samples)+'tau'+val[1]+'lc'+val[0]+'.txt', cl[1], fmt='%f')
            np.savetxt('wtln'+inp+str(samples)+'tau'+val[1]+'lc'+val[0]+'.txt', cl[0].weights_n(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'tau'+val[1]+'lc'+val[0]+'.txt', cl[0].weights_v(), fmt='%f')

        except:
            pass

