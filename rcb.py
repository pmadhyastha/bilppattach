#!/usr/bin/python

import scipy.io as sio
import numpy as np
import combo_me as co
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


encoding = co.ComboMaxentFeatEncoding.train(traindata, phih, phim, maph, mapm, pptype='for')
traintoks = encoding.train_toks()
traintokens = [(co.word_features(t),l) for t,l in encoding.train_toks()]
print 'type = ', inp, 'total samples = ', samples
print "total training examples for the pptype - 'for' ", len(traintoks)
devencode = co.ComboMaxentFeatEncoding.train(devdata, phidh, phidm, mapdh, mapdm, pptype='for')
devtoks = devencode.train_toks()
devtokens = [(co.word_features(t),l) for t,l in devencode.train_toks()]
print "total development examples for the pptype - 'for' ", len(devtoks)

def decreasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def increasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def minlog(L):
    val, idx = min((val, idx) for (idx, val) in enumerate(L))
    return val

#tau_vals_a = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]
#eta = [1600, 1000, 800, 400, 200, 100, 50, 10]
#lcvals = [0.5, 0.05, 0.005, 0.0005, 0.00005]

#tau_vals_a = [0.0000001]
#eta = [100]
#lcvals = [0.5]

cls = 1
cb = 1
if inp == 'None':
    fmins = {}
    for e in eta:
        main_string = str(e)+','+str(cls)+','+str(cb)
        try:
            print 'calling function type ', inp, ' with cb and cls = ', cb, cls, ' and eta = ', e
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=1005, eta=e, devset=devtoks, devencode=devencode)
            np.savetxt('combotracc25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.txt', cl[1], fmt='%f')
            np.savetxt('comboobj25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.txt', cl[2], fmt='%f')
            np.savetxt('combodevacc25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.txt', cl[3], fmt='%f')
            fmins[main_string] = minlog(cl[2])
        except:
            print 'problem with eta = ', e, ' cls = ', cls, ' cb ', cb

    mstr = sorted(fmins, key=fmins.get)
    final = mstr
    print final
    for i in range(len(final)):
        try:
            val = (final[i]).split(',')
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, eta=float(val[0]), devset=devtoks, devencode=devencode, tau_l=float(val[1]), tau_b=float(val[2]))
            np.savetxt('log'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'eta'+val[0]+'.txt', cl[2], fmt='%f')
            np.savetxt('tracc'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'eta'+val[0]+'.txt', cl[1], fmt='%f')
            np.savetxt('devacc'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'eta'+val[0]+'.txt', cl[3], fmt='%f')
            sio.mmwrite('wtbn'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'eta'+val[0]+'.mtx', cl[0].weight_bn())
            sio.mmwrite('wtbv'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'eta'+val[0]+'.mtx', cl[0].weight_bv())
            np.savetxt('wtln'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'eta'+val[0]+'.txt', cl[0].weight_ln(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'eta'+val[0]+'.txt', cl[0].weight_lv(), fmt='%f')
        except:
            pass

if inp == 'l2l2':
    fmins = {}
    for cls in tau_vals_a:
        for cb in tau_vals_a:
            for e in eta:
                main_string = str(e)+','+str(cls)+','+str(cb)
                try:
                    print 'calling function type ', inp, ' with cb and cls = ', cb, cls, ' and eta = ', e
                    cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=1005, eta=e, devset=devtoks, devencode=devencode, tau_l=cls, tau_b=cb, l_penalty='l2', b_penalty='l2')
                    np.savetxt('combotracc25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.txt', cl[1], fmt='%f')
                    np.savetxt('comboobj25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.txt', cl[2], fmt='%f')
                    np.savetxt('combodevacc25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.txt', cl[3], fmt='%f')
                    fmins[main_string] = minlog(cl[2])
                except:
                     print 'problem with eta = ', e, ' cls = ', cls, ' cb ', cb

    mstr = sorted(fmins, key=fmins.get)
    final = mstr
    print final
    for i in range(len(final)):
        try:
            val = (final[i]).split(',')
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, eta=float(val[0]), devset=devtoks, devencode=devencode, tau_l=float(val[1]), tau_b=float(val[2]), l_penalty='l2', b_penalty='l2')
            np.savetxt('log'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'eta'+val[0]+'.txt', cl[2], fmt='%f')
            np.savetxt('tracc'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'eta'+val[0]+'.txt', cl[1], fmt='%f')
            np.savetxt('devacc'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'eta'+val[0]+'.txt', cl[3], fmt='%f')
            sio.mmwrite('wtbn'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'eta'+val[0]+'.mtx', cl[0].weight_bn())
            sio.mmwrite('wtbv'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'eta'+val[0]+'.mtx', cl[0].weight_bv())
            np.savetxt('wtln'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'eta'+val[0]+'.txt', cl[0].weight_ln(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'eta'+val[0]+'.txt', cl[0].weight_lv(), fmt='%f')
        except:
            pass


if inp == 'l2pl2p':
    fmins = {}
    for cls in tau_vals_a:
        for cb in tau_vals_a:
            for lc in lcvals:
                main_string = str(lc)+','+str(cls)+','+str(cb)
                try:
                    print 'calling function type ', inp, ' with cb and cls = ', cb, cls, ' and lc = ', lc
                    cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=1005, LC_l=lc, LC_b=lc, devset=devtoks, devencode=devencode, tau_l=cls, tau_b=cb, l_penalty='l2p', b_penalty='l2p')
                    np.savetxt('combotracc25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'lc'+str(lc)+'.txt', cl[1], fmt='%f')
                    np.savetxt('comboobj25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'lc'+str(lc)+'.txt', cl[2], fmt='%f')
                    np.savetxt('combodevacc25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'lc'+str(lc)+'.txt', cl[3], fmt='%f')
                    fmins[main_string] = minlog(cl[2])
                except:
                     print 'problem with eta = ', e, ' cls = ', cls, ' cb ', cb

    mstr = sorted(fmins, key=fmins.get)
    final = mstr
    print final
    for i in range(len(final)):
        try:
            val = (final[i]).split(',')
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, LC_l=float(val[0]), LC_b=float(val[0]), devset=devtoks, devencode=devencode, tau_l=float(val[1]), tau_b=float(val[2]), l_penalty='l2p', b_penalty='l2p')
            np.savetxt('log'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[2], fmt='%f')
            np.savetxt('tracc'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[1], fmt='%f')
            np.savetxt('devacc'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[3], fmt='%f')
            sio.mmwrite('wtbn'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.mtx', cl[0].weight_bn())
            sio.mmwrite('wtbv'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.mtx', cl[0].weight_bv())
            np.savetxt('wtln'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[0].weight_ln(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[0].weight_lv(), fmt='%f')
        except:
            pass

if inp == 'l2pnn':
    fmins = {}
    for cls in tau_vals_a:
        for cb in tau_vals_a:
            for lc in lcvals:
                main_string = str(lc)+','+str(cls)+','+str(cb)
                try:
                    print 'calling function type ', inp, ' with cb and cls = ', cb, cls, ' and lc = ', lc
                    cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=1005, LC_l=lc, LC_b=lc, devset=devtoks, devencode=devencode, tau_l=cls, tau_b=cb, l_penalty='l2p', b_penalty='nn')
                    np.savetxt('combotracc25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'lc'+str(lc)+'.txt', cl[1], fmt='%f')
                    np.savetxt('comboobj25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'lc'+str(lc)+'.txt', cl[2], fmt='%f')
                    np.savetxt('combodevacc25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'lc'+str(lc)+'.txt', cl[3], fmt='%f')
                    fmins[main_string] = minlog(cl[2])
                except:
                     print 'problem with eta = ', e, ' cls = ', cls, ' cb ', cb

    mstr = sorted(fmins, key=fmins.get)
    final = mstr
    print final
    for i in range(len(final)):
        try:
            val = (final[i]).split(',')
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, LC_l=float(val[0]), LC_b=float(val[0]), devset=devtoks, devencode=devencode, tau_l=float(val[1]), tau_b=float(val[2]), l_penalty='l2p', b_penalty='nn')
            np.savetxt('log'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[2], fmt='%f')
            np.savetxt('tracc'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[1], fmt='%f')
            np.savetxt('devacc'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[3], fmt='%f')
            sio.mmwrite('wtbn'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.mtx', cl[0].weight_bn())
            sio.mmwrite('wtbv'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.mtx', cl[0].weight_bv())
            np.savetxt('wtln'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[0].weight_ln(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[0].weight_lv(), fmt='%f')
        except:
            pass

if inp == 'l2pl1':
    fmins = {}
    for cls in tau_vals_a:
        for cb in tau_vals_a:
            for lc in lcvals:
                main_string = str(lc)+','+str(cls)+','+str(cb)
                try:
                    print 'calling function type ', inp, ' with cb and cls = ', cb, cls, ' and lc = ', lc
                    cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=1005, LC_l=lc, LC_b=lc, devset=devtoks, devencode=devencode, tau_l=cls, tau_b=cb, l_penalty='l2p', b_penalty='l1')
                    np.savetxt('combotracc25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'lc'+str(lc)+'.txt', cl[1], fmt='%f')
                    np.savetxt('comboobj25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'lc'+str(lc)+'.txt', cl[2], fmt='%f')
                    np.savetxt('combodevacc25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'lc'+str(lc)+'.txt', cl[3], fmt='%f')
                    fmins[main_string] = minlog(cl[2])
                except:
                     print 'problem with eta = ', e, ' cls = ', cls, ' cb ', cb

    mstr = sorted(fmins, key=fmins.get)
    final = mstr
    print final
    for i in range(len(final)):
        try:
            val = (final[i]).split(',')
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, LC_l=float(val[0]), LC_b=float(val[0]), devset=devtoks, devencode=devencode, tau_l=float(val[1]), tau_b=float(val[2]), l_penalty='l2p', b_penalty='l1')
            np.savetxt('log'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[2], fmt='%f')
            np.savetxt('tracc'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[1], fmt='%f')
            np.savetxt('devacc'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[3], fmt='%f')
            sio.mmwrite('wtbn'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.mtx', cl[0].weight_bn())
            sio.mmwrite('wtbv'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.mtx', cl[0].weight_bv())
            np.savetxt('wtln'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[0].weight_ln(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[0].weight_lv(), fmt='%f')
        except:
            pass


if inp == 'l1nn':
    fmins = {}
    for cls in tau_vals_a:
        for cb in tau_vals_a:
            for lc in lcvals:
                main_string = str(lc)+','+str(cls)+','+str(cb)
                try:
                    print 'calling function type ', inp, ' with cb and cls = ', cb, cls, ' and lc = ', lc
                    cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=1005, LC_l=lc, LC_b=lc, devset=devtoks, devencode=devencode, tau_l=cls, tau_b=cb, l_penalty='l1', b_penalty='nn')
                    np.savetxt('combotracc25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'lc'+str(lc)+'.txt', cl[1], fmt='%f')
                    np.savetxt('comboobj25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'lc'+str(lc)+'.txt', cl[2], fmt='%f')
                    np.savetxt('combodevacc25'+inp+str(samples)+'cls'+str(cls)+'cb'+str(cb)+'lc'+str(lc)+'.txt', cl[3], fmt='%f')
                    fmins[main_string] = minlog(cl[2])
                except:
                     print 'problem with eta = ', e, ' cls = ', cls, ' cb ', cb

    mstr = sorted(fmins, key=fmins.get)
    final = mstr
    print final
    for i in range(len(final)):
        try:
            val = (final[i]).split(',')
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, LC_l=float(val[0]), LC_b=float(val[0]), devset=devtoks, devencode=devencode, tau_l=float(val[1]), tau_b=float(val[2]), l_penalty='l1', b_penalty='nn')
            np.savetxt('log'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[2], fmt='%f')
            np.savetxt('tracc'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[1], fmt='%f')
            np.savetxt('devacc'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[3], fmt='%f')
            sio.mmwrite('wtbn'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.mtx', cl[0].weight_bn())
            sio.mmwrite('wtbv'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.mtx', cl[0].weight_bv())
            np.savetxt('wtln'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[0].weight_ln(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'cl'+val[1]+'cb'+val[2]+'lc'+val[0]+'.txt', cl[0].weight_lv(), fmt='%f')
        except:
            pass

