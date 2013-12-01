#!/usr/bin/python

import scipy.io as sio
import numpy as np
import combo_me as co
import sys

samples = int(sys.argv[1])
inp = str(sys.argv[2])


traindata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in open('clean/cleantrain.txt')]
devdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in open('clean/cleandev.txt')]
testdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in open('clean/cleantest.txt')]
traindata = traindata[:samples]
phih = sio.mmread('clean/phih.mtx')
phim = sio.mmread('clean/phim.mtx')
phidh = sio.mmread('clean/phidevhead.mtx')
phidm = sio.mmread('clean/phidevmod.mtx')
maph = np.loadtxt('clean/forhead.txt', dtype=str)
mapm = np.loadtxt('clean/formod.txt', dtype=str)
mapdh = np.loadtxt('clean/devheads.txt', dtype=str)
mapdm = np.loadtxt('clean/devmods.txt', dtype=str)

phith = sio.mmread('clean/phitesthead.mtx')
phitm = sio.mmread('clean/phitestmod.mtx')
mapth = np.loadtxt('clean/testheads.txt', dtype=str)
maptm = np.loadtxt('clean/testmods.txt', dtype=str)

encoding = co.ComboMaxentFeatEncoding.train(traindata, phih, phim, maph, mapm, pptype='for')
traintoks = encoding.train_toks()
print 'type = ', inp, 'total samples = ', samples
print "total training examples for the pptype - 'for' ", len(traintoks)
devencode = co.ComboMaxentFeatEncoding.train(devdata, phidh, phidm, mapdh, mapdm, pptype='for')
devtoks = devencode.train_toks()
print "total development examples for the pptype - 'for' ", len(devtoks)

tau_vals_a = [1,0.1,0.001,0.0001,0.00001]
tau_vals_b = [1000, 10000, 100000, 1000000, 10000000]
devacc = []
#eta = [1, 0.1, 0.01]
if inp == 'None':
    for cls in tau_vals_a:
        for cb in tau_vals_a:
            print 'calling function type ', inp, ' with tau_l ', cls, ' and tau_b ', cb
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, tau_l=cls, tau_b=cb)
            np.savetxt('log'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[2], fmt='%f')
            sio.mmwrite('wtbn'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.mtx', cl[0].weight_bn())
            sio.mmwrite('wtbv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.mtx', cl[0].weight_bv())
            np.savetxt('wtln'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[0].weight_ln(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[0].weight_lv(), fmt='%f')

elif inp == 'l2l2':
    for cls in tau_vals_a:
        for cb in tau_vals_a:
            print 'calling function type ', inp, ' with tau_l ', cls, ' and tau_b ', cb
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, tau_l=cls, tau_b=cb, l_penalty='l2', b_penalty='l2')
            np.savetxt('log'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[2], fmt='%f')
            sio.mmwrite('wtbn'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.mtx', cl[0].weight_bn())
            sio.mmwrite('wtbv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.mtx', cl[0].weight_bv())
            np.savetxt('wtln'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[0].weight_ln(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[0].weight_lv(), fmt='%f')


elif inp == 'l2pl2p':
    for cls in tau_vals_a:
        cb = cls
        print 'calling function type ', inp, ' with tau_l ', cls, ' and tau_b ', cb
        cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, tau_l=cls, tau_b=cb, l_penalty='l2p', b_penalty='l2p')
        np.savetxt('log'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[2], fmt='%f')


elif inp == 'l2l1':
    for cls in tau_vals_a:
        for cb in tau_vals_b:
            print 'calling function type ', inp, ' with tau_l ', cls, ' and tau_b ', cb
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, tau_l=cls, tau_b=cb, l_penalty='l2', b_penalty='l1', LC_b=1000000000)
            np.savetxt('log'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[2], fmt='%f')
            sio.mmwrite('wtbn'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.mtx', cl[0].weight_bn())
            sio.mmwrite('wtbv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.mtx', cl[0].weight_bv())
            np.savetxt('wtln'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[0].weight_ln(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[0].weight_lv(), fmt='%f')



elif inp == 'l2nn':
    for cls in tau_vals_a:
        for cb in tau_vals_b:
            print 'calling function type ', inp, ' with tau_l ', cls, ' and tau_b ', cb
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, tau_l=cls, tau_b=cb, l_penalty='l2', b_penalty='nn', LC_b=1000000000)
            np.savetxt('log'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[2], fmt='%f')
            sio.mmwrite('wtbn'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.mtx', cl[0].weight_bn())
            sio.mmwrite('wtbv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.mtx', cl[0].weight_bv())
            np.savetxt('wtln'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[0].weight_ln(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[0].weight_lv(), fmt='%f')



elif inp == 'l1l2':
    for cls in tau_vals_b:
        for cb in tau_vals_a:
            print 'calling function type ', inp, ' with tau_l ', cls, ' and tau_b ', cb
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, tau_l=cls, tau_b=cb, l_penalty='l1', b_penalty='l2', LC_l=1000000000)
            np.savetxt('log'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[2], fmt='%f')
            sio.mmwrite('wtbn'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.mtx', cl[0].weight_bn())
            sio.mmwrite('wtbv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.mtx', cl[0].weight_bv())
            np.savetxt('wtln'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[0].weight_ln(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[0].weight_lv(), fmt='%f')

elif inp == 'l1nn':
    for cls in tau_vals_b:
        for cb in tau_vals_b:
            print 'calling function type ', inp, ' with tau_l ', cls, ' and tau_b ', cb
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, tau_l=cls, tau_b=cb, l_penalty='l1', b_penalty='nn', LC_l=1000000000, LC_b=1000000000)
            np.savetxt('log'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[2], fmt='%f')
            sio.mmwrite('wtbn'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.mtx', cl[0].weight_bn())
            sio.mmwrite('wtbv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.mtx', cl[0].weight_bv())
            np.savetxt('wtln'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[0].weight_ln(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[0].weight_lv(), fmt='%f')

elif inp == 'l1l1':
    for cls in tau_vals_b:
        for cb in tau_vals_b:
            print 'calling function type ', inp, ' with tau_l ', cls, ' and tau_b ', cb
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, tau_l=cls, tau_b=cb, l_penalty='l1', b_penalty='l1', LC_l=1000000000, LC_b=1000000000)
            np.savetxt('log'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[2], fmt='%f')
            sio.mmwrite('wtbn'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.mtx', cl[0].weight_bn())
            sio.mmwrite('wtbv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.mtx', cl[0].weight_bv())
            np.savetxt('wtln'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[0].weight_ln(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'.txt', cl[0].weight_lv(), fmt='%f')

print 'done'
#testencode = co.ComboMaxentFeatEncoding.train(testdata, phith, phitm, mapth, maptm, pptype='for')
#test_toks = testencode.train_toks()
#print co.accuracy(encoding, testencode, cl[0], testdata)

