#!/usr/bin/python

import scipy.io as sio
import numpy as np
import bilinear_me as co
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
eta = [1, 0.1, 0.01]
if inp == 'None':
    print 'here'
    for cls in tau_vals_a:
        print 'calling function type ', inp,' and tau_b ', cls
        cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, tau_b=cls, b_penalty=None)
        np.savetxt('log'+inp+str(samples)+'cb'+str(cls)+'.txt', cl[2], fmt='%f')
        sio.mmwrite('wtbn'+inp+str(samples)+'cb'+str(cls)+'.mtx', cl[0].weight_bn())
        sio.mmwrite('wtbv'+inp+str(samples)+'cb'+str(cls)+'.mtx', cl[0].weight_bv())

elif inp == 'l2':
    for cls in tau_vals_a:
        print 'calling function type ', inp,' and tau_b ', cls
        cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, tau_b=cls, b_penalty='l2')
        np.savetxt('log'+inp+str(samples)+'cb'+str(cls)+'.txt', cl[2], fmt='%f')
        sio.mmwrite('wtbn'+inp+str(samples)+'cb'+str(cls)+'.mtx', cl[0].weight_bn())
        sio.mmwrite('wtbv'+inp+str(samples)+'cb'+str(cls)+'.mtx', cl[0].weight_bv())

elif inp == 'l1':
    for cls in tau_vals_b:
        print 'calling function type ', inp,' and tau_b ', cls
        cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, tau_b=cls, b_penalty='l1')
        np.savetxt('log'+inp+str(samples)+'cb'+str(cls)+'.txt', cl[2], fmt='%f')
        sio.mmwrite('wtbn'+inp+str(samples)+'cb'+str(cls)+'.mtx', cl[0].weight_bn())
        sio.mmwrite('wtbv'+inp+str(samples)+'cb'+str(cls)+'.mtx', cl[0].weight_bv())

elif inp == 'nn':
    for cls in tau_vals_b:
        print 'calling function type ', inp,' and tau_b ', cls
        cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=100, tau_b=cls, b_penalty='nn')
        np.savetxt('log'+inp+str(samples)+'cb'+str(cls)+'.txt', cl[2], fmt='%f')
        sio.mmwrite('wtbn'+inp+str(samples)+'cb'+str(cls)+'.mtx', cl[0].weight_bn())
        sio.mmwrite('wtbv'+inp+str(samples)+'cb'+str(cls)+'.mtx', cl[0].weight_bv())




print 'done'
#testencode = co.ComboMaxentFeatEncoding.train(testdata, phith, phitm, mapth, maptm, pptype='for')
#test_toks = testencode.train_toks()
#print co.accuracy(encoding, testencode, cl[0], testdata)

