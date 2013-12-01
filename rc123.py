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
print 'type = ', inp, 'total samples = ', samples
print "total training examples for the pptype - 'for' ", len(traintoks)
devencode = co.ComboMaxentFeatEncoding.train(devdata, phidh, phidm, mapdh, mapdm, pptype='for')
devtoks = devencode.train_toks()
print "total development examples for the pptype - 'for' ", len(devtoks)

tau_vals_a = [200]
tau_vals_b = [1000, 10000, 100000, 1000000, 10000000]
devacc = []
#eta = [100000000, 100000, 10000, 1000, 100, 10, 1, 0.1, 0.01,0.0001, 0.00001, 0.000001, 0.0000001, 0.0000000001, 0.00000001, 0.0000000001, 0.0000000000001]
cls = 1
#eta = [1230, 1220, 1210, 1200]
eta = [9.84]
e = 9.84
cb = 0.0001
if inp == 'None':
    for cb in tau_vals_a:
        try:
    #    	    print 'calling function type ', inp, ' with tau_l ', cls, ' and tau_b ', cb, 'also with eta = ', e
            print 'calling function type ', inp, ' with tau_l ', cls, ' and tau_b ', cb, 'also with eta = ', e
            #    	    cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=20, tau_l=cls, tau_b=cb, eta=e/float(len(traintoks)))
            cl = co.ComboMaxent.combo_train(traintoks, encoding, max_iter=15, tau_l=cls, tau_b=cb, eta=e)
            np.savetxt('log'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.txt', cl[2], fmt='%f')
            np.savetxt('tracc'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.txt', cl[1], fmt='%f')
            sio.mmwrite('wtbn'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.mtx', cl[0].weight_bn())
            sio.mmwrite('wtbv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.mtx', cl[0].weight_bv())
            np.savetxt('wtln'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.txt', cl[0].weight_ln(), fmt='%f')
            np.savetxt('wtlv'+inp+str(samples)+'cl'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.txt', cl[0].weight_lv(), fmt='%f')
        except:
            print 'problem with eta = ', e

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

