#!/usr/bin/python
from __future__ import division
import scipy.io as sio
import numpy as np
import combo_me as co
import shutil

def accuracy(encoding_l, encoding_b, gold, filbn, filbv, filln, fillv):
    r,c = encoding.shape()
    score = []
    equal_eg = []
    bilaction = []
    total = 0
    for (tok, label) in gold:

        mark = 0
        total += 1
        noun = 0
        verb = 0
        equal = 0
        v, n, m = encoding_b.bil_u_encode(tok)
        featureset = encoding_l.ext_featstruct(tok)
        fvec_n = encoding_l.lin_encode(featureset, 'n')
        fvec_v = encoding_l.lin_encode(featureset, 'v')

        for (f_id, f_val) in fvec_n:
            noun += filln[f_id] * f_val
        for (f_id, fval) in fvec_v:
            verb += fillv[f_id] * f_val

   #     print noun, verb
        if noun == 0 and verb == 0:
            equal_eg.append((tok, label))
            mark = 1

        noun += np.dot(n, np.dot(filbn, m.transpose()))[0,0]
        verb += np.dot(v, np.dot(filbv, m.transpose()))[0,0]
        if mark != 0:
            if np.exp(noun) > np.exp(verb) and label == 'n':
                bilaction.append('True')
            elif np.exp(noun) < np.exp(verb) and label == 'v':
                bilaction.append('False')
            else:
                bilaction.append('EQUAL')

        if np.exp(noun) > np.exp(verb) and label == 'n':
            score.append(1)
        elif np.exp(verb) > np.exp(noun) and label == 'v':
            score.append(1)
        elif np.exp(noun) == np.exp(verb):
            equal += 1

    print 'number of equal scores = ', equal

    return float(np.sum(score)) / total, zip(equal_eg, bilaction)



def getdata(samples):
    traindata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in open('clean/cleantrain.txt')]
    devdata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in open('clean/cleandev.txt')]
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

    return encoding, traintoks, devencode, devtoks

inp = 'None'
samples = [1000]
#samples = [500]
tau_vals = [1, 0.1, 0.01, 0.001, 0.0001]
#tau_vals_a = [1,0.1,0.001,0.0001,0.00001]
#tau_vals_b = [1000, 10000, 100000, 1000000, 10000000]
devacc = {}
e = 9.84
t = t = 1
for s in samples:
    for cb in tau_vals:
        cls = 1
        print s, cb
        fname = 'res'+inp+str(s)+'cl'+str(cls)+'cb'+str(cb)+'eta'+str(e)
        wn = 'wtbn'+inp+str(s)+'cl'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.mtx'
        wv = 'wtbv'+inp+str(s)+'cl'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.mtx'
        ln = 'wtln'+inp+str(s)+'cl'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.txt'
        lv = 'wtlv'+inp+str(s)+'cl'+str(cls)+'cb'+str(cb)+'eta'+str(e)+'.txt'
#        try:
        weightbn = np.array(sio.mmread(wn))
        weightbv = np.array(sio.mmread(wv))
        weightln = np.array(np.loadtxt(ln, dtype=float))
        weightlv = np.array(np.loadtxt(lv, dtype=float))
        encoding, traintoks, devencode, devtoks = getdata(s)
        acc, listbil = accuracy(encoding_l=encoding, encoding_b=devencode, gold=devtoks, filbn=weightbn, filbv=weightbv, filln=weightln, fillv=weightlv)
        devacc[fname] = acc
        print listbil, acc
        np.savetxt(fname+'.txt', listbil, fmt='%s')
        #shutil.move(wn, 'done')
        #shutil.move(wv, 'done')
        #shutil.move(ln, 'done')
        #shutil.move(lv, 'done')
#        except:
        print fname, 'not found'
        devacc[fname] = 0

np.savetxt('devacc.txt', devacc.items(), fmt='%s')









