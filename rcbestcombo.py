#!/usr/bin/python

import scipy.io as sio
import numpy as np
import combo_me as co
#import maxent_new as maxent
import bilinear_me as bilme
import sys

samples = int(sys.argv[1])
ppt = str(sys.argv[2])
filbn = str(sys.argv[3])
filbv = str(sys.argv[4])
filln = str(sys.argv[5])
fillv = str(sys.argv[6])
print 'pptype = ', ppt

#if float(sys.argv[5]):
#    eta2 = float(sys.argv[5])
#    eta = np.random.uniform(float(sys.argv[3]), float(sys.argv[5]), 10)
#else:
#    eta = eta1
filbn = np.matrix(sio.mmread(filbn))
filbv = np.matrix(sio.mmread(filbv))
filln = np.array(np.loadtxt(filln, dtype=float))
fillv = np.array(np.loadtxt(fillv, dtype=float))



def getdicts(data):

    noun1 = []
    verb1 = []
    mod1 = []
    nounverb = []
    verbmod = []
    nounmod = []
    nvm = []

    for (tok, label) in data:
        noun1.append(tok[1])
        verb1.append(tok[0])
        mod1.append(tok[3])
        nounverb.append((tok[0],tok[1]))
        verbmod.append((tok[0],tok[3]))
        nounmod.append((tok[1],tok[3]))
        nvm.append((tok[0], tok[1], tok[3]))

    return noun1, verb1, mod1, nounverb, nounmod, verbmod, nvm

def accuracy(encoding_l, encoding_b, gold, filbn, filbv, filln, fillv, rank=None, regtype=None):

    r,c = encoding.shape()
    score = []
    equal_eg = []
    bilaction = []
    total = 0

    if rank:
        un, sn, vtn = np.linalg.svd(filbn)
        uv, sv, vtv = np.linalg.svd(filbv)
        iden = sn[rank]
        idev = sv[rank]
        sn = np.maximum(sn, iden)
        sv = np.maximum(sv, iden)

        filbn = np.dot(un, np.dot(np.diag(sn), vtn))
        filbv = np.dot(uv, np.dot(np.diag(sv), vtv))


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



def getdata(samples, ppt):
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

    encoding = co.ComboMaxentFeatEncoding.train(traindata, phih, phim, maph, mapm, pptype=ppt)
    traintoks = encoding.train_toks()
    print "total training examples for the pptype - ppt ", len(traintoks)
    devencode = co.ComboMaxentFeatEncoding.train(devdata, phidh, phidm, mapdh, mapdm, pptype=ppt)
    devtoks = devencode.train_toks()
    print "total development examples for the pptype - ppt ", len(devtoks)

    return encoding, traintoks, devencode, devtoks

encoding, traintoks, devencode, devtoks = getdata(samples, ppt)

acc, zpfle = accuracy(encoding_l=encoding, encoding_b=devencode, gold=devtoks, filbn=filbn, filbv=filbv, filln=filln, fillv=fillv)
print acc

def extractoov(traintoks, devtoks, encoding, filbn, filbv, filln, fillv, ppt):
    phidh = sio.mmread('clean/devh1k.mtx')
    phidm = sio.mmread('clean/devm1k.mtx')
    mapdh = np.loadtxt('clean/devheads.txt', dtype=str)
    mapdm = np.loadtxt('clean/devmods.txt', dtype=str)


    noun1, verb1, mod1, nounverb, nounmod, verbmod, nvm1 = getdicts(traintoks)
    dnoun1, dverb1, dmod1, dnounverb, dnounmod, dverbmod, dnvm = getdicts(devtoks)

    nlist = []
    vlist = []
    mlist = []
    nvlist = []
    vmlist = []
    nmlist = []
    nvmlist = []

    for item in list(set(dnoun1).difference(set(noun1))):
        for (tok, label) in devtoks:
            if tok[1] == item:
                nlist.append((tok, label))
    del item, tok, label
    for item in list(set(dverb1).difference(set(verb1))):
        for (tok, label) in devtoks:
            if tok[0] == item:
                vlist.append((tok, label))
    del item, tok, label
    for item in list(set(dmod1).difference(set(mod1))):
        for (tok, label) in devtoks:
            if tok[3] == item:
                mlist.append((tok, label))
    del item, tok, label
    for item in list(set(dnounverb).difference(set(nounverb))):
        for (tok, label) in devtoks:
            if tok[0] == item[0] and tok[1] == item[1]:
                nvlist.append((tok, label))
    del item, tok, label
    for item in list(set(dnounmod).difference(set(nounmod))):
        for (tok, label) in devtoks:
            if tok[1] == item[0] and tok[3] == item[1]:
                nmlist.append((tok, label))
    del item, tok, label
    for item in list(set(dverbmod).difference(set(verbmod))):
        for (tok, label) in devtoks:
            if tok[0] == item[0] and tok[3] == item[1]:
                vmlist.append((tok, label))
    del item, tok, label
    for item in list(set(dnvm).difference(set(nvm1))):
        for (tok, label) in devtoks:
            if tok[0] == item[0] and tok[1] == item[1] and tok[3] == item[2]:
                nvmlist.append((tok, label))
    del item, tok, label

    dtlist = [nlist, mlist, vlist, nmlist, vmlist, nvmlist]
    vocablist = ['nlist', 'mlist', 'vlist', 'nmlist', 'vmlist', 'nvmlist']
    m = 0

    for lset in dtlist:
        devencode = co.ComboMaxentFeatEncoding.train(list(lset), phidh, phidm, mapdh, mapdm, pptype=ppt)
        devtoks = devencode.train_toks()

        oacc, zpfle = accuracy(encoding_l=encoding, encoding_b=devencode, gold=devtoks, filbn=filbn, filbv=filbv, filln=filln, fillv=fillv)
        print 'accuracy score for ', vocablist[m], oacc
        m += 1

extractoov(traintoks, devtoks, encoding, filbn, filbv, filln, fillv, ppt)

