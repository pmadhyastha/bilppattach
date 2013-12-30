import glob
import os
import re
import shutil
import numpy as np
import rcbestcombo as rcb
import scipy.io as sio
import combo_me as co
import sys
import pickle
os.chdir("/home/pranava/Documents/phd/python/code/bilppattach/combo-models-good")
destination = ("/home/pranava/Documents/phd/python/code/bilppattach/combo-models-done")
source = ("/home/pranava/Documents/phd/python/code/bilppattach/combo-models-good")

class ngram(dict):
    """Based on perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return super(ngram, self).__getitem__(item)
        except KeyError:
            value = self[item] = type(self)()
            return value

maindict = ngram()
oovdict = {}
accrank1 = {}
accrank2 = {}
for files in glob.glob("bestwtbn*cl1e-12*"):
    base = re.findall(r'l2pl1|l2pnn|l2pl2p|\d{3,5}|cl[e0-9-]+|cb[0-9-\.]+|eta[0-9.]+|on|for|with|to|from|in', files)
    samples = int(base[1])
    ppt = str(base[5])
    regtype = str(base[0])
    cl = str(base[2])
    cb = str(base[3])
    eta = str(base[4])

    bnfile = 'bestwtbn'+''.join(base)+'.mtx'
    bvfile = 'bestwtbv'+''.join(base)+'.mtx'
    lnfile = 'bestwtln'+''.join(base)+'.txt'
    lvfile = 'bestwtlv'+''.join(base)+'.txt'

    filbn = np.matrix(sio.mmread(bnfile))
    filbv = np.matrix(sio.mmread(bvfile))
    filln = np.array(np.loadtxt(lnfile, dtype=float))
    fillv = np.array(np.loadtxt(lvfile, dtype=float))

    encoding, traintoks, devencode, devtoks = rcb.getdata(samples, ppt)
#    print encoding.shape()
    acc = rcb.accuracy(encoding_l=encoding, encoding_b=devencode, gold=devtoks, filbn=filbn, filbv=filbv, filln=filln, fillv=fillv)
    frm = regtype+cl+cb+eta

    print ppt, len(traintoks), cl, cb, eta, 'accuracy =', acc
    maindict[ppt][len(traintoks)][frm] = acc

    if samples == 20801:
        recdict = rcb.extractoov(traintoks, devtoks, encoding, filbn, filbv, filln, fillv, ppt)
        oovdict[ppt+' '+frm] = recdict
        print ppt, recdict

        for r in [10, 200, 400, 800]:
            acc = rcb.accuracy(encoding_l=encoding, encoding_b=devencode, gold=devtoks, filbn=filbn, filbv=filbv, filln=filln, fillv=fillv, rank=r)
            accrank1[ppt+' '+frm+' '+str(r)] = acc
        if regtype == 'l2pl1' or regtype == 'l2pl2p':
            for r in [10, 200, 400, 800]:
                acc = rcb.accuracy(encoding_l=encoding, encoding_b=devencode, gold=devtoks, filbn=filbn, filbv=filbv, filln=filln, fillv=fillv, rank=r)
                accrank2[ppt+' '+frm+' '+str(r)] = acc


    pickle.dump(dict(maindict), open('/home/pranava/Documents/phd/python/code/bilppattach/maindict.txt', "wb"))
    pickle.dump(oovdict, open('/home/pranava/Documents/phd/python/code/bilppattach/oovdict.txt', "wb"))
    pickle.dump(accrank1, open('/home/pranava/Documents/phd/python/code/bilppattach/accrank1dict.txt', "wb"))
    pickle.dump(accrank2, open('/home/pranava/Documents/phd/python/code/bilppattach/accrank2dict.txt', "wb"))

    shutil.move(bnfile, destination)
    shutil.move(bvfile, destination)
    shutil.move(lnfile, destination,)
    shutil.move(lvfile, destination,)

#np.savetxt('l1nnfinalall.txt', devacclist.items(), fmt='%s')

