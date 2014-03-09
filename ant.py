import scipy.io as sio
import numpy as np
import fix_combo_me as co
import maxent
import testmaxentfn
import sys

samples = 20801
ppt = 'with'

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

trencode = co.ComboMaxentFeatEncoding.train(traindata, phih, phim, maph, mapm, pptype=ppt)
traintoks = trencode.train_toks()
traintokens = [(co.word_features(t),l) for t,l in trencode.train_toks()]
devencode = co.ComboMaxentFeatEncoding.train(devdata, phidh, phidm, mapdh, mapdm, pptype=ppt)
devtoks = devencode.train_toks()
devtokens = [(co.word_features(t),l) for t,l in devencode.train_toks()]

encoding = maxent.BinaryMaxentFeatureEncoding.train(traintokens, labels=None)
inweights = np.zeros(encoding.length())

classify = testmaxentfn.Classification(traintokens=traintokens, develtokens=devtokens, encoding=encoding, weights=inweights, tau=0.01, itr=50, tp=0)
