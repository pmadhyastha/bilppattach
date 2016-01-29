
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
samples = [l.strip().split() for l in open('datasets/cleantrain.txt')]
sam = [(lambda x: if x[3] is 'for')(x) for x in samples]
sam = [lambda x:x if x[3] is 'for' for x in samples]
def f(w, pp):
    if w[3] is pp:
        return list(w[i] for i in [1,2,4])
    
sam = [f(x,'for') for x in samples]
f(['39973', 'setting', 'stage', 'for', 'progress', 'v'], 'for')
import billogistic as bil
samples, Vdict, Ndict, Mdict, Y = bil.extdata()
sio.mmread('datasets/trainhw2v.mtx')
bil.main(samples, Vdict, Ndict, Mdict, Y, 2, 0.001, 0.00001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.001, 0.00001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 12, 0.001, 0.00001)
samples[0]
Y[0]
Y[1]
Y
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 12, 0.001, 0.00001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 12, 0.001, 0.00001)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 12, 0.001, 0.00001)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 12, 0.001, 0.00001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 12, 0.00001, 0.00001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 12, 0.00001, 0.01)
bil.main(samples, Vdict, Ndict, Mdict, Y, 12, 1, 0.01)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 1, 10)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 1, 1)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 1, 0.1)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.1, 0.01)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.01, 0.01)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.001, 0.01)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.01, 10)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.001, 10)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.001, 1)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.001, 0.1)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.001, 0.01)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.001, 0.001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.0001, 0.001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.1, 0.001)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.1, 0.001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.01, 0.000001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.01, 0.01)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.1, 0.1)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.1, 0.001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.1, 0.0000000001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.1, 0.000000000)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.1, 0.000000000)
samples, Vdict, Ndict, Mdict, Y = bil.extdata("on")
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.1, 0.000000000)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.1, 0.1)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.01, 0.1)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.001, 0.1)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.0001, 0.1)
bil.main(samples, Vdict, Ndict, Mdict, Y, 100, 0.0001, 0.0001)
get_ipython().magic(u'ls ')
import logreg
