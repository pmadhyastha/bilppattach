
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
get_ipython().magic('ls datasets/')
get_ipython().magic('ls ')
samples = [l.strip().split() for l in open('datasets/cleantrain.txt')] 
samples
sam = [(lambda x: if x[3] is 'for')(x) for x in samples]
sam = [(lambda x: x[3] is 'for')(x) for x in samples]
sam
sam = [(lambda x:x if x[3] is 'for')(x) for x in samples]
sam = [(lambda x:x if x[3] == 'for')(x) for x in samples]
sam = [(lambda x:x if x[3] is 'for' else pass)(x) for x in samples]
sam = [lambda x:x if x[3] is 'for' for x in samples]
fpp = lambda x: x[3] is 2 and x
fpp[samples[1]]
fpp = lambda x: x if x[3] is 'for'
fpp = lambda x: x if x[3] is 'for' else pass
fpp = lambda x: x if x[3]  ==  'for' else pass
lambda x: True if x % 2 == 0 else False
f  = lambda x: True if x % 2 == 0 else False
def f(w):
    if w[3] is 'for':
        return w[1:4]
    
samples
f(['1877', 'leading', 'bid', 'for', 'corp', 'n'],)
def f(w):
    if w[3] is 'for':
        return w[1:3]+w[4]
    
f(['1877', 'leading', 'bid', 'for', 'corp', 'n'],)
def f(w):
    if w[3] is 'for':
        return w[1:3,4]
    
f(['1877', 'leading', 'bid', 'for', 'corp', 'n'],)
def f(w):
    if w[3] is 'for':
        return w[1:3].append(w[4])
    
f(['1877', 'leading', 'bid', 'for', 'corp', 'n'],)
def f(w):
    if w[3] is 'for':
        return w[1:5]
    
f(['1877', 'leading', 'bid', 'for', 'corp', 'n'],)
def f(w):
    if w[3] is 'for':
        return w[1,2,4]
    
f(['1877', 'leading', 'bid', 'for', 'corp', 'n'],)
def f(w):
    if w[3] is 'for':
        return w[1,2,4]
    
def f(w):
    if w[3] is 'for':
        return list(w[i] for i in [1,2,4])
    
f(['1877', 'leading', 'bid', 'for', 'corp', 'n'],)
def f(w, pp):
    if w[3] is pp:
        return list(w[i] for i in [1,2,4])
    
f(['1877', 'leading', 'bid', 'for', 'corp', 'n'],'for')
f(['1877', 'leading', 'bid', 'for', 'corp', 'n'],'in')
sam = [f(x,'for') for x in samples]
sam
print f(x,'for') for x in samples
for s in sam:
    if s:
        print s
        
for s in sam:
    if s:
        print(s)
        
sam = []
for s in samples:
    sam.append(f(x,'for'))
    
for s in samples:
    sam.append(f(s,'for'))
    
sam
for s in samples:
    f(s,'for')
    
for s in samples:
    print f(s,'for')
    
for s in samples:
    print(f(s,'for')
    )
    
for s in samples:
    print(f(s,'for')
    )
    
for s in samples:
    print s
    
for s in samples:
    print(s)
    
f(['39973', 'setting', 'stage', 'for', 'progress', 'v'], 'for')
get_ipython().magic('ls ')
sam = []
for s in samples:
    if s[3] == 'for':
        sam.append(list(s[i] for i in [1,2,4]))
        
sam
get_ipython().magic('ls ')
get_ipython().magic('ls datasets/')
get_ipython().magic('ls ')
import billogistic as bil
quit()

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
import billogistic as bil
samples, Vdict, Ndict, Mdict, Y = bil.extdata()
samples
reload(bil)
samples, Vdict, Ndict, Mdict, Y = bil.extdata()
reload(bil)
samples, Vdict, Ndict, Mdict, Y = bil.extdata()
samples
reload(bil)
samples
samples, Vdict, Ndict, Mdict, Y = bil.extdata()
samples
reload(bil)
samples, Vdict, Ndict, Mdict, Y = bil.extdata()
samples
reload(bil)
samples
samples, Vdict, Ndict, Mdict, Y = bil.extdata()
get_ipython().system(u'less datasets/forhead.txt')
samples, Vdict, Ndict, Mdict, Y = bil.extdata()
reload(bil)
samples, Vdict, Ndict, Mdict, Y = bil.extdata()
reload(bil)
reload(bil)
samples, Vdict, Ndict, Mdict, Y = bil.extdata()
reload(bil)
samples, Vdict, Ndict, Mdict, Y = bil.extdata()
reload(bil)
samples, Vdict, Ndict, Mdict, Y = bil.extdata()
get_ipython().magic(u'ls datasets/')
sio.mmread('datasets/trainhw2v.mtx')
reload(bil)
sio.mmread('datasets/trainhw2v.mtx')
reload(bil)
samples, Vdict, Ndict, Mdict, Y = bil.extdata()
bil.main(samples, Vdict, Ndict, Mdict, Y, 2, 0.001, 0.00001) 
samples, Vdict, Ndict, Mdict, Y = bil.extdata()
bil.main(samples, Vdict, Ndict, Mdict, Y, 2, 0.001, 0.00001)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 2, 0.001, 0.00001)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 2, 0.001, 0.00001)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 2, 0.001, 0.00001)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 2, 0.001, 0.00001)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 2, 0.001, 0.00001)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 2, 0.001, 0.00001)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 2, 0.001, 0.00001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 2, 0.001, 0.00001)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 2, 0.001, 0.00001)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 2, 0.001, 0.00001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.001, 0.01)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.001, 0.01)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.000001, 0.01)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 10, 100)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 10, 0.0000001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 10, 100)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 10, 100)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.01, 0.001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.1, 0.001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.0001, 0.001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.0001, 0.01)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.0001, 1)
x = np.random(10) 
x = np.random.rand(10) 
np.dot(x,x) 
np.dot(x.transpose(),x)
from scipy.optimize import fmin_bfgs
def train_w(samples, Vdict, Ndict, Mdict, Y,C):
    operator = bil.Bilnear(samples, Vdict, Ndict, Mdict, Y)
    def f(w):
        print operator.log_l(w,C)
        return operator.log_l(w,C)
    def fprime(w):
        return operator.log_l_grad(w,C)
    k = len(Vdict.values()[0])
    winit = np.matrix(np.zeros((k,k), dtype=np.float))
    return fmin_bfgs(f, winit, fprime)

reload(bil)
w = train_w(samples, Vdict, Ndict, Mdict, Y, 0.001)
def train_w(samples, Vdict, Ndict, Mdict, Y,C):
    operator = bil.Bilnear(samples, Vdict, Ndict, Mdict, Y)
    operator.preprocess()
    def f(w):
        print operator.log_l(w,C)
        return operator.log_l(w,C)
    def fprime(w):
        return operator.log_l_grad(w,C)
    k = len(Vdict.values()[0])
    winit = np.matrix(np.zeros((k,k), dtype=np.float))
    return fmin_bfgs(f, winit, fprime)

w = train_w(samples, Vdict, Ndict, Mdict, Y, 0.001)
w = train_w(samples, Vdict, Ndict, Mdict, Y, 0.01)
def train_w(samples, Vdict, Ndict, Mdict, Y,C):
    operator = bil.Bilnear(samples, Vdict, Ndict, Mdict, Y)
    operator.preprocess()
    def f(w):
        print operator.log_l(w,C)
        return operator.log_l(w,C)
    def fprime(w):
        return operator.log_l_grad(w,C)
    k = len(Vdict.values()[0])
    winit = np.matrix(np.zeros((k,k), dtype=np.float))
    return fmin_bfgs(f, winit, fprime, disp=False)

w = train_w(samples, Vdict, Ndict, Mdict, Y, 0.01)
def train_w(samples, Vdict, Ndict, Mdict, Y,C):
    operator = bil.Bilnear(samples, Vdict, Ndict, Mdict, Y)
    operator.preprocess()
    def f(w):
        print operator.log_l(w,C)
        return operator.log_l(w,C)
    def fprime(w):
        return operator.log_l_grad(w,C)
    k = len(Vdict.values()[0])
    winit = np.matrix(np.zeros((k,k), dtype=np.float))
    return fmin_bfgs(f, winit, fprime)

w = train_w(samples, Vdict, Ndict, Mdict, Y, 0.01)
def train_w(samples, Vdict, Ndict, Mdict, Y,C):
    operator = bil.Bilnear(samples, Vdict, Ndict, Mdict, Y)
    operator.preprocess()
    def f(w):
        print operator.log_l(w,C)
        return operator.log_l(w,C)
    def fprime(w):
        return operator.log_l_grad(w,C)
    k = len(Vdict.values()[0])
    winit = np.matrix(np.zeros((k,k), dtype=np.float))
    return fmin_bfgs(f, winit, fprime, maxiter=100)

w = train_w(samples, Vdict, Ndict, Mdict, Y, 0.01)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.0001, 1)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.0001, 1)
reload(bil)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.0001, 1)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.0001, 0.1)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.0001, 0.0001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.0001, 0.0000001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.001, 0.1)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.001, 0.001)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.001, 0.001)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.001, 0.01)
bil.main(samples, Vdict, Ndict, Mdict, Y, 10, 0.1, 0.01)
bil.main(samples, Vdict, Ndict, Mdict, Y, 1000, 0.1, 0.01)
bil.main(samples, Vdict, Ndict, Mdict, Y, 1000, 0.001, 0.01)
reload(bil)
bil.main(samples, Vdict, Ndict, Mdict, Y, 1000, 0.001, 0.01)
