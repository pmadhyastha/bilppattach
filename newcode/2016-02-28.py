
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
301*301
x = np.random.rand(201, 201*201) 
x
x.shape
x = np.random.rand(201, 301*301)
x
quit()

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
x = np.random.rand(301, 301*301) 
np.linalg.svd(x) 
qui
quit()

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
x = np.random.rand(301, 301*301)
y = np.random.rand(301) 
z = np.random.rand(301*301) 
y.dot(x.dot(z.T)) 
get_ipython().magic(u'timeit (y.dot(x.dot(z.T)))')
y.T.dot(z) 
y.dot(z)
np.outer(y.T, z) 
get_ipython().magic(u'timeit (np.outer(y.T, z))')
get_ipython().magic(u'timeit (np.outer(y.T, z))')
quit()

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
help(np.linalg.svd) 
import scipy
help(scipy.linalg.svd) 
help(scipy.linalg.svd)
quit()

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
from bmli
from bmlib import lsvd
x = np.random.rand(10, 100000) 
lsvd.stdsvd(x) 

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
x = np.random.rand(10, 1000)
lsvd.stdsvd(x)
from bmlib import lsvd
np.linalg.svd(x) 
get_ipython().magic(u'timeit (np.linalg.svd(x))')
get_ipython().magic(u'timeit (lsvd.fastsvd(x))')
get_ipython().magic(u'timeit (lsvd.stdsvd(x))')
x = np.random.rand(301, 301*301)
x.shape
get_ipython().magic(u'timeit (np.linalg.svd(x))')
get_ipython().magic(u'timeit (lsvd.fastsvd(x))')
get_ipython().magic(u'timeit (lsvd.stdsvd(x))')

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
x = np.random.rand(301, 301*301)
from bmlib import lsvd
lsvd.stdsvd(x)

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
from bmlib import lsvd
x = np.random.rand(301, 301*301)
x = np.random.rand(201, 201*201)
lsvd.stdsvd(x)

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
get_ipython().system(u'pip install fbpca')
x = np.random.rand(100, 100) 
import fbpca
import fbpca
help(fbpca.svd)
fbpca.svd(x) 
get_ipython().magic(u'timeit (fbpca.svd(x))')
get_ipython().magic(u'timeit (np.linalg.svd(x) )')
get_ipython().magic(u'timeit (np.linalg.svd(x) )')
x = np.random.rand(201, 201*201) 
x
x.shape
fbpca.svd(x)

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
get_ipython().system(u'pip install pycula')
get_ipython().system(u'easy_install pycula ')
import fbpca
import fbpca
x = np.random.rand(201, 201*201)
x = np.random.rand(201, 201)
fbpca.svd(x) 
get_ipython().magic(u'timeit (fbpca.svd(x))')
from sklearn.utils.extmath import randomized_svd
get_ipython().magic(u'timeit (randomized_svd.svd(x))')
get_ipython().magic(u'timeit (randomized_svd(x))')
get_ipython().magic(u'timeit (randomized_svd(x, x.shape[0]))')
get_ipython().magic(u'timeit (fbpca.svd(x))')
x = np.random.rand(101, 101*101) 
get_ipython().magic(u'timeit (fbpca.svd(x))')
get_ipython().magic(u'timeit (randomized_svd(x, x.shape[0] ))')
x = np.random.rand(101, 201*201)
get_ipython().magic(u'timeit (randomized_svd(x, x.shape[0] ))')
randomized_svd(x, x.shape[0] ) 
x = np.random.rand(201, 201*201)
randomized_svd(x, x.shape[0] )
U, S, V = randomized_svd(x, x.shape[0] )
U.shape
S.shape
V.shape
x = np.random.rand(301, 301*301)
U, S, V = randomized_svd(x, x.shape[0] )
import scipy 
import fbpca
z = U.dot(scipy.linalg.diagsvd(S, V.shape[0], V.shape[1]).dot(V.T))
help(fbpca.diffsnorm)
fbpca.diffsnorm(x, U, S, V) 
print("%.5f" % fbpca.diffsnorm(x, U, S, V))
print("%.11f" % fbpca.diffsnorm(x, U, S, V))
scipy.linalg.diagsvd(S, VT.shape) 
scipy.linalg.diagsvd(S, V.shape)
V.shape
exit()

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
x = np.random.rand(301, 301*301)
from sklearn.utils.extmath import randomized_svd
randomized_svd(x, x.shape[1]) 
randomized_svd(x, x.shape[0])
U, S, V = randomized_svd(x, x.shape[0] )
z = U.dot(scipy.linalg.diagsvd(S, V.shape[0], V.shape[1]).dot(V.T))
import scipy
z = U.dot(scipy.linalg.diagsvd(S, V.shape[0], V.shape[1]).dot(V.T))
z
z.shape
x
x.shape
x = np.random.rand(101, 101*101)
U, S, V = randomized_svd(x, x.shape[0] )
U, S, V = randomized_svd(x, x.shape[1] )

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
from sklearn.utils.extmath import randomized_svd
import scipy
x = np.random.rand(101, 101*101)
U, S, V = randomized_svd(x, x.shape[1], transpose=True )

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
from extmath import randomized_svd 
from sklearn.utils import random_state
from sklearn.utils.extmath import chech_random_state
from sklearn.utils.extmath import ccheck_random_state
from sklearn.utils.extmath import check_random_state
from sklearn.utils.extmath.fixes import np_version
from sklearn.utils.fixes import np_version
from sklearn.utils._logistic_sigmoid import np_version
from sklearn.utils._logistic_sigmoid import _log_logistic_sigmoid
import extmath
import extmath
x = np.random.rand(101, 101*101)
extmath.randomized_svd(x, x.shape[0]) 
U, S, V = extmath.randomized_svd(x, x.shape[0])
V.shape
U, S, V = extmath.randomized_svd(x, x.shape[0], transpose=False)
U.shape 
V.shape
import fbpca
u, s, v = fbpca.svd(x) 
v.shape
x = np.random.rand(201, 201*201)
u, s, v = fbpca.svd(x)

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
x = np.random.rand(201, 201*201)
import extmath
extmath.randomized_svd(x, x.shape[0]
)
import fbpca
x = np.random.rand(201, 201*201)
u, s, v = fbpca.svd(x)
v
v.shape
u.shape
s.shape
U, S, V = fbpca.svd(x)

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
import fbpca
x = np.random.rand(301, 301*301)
U, S, V = fbpca.svd(x)
x.shape
from bmlib
from bmlib import lsvd
U, S, V = lsvd.stdsvd(x)
quit()

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
import fbpca
x = np.random.rand(301, 301*301)
U, S, V = fbpca.svd(x)
exit()

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
from glob import glob
import bilpreplogistic as bpl
bil = np.load('bil_all_0.000000001_0.001_500/model5000.0000000010.001all.npy')
bpl.test('all', bil) 
bpl.test('all', bil, wetyp='skipdep' )
bil = np.load('bil_all_0.00000001_0.00000001_500_w2v50/model5000.000000010.00000001all.npy')
bpl.test('all', bil, wetyp='skipdep' )
bpl.test('all', bil, wetyp='w2v50' )
bil = np.load('bil_all_0.0000001_0.00001_500/model5000.00000010.00001all.npy')
bpl.test('all', bil, wetyp='skipdep' )
f = open('bilresults_sofar.txt', 'w') 
for dir in glob('bil_*500'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    f.write('\n%s\n' % dir)
    f.write(bpl.test('all', bil, 'skipdep'))
    
f.close() 
for dir in glob('bil_*500'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'skipdep')
    print dir 
    
for dir in glob('bil_*w2v50'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'w2v50')
    print dir
    
for dir in glob('bil_*g6b50'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'w2v50')
    print dir
    
for dir in glob('bil_*gl6b50'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'gl6b50')
    print dir
    
for dir in glob('bil_*gl6b100'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'gl6b100')
    print dir
    
for dir in glob('bil_*w2v100'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'w2v100')
    print dir
    
for dir in glob('bil_*500'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'w2v100')
    print dir
    
for dir in glob('bil_*500'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'skipdep')
    print dir
    
bil = np.load('bil_all_0.0000001_0.00001_500/model5000.00000010.00001all.npy')
bpl.test('all', bil, 'skipdep') 
bpl.test('of', bil, 'skipdep')
bpl.test('in', bil, 'skipdep')
bpl.test('to', bil, 'skipdep')
bpl.test('to', for, 'skipdep')
bpl.test('for', bil, 'skipdep')
bpl.test('on', bil, 'skipdep')
bpl.test('from', bil, 'skipdep')
bpl.test('with', bil, 'skipdep')
bpl.test('R', bil, 'skipdep')
bpl.test('AT', bil, 'skipdep')
bpl.test('at', bil, 'skipdep')
bpl.test('as', bil, 'skipdep')
bpl.test('by', bil, 'skipdep')
bpl.test('in', bil, 'skipdep')
bpl.test('to', bil, 'skipdep')
get_ipython().magic(u'ls ')
quit()

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
import bilpreplogistic as bpl
for dir in glob('bil_*500'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'skipdep')
    print dir
    
from glob import glob
for dir in glob('bil_*500'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'skipdep')
    print dir
    
for dir in glob('bil_*50'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'skipdep')
    print dir
    
for dir in glob('bil_*50*'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'skipdep')
    print dir
    
for dir in glob('bil_*50'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'skipdep')
    print dir
    
for dir in glob('bil_*_50'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'skipdep')
    print dir
    
for dir in glob('bil_*_500'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'skipdep')
    print dir
    
for dir in glob('bil_*_w2v50'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'w2v50')
    print dir
    
for dir in glob('bil_*_w2v100'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil, 'w2v100')
    print dir
    
