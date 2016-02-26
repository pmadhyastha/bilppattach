
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
from sklearn.utils import array2d, as_float_array 
quit()

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
from sklearn.utils import array2d, as_float_array
from sklearn.linear_model import SGDClassifier
quit()

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
import billogistic as bil 
quit()

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
import billogistic as bil
x = np.random.rand(100, 100) 
bil.PCA(x).fit_transform(x) 
x
bil.XCA(x).fit_transform(x)
bil.ZCA(x).fit_transform(x)
import sklearn.utils as sku
sku.as_float_array(x) 
sku.as_float_array(x).shape
from sklearn.decomposition import PCA
x
PCA.fit_transform(x)
PCA(x,whiten=True, n_components=100).fit_transform(x)
PCA(x,whiten=True).fit_transform(x)
PCA(x,n_components=100, whiten=True).fit_transform(x)
PCA(x,n_components=100, whiten=True).fit_transform()
PCA(x, whiten=True).fit_transform()
PCA(x, whiten=True).fit_transform(x)
PCA(x, whiten=True).fit(x)
PCA(x, whiten=True)
PCA(n_components=100).fit_transform(x)
PCA(n_components=100, whiten=True).fit_transform(x)
y = PCA(n_components=100, whiten=True).fit_transform(x)
x.shape
y.shape
zz = bil.ZCA()
zz.fit_transform(x)
zz.fit_transform(x).shape
pc = bil.PCA()
pc.fit_transform(x) 
x
y
x
y
pc.fit_transform(x)
y = PCA(whiten=True).fit_transform(x)
y.shape
y
np.zeros(19)
x 
x = np.random.rand(10)
np.dot(x, 0.4)
x
x.dtype
x.dtype
x = np.array(10, 10) 
x = np.ones(10, 10)
x = np.random.rand(10, 10)
x[1].shape
x[0].shape
ss.issparse
ss.isspmatrix
