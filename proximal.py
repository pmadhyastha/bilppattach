#!/usr/bin/python

import numpy as np
import hashlib

def prox_l22(matrix, nu):
    return ((1./(1.+nu)) * u)

def _load_Lipschitz_constant(W):
    try:
        LC = np.load('./.%s.npy' % sha1(K).hexdigest())
    except:
        LC = 1/norm(np.dot(K, K.transpose()), 2)
        np.save('./.%s.npy' % sha1(K).hexdigest(), mu)
    return LC

