from __future__ import division
from sklearn import preprocessing
from scipy.io import mmread
from time import time
import numpy as np

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


class Combo(object):

    def __init__(self, samples, Vdict, Ndict, Mdict, Y, Winit='zeros'):
        self.samples = samples
        self.Vdict = Vdict
        self.Ndict = Ndict
        self.Mdict = Mdict
        self.ll = None
        self.Y = Y
        self.dim = Vdict.values()[0].shape[1]
        self.nsamples = len(self.samples)
        self.Xi = {}
        self.grad = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        if Winit is 'random':
            self.Wmat = np.matrix(np.random.rand(self.dim, self.dim))
        elif Winit is 'zeros':
            self.Wmat = np.matrix(np.zeros((self.dim, self.dim), dtype=np.float))
        elif Winit is 'identity':
            self.Wmat = np.matrix(np.identity(self.dim, dtype=np.float))
        else:
            self.Wmat = Winit
        self.norm = np.linalg.norm(self.Wmat)
        self.lsigma = None
        self.rsigma = None
        self.pcaSigma = None
        self.zcaSigma = None
        self.ryotaSigma = None
        self.Xpi = {}

