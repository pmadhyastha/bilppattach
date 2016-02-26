
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
traindata = [(d.strip().split()[1:5], d.strip.split()[5]) for d in open('datasets/cleantrain.txt') if d.strip().split()[3] == 'into']
traindata = [(d.strip().split()[1:5], d.strip().split()[5]) for d in open('datasets/cleantrain.txt') if d.strip().split()[3] == 'into']
traindata[1]
traindata[1][1]
traindata[1]
traindata[1][0]
traindata[1][0][3,4]
traindata[1][0]
traindata[1]
traindata[1][0]
traindata[1][0][1]
trndt = [list(t[0][0],t[0][1],t[0][3]) for t in traindata]
trndt = [list(t[0][i] for i in [0,1,3]) for t in traindata]
trndt[0]
traindata[0]
get_ipython().system('top')
