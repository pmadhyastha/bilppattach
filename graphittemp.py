#!/usr/bin/python

from __future__ import print_function, unicode_literals, division
from collections import defaultdict as dd
import glob
import os
import re
import shutil
import numpy as np
import sys

bestlc = {}
taulcdict = dd(list)
direc = sys.argv[1]
inp = sys.argv[2]
os.chdir(direc)

for files in glob.glob("bdevaccnn20801*with.txt"):
    try:
        base = re.findall("bdevacc|l1|l2p|nn|\d{3,5}|tau[e0-9\-\.]+|lc[e0-9\.-]+|with", files)

        sample = int(base[2])
        ppt = str(base[5])
        regtype = str(base[1])

        tau = str(re.findall(r'[e0-9-\.]+',base[3])[0])
        lc = str(re.findall(r'[e0-9-\.]+',base[4])[0])
        scores = np.loadtxt(files)

        objective = np.loadtxt('bobjective'+base[1]+base[2]+base[3]+base[4]+base[5]+'.txt')
        norm = np.loadtxt('sumnorm'+base[1]+base[2]+base[3]+base[4]+base[5]+'.txt')
        tracc = np.loadtxt('btracc'+base[1]+base[2]+base[3]+base[4]+base[5]+'.txt')

        best = scores.max()
        iteration = scores.argmax()

    #    iteration = scores.argmax() + 1
        indicator = np.sort(objective)[-1]
        if indicator < 1:
            objcordlist = []

            for ind, val in enumerate(objective):
                objcordlist.append((ind+1, val))

            taulcdict[float(tau)].append((float(lc), objcordlist))
    except:
        continue

def printdict(inp):
    if inp == 'taulc':
        sortedtau = np.sort(taulcdict.keys()).tolist()
        for tau in sortedtau:
            printtop(tau)
            lcdict = dict(taulcdict[tau])
            sortedlc = np.sort(lcdict.keys()).tolist()
            bestobj = 1
            for lc in sortedlc:
                print ("\\addplot")
                print ("    coordinates{")
                print ("    ", ''.join(str(it) for it in lcdict[lc]))
                print ("    };")
                print ("   \\addlegendentry{lc=",lc,"}")
                try:
                    tempobj = ((lcdict[lc])[99])[1]
                    if bestobj > tempobj:
                        bestobj = tempobj
                        bestlc[tau] = (lc, bestobj)
                except:
                    continue

            bestset = [tau, bestlc[tau]]

            printbottom(bestset)


def printtop(val):
    print ('\\begin{figure}')
    print ('\\begin{tikzpicture}')
    print ('\\begin{axis}[')
    print ('    title={Tau = ', val, '},')
    print ('    height=\\textwidth,')
    print ('    width=\\textwidth,')
    print ('    xlabel={Iterations},')
    print ('    ylabel={Objective},')
    print ('    legend style={at={(0.5, -0.5)}, anchor=west},')
    print ('    ymajorgrids=true,')
    print ('    xmajorgrids=true,')
    print ('    grid style=dashed,]')

def printbottom(bestset):
    print ('\\end{axis}')
    print ('\\end{tikzpicture}')
    print('\\caption{For Tau = ',bestset[0], 'best LC = ', bestset[1][0], 'that obtains objective at iteration 99 = ', bestset[1][1], '}')
    print ('\\end{figure}')
    print ('')
    print ('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print ('')


printdict(inp)
