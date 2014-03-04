from __future__ import print_function, unicode_literals, division
from collections import defaultdict as dd
import glob
import os
import re
import shutil
import numpy as np
import sys

dicttaub = dd(list)
dicttauit = dd(list)
dicttauob = dd(list)

dictlcb = dd(list)
dictlcit = dd(list)
dictlcob = dd(list)

direc = sys.argv[1]
wtype = str(sys.argv[2])
inptype = str(sys.argv[3])
os.chdir(direc)
#directory = direc+'comdevacc*'

print ('-------------------------------------------------------------------------------------------')
print (' Prep Reg-Type Samples   Tau      LC     Data    Fix          B-acc    B-Obj    B-CNorm   B-LNorm B-BNorm  B-Iter       Fin-acc   Fin-Obj  Fin-CNorm Fin-LNorm Fin-BNorm Total-Runs')
print ('-------------------------------------------------------------------------------------------')

for files in glob.glob("comdevacc*20801*019*with*"):
    try: 
        base = re.findall("comdevacc|l2pl1|l2pl2p|l2pnn|\d{3,5}|cl[e0-9\-\.]+|cb[e0-9-\.]+|lc[e0-9\.-]+|with", files)
        sample = int(base[2])
        ppt = str(base[6])
        regtype = str(base[1])
        cl = str(re.findall(r'[e0-9-\.]+', base[3])[0])
        cb = str(re.findall(r'[e0-9-\.]+',base[4])[0])
        lc = str(re.findall(r'[e0-9-\.]+',base[5])[0])
        scores = np.loadtxt(files)
        iteration = scores.argmax() + 1
        objective = np.loadtxt('comlog'+base[1]+base[2]+base[3]+base[4]+base[5]+base[6]+'.txt')
        cnorm = np.loadtxt('combonorm'+base[1]+base[2]+base[3]+base[4]+'eta'+lc+base[6]+'.txt')
        bnorm = np.loadtxt('bilnorm'+base[1]+base[2]+base[3]+base[4]+'eta'+lc+base[6]+'.txt')
        lnorm = np.loadtxt('linnorm'+base[1]+base[2]+base[3]+base[4]+'eta'+lc+base[6]+'.txt')
        best = scores.max()

        try:
            final_val = scores[-1]
            lastobj = objective[-1]
            lastcnorm = cnorm[-1]
            lastlnorm = lnorm[-1]
            lastbnorm = bnorm[-1]
            bestobj = objective[iteration]
            bestcnorm = cnorm[iteration]
            bestlnorm = lnorm[iteration]
            bestbnorm = bnorm[iteration]
            total = len(objective) 

        except:

            if len(scores):
                final_val = scores[-1]
            else:
                final_val = scores
            if len(objective):
                lastobj = objective[-1]
            else:
                lastobj = objective
            if len(cnorm):
                lastcnorm = cnorm[-1]
            else:
                lastcnorm = cnorm
            if len(bnorm):
                lastbnorm = bnorm[-1]
            else:
                lastbnorm = bnorm
            if len(lnorm):
                lastlnorm = lnorm[-1]
            else:
                lastlnorm = lnorm
            if len(objective):
                bestobj = objective[0]
            else:
                bestobj = objective
            if len(cnorm):
                bestcnorm = cnorm[0]
            else:
                bestcnorm = cnorm
            if len(bnorm):
                bestbnorm = bnorm[0]
            else:
                bestbnorm = bnorm
            if len(lnorm):
                bestlnorm = lnorm[0]
            else:
                bestlnorm = lnorm
                total = 1  

        dicttaub[(float(cl),float(cb))].append((float(lc), best))     
        dicttauit[(float(cl),float(cb))].append((iteration, best))     
        dicttauob[(float(cl),float(cb))].append((bestobj-lastobj, best))     


#    print (final_val,)# vbest, bestobj, bestcnorm, bestlnorm, bestbnorm)
        print (' %2.5s %7s %2d %7s %7s %8s Normalized %1s %15.5f  %10.5f  %10.5f %10.5f %11.5f %4d %15.5f %10.5f %11.5f %15.5f %10.5f %4d' %(ppt, regtype, sample, cl, cb, lc, wtype, best, bestobj, bestcnorm, bestlnorm, bestbnorm, iteration, final_val, lastobj, lastcnorm, lastbnorm, lastlnorm, total))
    except:
        continue
#print (dicttauob)
def printing(inptype):
    if inptype == 'dicttaub':
        d = dicttaub
    elif inptype == 'dicttauit':
        d = dicttauit
    elif inptype == 'dicttauob':
        d = dicttauob
    elif inptype == 'dictlcb':
        d = dictlcb
    elif inptype == 'dictlcit':
        d = dictlcit
    elif inptype == 'dictlcob':
        d = dictlcob

    for t in d.keys():
        print ("\\addplot")
        print ("    coordinates{")
        print ("    ", ''.join(str(it) for it in d[t]))
        print ("    };")
        print ("   \\addlegendentry{tau=",t,"}")

#printing(inptype)
