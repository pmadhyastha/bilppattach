from __future__ import print_function, unicode_literals, division
from collections import defaultdict as dd
import glob
import os
import re
import shutil
import numpy as np
import sys

objdict = {}
bestdict = []
finaldict = [] 

bestobj = []
finalobj = []

direc = sys.argv[1]
inp = sys.argv[2]
os.chdir(direc)
for files in glob.glob("comdevaccl2pnn20801cl1e-05cb0.0001lc*with.txt"):
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
    tracc = np.loadtxt('comtracc'+base[1]+base[2]+base[3]+base[4]+base[5]+base[6]+'.txt')

    best = scores.max()
    iteration = scores.argmax()
    
    objcordlist = []
    for ind, val in enumerate(objective):
        objcordlist.append((ind+1, val))
    
    objdict[float(lc)] = (objcordlist)
    bestdict.append((float(lc), best))
    bestobj.append((float(lc), objective[iteration]))
    finaldict.append((float(lc), scores[-1]))
    finalobj.append((float(lc), objective[-1]))
    
print (objdict.keys())
def printdict(inp):
    if inp == 'obj':
        d = objdict
        for t in d.keys():
            print ("\\addplot")
            print ("    coordinates{")
            print ("    ", ''.join(str(it) for it in d[t]))
            print ("    };")
            print ("   \\addlegendentry{tau=",t,"}")

    elif inp == 'best':
        d = bestdict
        print ("\\addplot")
        print ("    coordinates{")
        print ("    ", ''.join(str(it) for it in d))
        print ("    };")
        print ("   \\addlegendentry{Best Devel Values}")

    elif inp == 'final':
        d = finaldict 
        print ("\\addplot")
        print ("    coordinates{")
        print ("    ", ''.join(str(it) for it in d))
        print ("    };")
        print ("   \\addlegendentry{Final Devel Values}")

    elif inp == 'bestobj':
        d = bestobj
        print ("\\addplot")
        print ("    coordinates{")
        print ("    ", ''.join(str(it) for it in d))
        print ("    };")
        print ("   \\addlegendentry{Best Objective}")

    elif inp == 'finalobj':
        d = finalobj 
        print ("\\addplot")
        print ("    coordinates{")
        print ("    ", ''.join(str(it) for it in d))
        print ("    };")
        print ("   \\addlegendentry{Final Objective}")


def printtop(): 
    print ('\\begin{tikzpicture}')
    print ('\\begin{axis}[')
    print ('    title={Linear Maxent Models},')
    print ('    height=\textwidth,')
    print ('    width=\textwidth,')
    print ('    xlabel={},')
    print ('    ylabel={},')
    print ('    legend style={at={(1.5, -0.5)}, anchor=west},')
    print ('    ymajorgrids=true,')
    print ('    xmajorgrids=true,')
    print ('    grid style=dashed,]')

def printbottom():
    print ('\\end{axis}')
    print ('\\end{tikzpicture}')


printdict(inp)
