import numpy as np

def decreasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def increasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

val, idx = min((val, idx) for (idx, val) in enumerate(loglist))

fmins = {}
trmon = {}
logmon = {}

fmins[eta] = val
trmon[eta] = increasing(trlist[:5])
logmon[eta] = decreasing(loglist[:5])

sleta = sorted(fmins, key=fmins.get)
fleta = []

for eta in sleta:
    if trmon[eta] == True and logmon[eta] == True:
        fleta.append(eta)

eta = fleta[0]




