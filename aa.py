import numpy as np
import matplotlib.pyplot as plt

t100 = ['1000tau100eta0.0001', '2000tau100eta1e-05', '3000tau100eta1e-05', '4000tau100eta1e-05', '5000tau100eta1e-05', '6000tau100eta1e-05', '7000tau100eta1e-05', '8000tau100eta1e-05', '9000tau100eta1e-05', '10000tau100eta1e-05', '11000tau100eta1e-05', '12000tau100eta1e-05', '13000tau100eta1e-05', '14000tau100eta1e-05', '15000tau100eta1e-05', '16000tau100eta1e-05', '17000tau100eta1e-05', '18000tau100eta1e-05', '19000tau100eta1e-05', '20000tau100eta1e-05', '20801tau100eta1e-05']
t50 = ['1000tau50eta0.0001', '2000tau50eta0.0001', '3000tau50eta0.0001', '4000tau50eta0.0001', '5000tau50eta1e-05', '6000tau50eta1e-05', '7000tau50eta1e-05', '8000tau50eta1e-05', '9000tau50eta1e-05', '10000tau50eta1e-05', '11000tau50eta1e-05', '12000tau50eta1e-05', '13000tau50eta1e-05', '14000tau50eta1e-05', '15000tau50eta1e-05', '16000tau50eta1e-05', '17000tau50eta1e-05', '18000tau50eta1e-05', '19000tau50eta1e-05', '20000tau50eta1e-05', '20801tau50eta1e-05']
t10 = ['1000tau10eta0.001', '2000tau10eta0.0001', '3000tau10eta0.0001', '4000tau10eta0.0001', '5000tau10eta0.0001', '6000tau10eta0.0001', '7000tau10eta0.0001', '8000tau10eta0.0001', '9000tau10eta0.0001', '10000tau10eta0.0001', '11000tau10eta0.0001', '12000tau10eta0.0001', '13000tau10eta0.0001', '14000tau10eta0.0001', '15000tau10eta0.0001', '16000tau10eta0.0001', '17000tau10eta0.0001', '18000tau10eta0.0001', '19000tau10eta0.0001', '20000tau10eta0.0001', '20801tau10eta0.0001' ]
t1 = ['1000tau1eta0.01', '2000tau1eta0.001', '3000tau1eta0.001', '4000tau1eta0.001', '5000tau1eta0.001', '6000tau1eta0.001', '7000tau1eta0.001', '8000tau1eta0.001', '9000tau1eta0.001', '10000tau1eta0.001', '11000tau1eta0.001', '12000tau1eta0.001', '13000tau1eta0.001', '14000tau1eta0.001', '15000tau1eta0.001', '16000tau1eta0.001', '17000tau1eta0.001', '18000tau1eta0.001', '19000tau1eta0.001', '20000tau1eta0.001', '20801tau1eta0.001']
t01 = ['1000tau0.1eta0.1', '2000tau0.1eta0.05', '3000tau0.1eta0.1', '4000tau0.1eta0.01', '5000tau0.1eta0.01', '6000tau0.1eta0.01', '7000tau0.1eta0.05', '8000tau0.1eta0.01', '9000tau0.1eta0.01', '10000tau0.1eta0.01', '11000tau0.1eta0.01', '12000tau0.1eta0.01', '13000tau0.1eta0.01', '14000tau0.1eta0.01', '15000tau0.1eta0.01', '16000tau0.1eta0.01', '17000tau0.1eta0.01', '18000tau0.1eta0.01', '19000tau0.1eta0.01', '20000tau0.1eta0.01', '20801tau0.1eta0.01']
t001 = ['1000tau0.01eta1', '2000tau0.01eta0.5', '3000tau0.01eta1', '4000tau0.01eta0.5', '5000tau0.01eta0.1', '6000tau0.01eta0.5', '7000tau0.01eta0.5', '8000tau0.01eta0.1', '9000tau0.01eta0.05', '10000tau0.01eta0.1', '11000tau0.01eta0.1', '12000tau0.01eta0.05', '13000tau0.01eta0.1', '14000tau0.01eta0.05', '15000tau0.01eta0.05', '16000tau0.01eta0.1', '17000tau0.01eta0.1', '18000tau0.01eta0.05', '19000tau0.01eta0.05', '20000tau0.01eta0.05', '20801tau0.01eta0.05', '',
        ]


tl2100 = []
tl250 = []
tl210 = []
tl21 = []
tl201 = []
tl2001 = []

for e in t100:
    nlogl = np.loadtxt('devsets/devaccl2'+str(e)+'.txt', dtype=float)
    tl2100.append(nlogl[-1])

for e in t50:
    nlogl = np.loadtxt('devsets/devaccl2'+str(e)+'.txt', dtype=float)
    tl250.append(nlogl[-1])

for e in t10:
    nlogl = np.loadtxt('devsets/devaccl2'+str(e)+'.txt', dtype=float)
    tl210.append(nlogl[-1])

for e in t1:
    nlogl = np.loadtxt('devsets/devaccl2'+str(e)+'.txt', dtype=float)
    tl21.append(nlogl[-1])

for e in t01:
    nlogl = np.loadtxt('devsets/devaccl2'+str(e)+'.txt', dtype=float)
    tl201.append(nlogl[-1])

for e in t001:
    nlogl = np.loadtxt('devsets/devaccl2'+str(e)+'.txt', dtype=float)
    tl2001.append(nlogl[-1])

lcsamples = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 20801]

plt.figure(figsize=(20, 20))
plt.subplot(1, 1, 1)
plt.ylabel(r'development accuracy', fontsize=18)
plt.plot(lcnone, label='none')
plt.plot(lcl2, label='l2')
plt.plot(lcl1, label='l1')
plt.xticks(np.arange(len(lcsamples)), lcsamples)
plt.legend(bbox_to_anchor=(1, 1), loc=4, borderaxespad=0.)
plt.grid()


