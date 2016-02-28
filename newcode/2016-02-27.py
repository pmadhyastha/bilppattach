
import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as ss
import combopre
import comboprep
trX, trY, trXl, trV, trN, trP, trM, deX, deY, deXl, deV, deN, deP, deM, featset = comboprep.dataextract(pp='all')
bestlinmodel = combo_all_0.000000001_0.000001_0.000000001_0.00001_50/modellin500.0000000010.000001all.npy 
bestlinmodel = np.load('combo_all_0.000000001_0.000001_0.000000001_0.00001_50/modellin500.0000000010.000001all.npy')
bestbilmodel = np.load('combo_all_0.000000001_0.000001_0.000000001_0.00001_50/modelbil500.0000000010.00001all.npy')
comboprep.test('all', bestbilmodel, bestlinmodel, featset)
bestbilmodel = np.load('combo_all_0.000000001_0.00000001_0.000000001_0.01_500/modelbil5000.0000000010.01all.npy')
bestlinmodel = np.load('combo_all_0.000000001_0.00000001_0.000000001_0.01_500/modellin5000.0000000010.00000001all.npy')
comboprep.test('all', bestbilmodel, bestlinmodel, featset)
bestlinmodel = np.load('combo_all_0.000000001_0.000001_0.000000001_0.00001_500/modelbil5000.0000000010.00001all.npy')
bestlinmodel = np.load('combo_all_0.000000001_0.000001_0.000000001_0.00001_500/modellin5000.0000000010.000001all.npy')
bestbilmodel = np.load('combo_all_0.000000001_0.000001_0.000000001_0.00001_500/modelbil5000.0000000010.00001all.npy')
comboprep.test('all', bestbilmodel, bestlinmodel, featset)
bestbilmodel = np.load('combo_all_0.000000001_0.01_0.000000001_0.00000001_100_modl_modb/modelbil1000.0000000010.00000001all.npy')
bestlinmodel = np.load('combo_all_0.000000001_0.01_0.000000001_0.00000001_100_modl_modb/modellin1000.0000000010.01all.npy')
comboprep.test('all', bestbilmodel, bestlinmodel, featset)
from glob import glob
for dir in glob('combo*'): 
    bilmodel = np.load(dir+'/modelbil*')
    linmodel = np.load(dir + '/modellin*') 
    comboprep.test('all', bilmodel, linmodel, featset) 
    
for dir in glob('combo*'): 
    bilmodel = np.load(glob(dir+'/modelbil*'))
    linmodel = np.load(glob(dir + '/modellin*')) 
    comboprep.test('all', bilmodel, linmodel, featset)
    
for dir in glob('combo*'):
    print glob(dir+'/modelbil*') 
    
for dir in glob('combo*'):
    print glob(dir+'/modelbil*')[0]
    
for dir in glob('combo*'):
    if len(glob(dir + '/modelbil*)) != 0: 
        print glob(dir + '/modelbil*)[0]
        
for dir in glob('combo*'):
    if len(glob(dir + '/modelbil*)) != 0: 
        print glob(dir + '/modelbil*')[0]
        
for dir in glob('combo*'):
    if len(glob(dir + '/modelbil*')) != 0: 
        print glob(dir + '/modelbil*')[0]
        
for dir in glob('combo*'):
    if len(glob(dir + '/modelbil*')) != 0: 
        bil = glob(dir + '/modelbil*')[0]
    if len(glob(dir + '/modellin*')) != 0:
        lin = lob(dir + '/modellin*')[0]
    comboprep.test('all', bilmodel, linmodel, featset)
    
for dir in glob('combo*'):
    if len(glob(dir + '/modelbil*')) != 0: 
        bil = glob(dir + '/modelbil*')[0]
    if len(glob(dir + '/modellin*')) != 0:
        lin = glob(dir + '/modellin*')[0]
    comboprep.test('all', bilmodel, linmodel, featset)
    
for dir in glob('combo*'):
    if len(glob(dir + '/modelbil*')) != 0: 
        bil = glob(dir + '/modelbil*')[0]
    if len(glob(dir + '/modellin*')) != 0:
        lin = glob(dir + '/modellin*')[0]
    comboprep.test('all', bil, lin, featset)
    
comboprep.test('all', bestbilmodel, bestlinmodel, featset)
comboprep.test(prep='all', bestbilmodel, bestlinmodel, featset)
for dir in glob('combo*'):
    if len(glob(dir + '/modelbil*')) != 0: 
        bil = glob(dir + '/modelbil*')[0]
    if len(glob(dir + '/modellin*')) != 0:
        lin = glob(dir + '/modellin*')[0]
    comboprep.test(prep='all', bil, lin, featset)
    
for dir in glob('combo*'):
    if len(glob(dir + '/modelbil*')) != 0: 
        bil = glob(dir + '/modelbil*')[0]
    if len(glob(dir + '/modellin*')) != 0:
        lin = glob(dir + '/modellin*')[0]
    comboprep.test(prep='all', bil, lin, featset)
    
l;s
get_ipython().magic(u'ls ')
comboprep.test('all', bil, lin, featset) 
comboprep.test(prep='all', bestbilmodel, bestlinmodel, featset)
comboprep.test'all', bestbilmodel, bestlinmodel, featset)
comboprep.test('all', bestbilmodel, bestlinmodel, featset)
comboprep.test('all', bil, lin, featset)
bil
for dir in glob('combo*'):
    if len(glob(dir + '/modelbil*')) != 0: 
        bil = np.load(glob(dir + '/modelbil*')[0])
    if len(glob(dir + '/modellin*')) != 0:
        lin = np.load(glob(dir + '/modellin*')[0])
    comboprep.test(prep='all', bil, lin, featset)
    
for dir in glob('combo*'):
    if len(glob(dir + '/modelbil*')) != 0: 
        bil = np.load(glob(dir + '/modelbil*')[0])
    if len(glob(dir + '/modellin*')) != 0:
        lin = np.load(glob(dir + '/modellin*')[0])
    comboprep.test('all', bil, lin, featset)
    
get_ipython().magic(u'ls *.py')
import bilpreplogistic as bpl 
for dir in glob('bil_*'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil)
    
reload(bpl) 
for dir in glob('bil_*'):
    if len(glob(dir + '/model*')) != 0: 
        bil = np.load(glob(dir + '/model*')[0])
    bpl.test('all', bil)
    
get_ipython().system(u'vim -')
:q!
get_ipython().magic(u'cat > resultsbil.txt ')
