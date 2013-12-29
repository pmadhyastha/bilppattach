import glob
import os
import re
import shutil
import numpy as np
os.chdir('main direc name')
destination = ('main direc name')
source = ('final throwing time for the directory')
devacclist = {}

for files in glob.glob("combo-models/best*"):
    base = re.findall(r'l2pl1|l2pnn|l2pl2p|\d{3,5}|cl[e0-9-]+|cb[0-9-\.]+|eta[0-9.]+|on|for|with|to|from|in', files)
    samples = base[1]
    pptype = base[5]
    regtype = base[0]

shutil.move(destination, source)

np.savetxt('l1nnfinalall.txt', devacclist.items(), fmt='%s')

