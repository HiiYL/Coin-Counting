"""
evaluateSet.py

DO NOT MODIFY ANY CODES IN THIS FILE
OTHERWISE YOUR RESULTS MAY BE INCORRECTLY EVALUATED! 

@author: John See, 2016
For questions or bug reporting, please send an email to johnsee@mmu.edu.my

"""

import cv2
import numpy as np
import pickle
import importlib
import sys, getopt
from prettytable import PrettyTable
import coinCounting as cc

# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"i:vh")
 
# Defaults
verbose = False
pk = 'coinset1.pkl'

################################################
# o == option    a == argument passed to the o #
################################################

# parsing command line args
for o, a in myopts:
    #print(o)
    #print(a)
    if o == '-i':
        pk = a    
    if o == '-v':
        verbose = True
    elif o == '-h':
        print("\nUsage: %s -v               for extra verbosity" % sys.argv[0])
        print("       %s -i filename      to specify pickled coin set data" % sys.argv[0])
        print("       %s -i filename -v   do both" % sys.argv[0])
        sys.exit()
    else:
        print(' ')

# Reload module
importlib.reload(cc)

# Load pickled coin set
coinMat, gt = pickle.load( open(pk, "rb" ) )

# Prepare some variables
ans = np.zeros(10,)
diff = np.zeros(10,)
time = np.zeros(10,)
hits = np.zeros((10,), dtype=np.uint8)

# Evaluate each image and compare with ground-truth
for i in range(10):
    e1 = cv2.getTickCount()
    ans[i] = cc.coinCount(coinMat, i)             # your code executes here
    e2 = cv2.getTickCount()
    time[i] = (e2 - e1) / cv2.getTickFrequency()
    diff[i] = np.abs(gt[i]-ans[i])
    if ans[i] == gt[i]:
        hits[i] = 1;        
    
# Print performance scores        
if verbose:
    print('########  DETAILED RESULTS  ########')
    t = PrettyTable(['Image', 'Accuracy Hit', 'Error (RM)', 'Run time (s)'])
    for i in range(10):
        t.add_row([i+1, hits[i], "%.2f"%(diff[i]), "%.4f"%(time[i]) ])
    t.add_row([' ',' ',' ',' '])
    t.add_row(['All', np.sum(hits), "%.2f"%(np.sum(diff)), "%.4f"%(np.sum(time))])
    print(t)
else:
    print('Total error: $ %.2f'%(np.sum(diff)))
    print('Accuracy: %d%%'%(np.sum(hits)*10))
    print('Code runtime: %.4f seconds'%(np.sum(time)))
        
        
# END OF EVALUATION CODE####################################################