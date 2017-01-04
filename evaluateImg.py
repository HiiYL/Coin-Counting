"""
evaluateImg.py

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
import coinCounting as cc

# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"i:h")
 
# Defaults
pk = 'coinset1.pkl'
n = []

################################################
# o == option    a == argument passed to the o #
################################################

# parsing command line args
for o, a in myopts:
    if o == '-i':
        pk = a    
    elif o == '-h':
        print("\nUsage: %s n       n denotes image number in coin set, n=1,...,10" % sys.argv[0])
        print("       %s -i filename     to specify pickled coin set data" % sys.argv[0])
        print("       %s -i filename n   do both" % sys.argv[0])
        sys.exit()
    else:
        print(' ')

if not args:
    print("Error: No image number provided!")

for ar in args:
    try:
        n = int(args[0])
    except ValueError:
        print("Error: Input argument expecting a number!")
        sys.exit()
       
        
# Reload module
importlib.reload(cc)

# Load pickled coin set
coinMat, gt = pickle.load( open(pk, "rb" ) )

# Prepare some variables
ans = np.zeros(1,)
diff = np.zeros(1,)
time = np.zeros(1,)

# Evaluate image and compare with ground-truth
e1 = cv2.getTickCount()
ans = cc.coinCount(coinMat, n-1)             # your code executes here
e2 = cv2.getTickCount()
time = (e2 - e1) / cv2.getTickFrequency()
diff = np.abs(gt[n-1]-ans)
     
# Print performance scores        
print('Evaluating coin image %d from %s'%(n,pk))
print('Your answer: %.2f | Ground-truth: %.2f '%(ans,gt[n-1]))
print('You are off by %.2f'%(diff))
print('Code runtime: %.2f seconds'%(time))
        
        
# END OF EVALUATION CODE####################################################