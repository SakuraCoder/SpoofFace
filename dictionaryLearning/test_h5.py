import time
import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import h5py

f = h5py.File(sys.argv[1],'r')

X = f['data']
Y = f['label']

print X.shape
print Y.shape
print np.mean(X, axis = 1)
print np.std(X, axis = 1)