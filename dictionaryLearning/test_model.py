import time
import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import h5py

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import normalize

def findThreshold(scores, labels):
	r = np.arange(0, 1, 0.001)
	FAR = []
	FRR = []
	for i in range(len(r)):
		FA = 0
		FR = 0
		for j in range(len(scores)):
			if(scores[j] >= r[i]):
				if(labels[j] == 0):
					FA = FA + 1
			else:
				if(labels[j]):
					FR = FR + 1
		FAR.append(1.0*FA/ len(scores))
		FRR.append(1.0*FR/ len(scores))

	diff = np.asarray(FAR) - np.asarray(FRR)
	diff = [abs(num) for num in diff]
	index = diff.index(min(diff))
	return r[index], 0.5*(FRR[index] + FAR[index])



dico = joblib.load(sys.argv[1]) 
clf = joblib.load(sys.argv[2])
f = h5py.File(sys.argv[3],'r')
f_test = h5py.File(sys.argv[4],'r')
X = f['data']
Y = f['label']
X = np.float64(X)
code = dico.transform(X)
scores = clf.decision_function(code)
minV = min(scores)
maxV = max(scores)
for i in range(len(scores)):
	scores[i] = 1.0*(scores[i] - minV)/ (maxV - minV)

# print clf.predict(code)
# print Y
# print scores
r = findThreshold(scores, Y)
print r
#print dico.components_
#print clf.support_vectors_[0]

X = f_test['data']
Y = f_test['label']
X = np.float64(X)

code = dico.transform(X)
scores = clf.decision_function(code)
minV = min(scores)
maxV = max(scores)
for i in range(len(scores)):
	scores[i] = 1.0*(scores[i] - minV)/ (maxV - minV)
FA = 0
FR = 0
for j in range(len(scores)):
	if(scores[j] >= r):
		if(Y[j] == 0):
			FA = FA + 1
	else:
		if(Y[j]):
			FR = FR + 1
FAR = 1.0*FA / len(scores)
FRR = 1.0*FR / len(scores)
HTER = 0.5*(FAR + FRR)
print "FAR = ", FAR
print "FRR = ", FRR
print "HTER = ", HTER
