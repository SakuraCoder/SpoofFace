from time import time
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
from sklearn.preprocessing import normalize, scale


def getDataFromFiles(fileName):
	f = open(fileName)

	lines = f.readlines()
	faces = []
	labels = []
	for line in lines:
		items = line.strip().split()
		img = cv2.imread(os.path.join("/data2/RA2/Datasets/replay-attack",items[0]), 0)
		img = img.flatten()
		faces.append(img)
		labels.append(int(items[1]))
	faces = np.asarray(faces)
	labels = np.asarray(labels)
	return faces, labels

# print "hello"
# L = np.array([[1,3], [3,4]])
# L =  scale(L, axis = 1)
# print L
# print L.mean(axis = 1)
# print L.std(axis = 1)
# print normalize(L)
datas, labels = getDataFromFiles(sys.argv[1] + '.txt')

# for i in range(len(datas)):
#  	datas[i] = datas[i] - datas[i].mean()
# datas = normalize(datas)
#datas = scale(datas)
#print datas[0]
if(sys.argv[1] == 'train'):
	m = np.mean(datas, axis = 0)
	sd = np.std(datas, axis = 0)
	fnorm = h5py.File('norm.h5', 'w')
	fnorm['mean'] = m
	fnorm['std'] = sd
	fnorm.close()
else:
	fnorm = h5py.File('norm.h5', 'r')
	m = fnorm['mean']
	sd = fnorm['std']
	m = np.float64(m)
	sd = np.float64(sd)
	fnorm.close()
datas = datas - m
datas = datas / sd
#print datas[0]
	
f = h5py.File(sys.argv[1] + '.h5', 'w')
f['data'] = datas
f['label'] = labels
f.close()