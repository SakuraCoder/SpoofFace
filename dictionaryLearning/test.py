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

f = h5py.File(sys.argv[1],'r')

features = []
V = []
X = f['data']
Y = f['label']
#X = np.float64(X)
#X = normalize(X, axis = 0)
# print X[0]
# print X.shape
print "learn the dictionary"
dico = DictionaryLearning(n_components=512, alpha=1, max_iter=10, verbose = 20, transform_algorithm = 'lasso_lars')
#dico = MiniBatchDictionaryLearning(n_components=512, alpha=1, batch_size = 32, n_iter=30000, verbose = 20, transform_algorithm = 'lasso_lars')
#dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500, verbose = 20)
dico.fit(X)
print "learn over"
V = dico.components_
# f_out = h5py.File(sys.argv[2],'w')
# f_out['dico'] = V
# f_out.close()
joblib.dump(dico, 'dico_model.pkl') 
print V.shape
#clf = SVC()
#clf.fit(code, Y)
