import time
import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import h5py

from sklearn.decomposition import MiniBatchDictionaryLearning, DictionaryLearning
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import normalize

f = h5py.File(sys.argv[1],'r')

features = []
V = []
X = f['data']
Y = f['label']
X = np.float64(X)
print "learn the dictionary"
# dico = DictionaryLearning(n_components=512, alpha=1, max_iter=20, verbose = 20)
# dico.fit(X)
dico = MiniBatchDictionaryLearning(n_components=512, alpha=1, batch_size = 32, n_iter=3000)
for i in range(100):
	dico.partial_fit(X)
	print "epoch " +str(i) + " done"
	code = dico.transform(X)
	error = X - np.dot(code, dico.components_)
	print "error = ", np.sum(error)
	joblib.dump(dico, 'model/dico_model_batch_iter' + str(i) + '.pkl')
#dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500, verbose = 20)
#dico.fit(X)
print "learn over"
V = dico.components_
# f_out = h5py.File(sys.argv[2],'w')
# f_out['dico'] = V
# f_out.close()
joblib.dump(dico, 'dico_model_batch.pkl') 
print V.shape
#clf = SVC()
#clf.fit(code, Y)