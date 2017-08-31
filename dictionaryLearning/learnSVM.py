import time
import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import h5py

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import normalize

f_data = h5py.File(sys.argv[1],'r')
#f_model = h5py.File(sys.argv[2],'r')
dico = joblib.load(sys.argv[2]) 
#dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
#dico.components_ = f['dico']
data = f_data['data']
label = f_data['label']
#data = np.float64(data)
#data = normalize(data, axis = 0)
#print data.dtype
#print data[0:5]
#print dico.components_
code = dico.transform(data)

#print code
print "learn SVM"
#clf = SVC(C = 10, gamma = 1, verbose = 20)
clf = LinearSVC(max_iter=10000000)
clf.fit(code, label)
print "learn over"
joblib.dump(clf, 'SVM_model.pkl')
