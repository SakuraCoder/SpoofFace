import time
import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import h5py
import optunity
import optunity.metrics

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import normalize

f_data = h5py.File(sys.argv[1],'r')
dico = joblib.load(sys.argv[2]) 

data = f_data['data']
label = f_data['label']
code = dico.transform(data)

f = h5py.File('transformed_data.h5','w')
f['data'] = code
f.close()

print "transform done"

@optunity.cross_validated(x = code, y = label, num_folds = 10, num_iter = 2)
def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
	model = SVC(C = 10 ** logC, gamma = 10 ** logGamma).fit(x_train, y_train)
	decision_values = model.decision_function(x_test)
	return optunity.metrics.roc_auc(y_test, decision_values)

hps, _, _ = optunity.maximize(svm_auc, num_evals = 200, logC = [-5, 2], logGamma = [-5, 1])
print "C = ", 10 ** hps['logC']
print "Gamma = ", 10 ** hps['logGamma']
print hps['logC']
print hps['logGamma']