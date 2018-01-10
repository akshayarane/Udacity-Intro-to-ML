import sys
sys.path.append("/home/akshaya/udacity-ml-course/ud120-projects/tools/")
sys.path.append('/home/akshaya/udacity-ml-course/ud120-projects/choose_your_own')
sys.path.append('/home/akshaya/udacity-ml-course/ud120-projects/svm')

import os
os.chdir('/home/akshaya/udacity-ml-course/ud120-projects/svm')


from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import sys
import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

from sklearn.metrics import accuracy_score


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
from sklearn.svm import SVC

def submitAccuracy():
    return accuracy_score(pred, labels_test)

clf = SVC(kernel="linear")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

#print accuracy_score(pred, labels_test)

prettyPicture(clf, features_test, labels_test)
plt.show()