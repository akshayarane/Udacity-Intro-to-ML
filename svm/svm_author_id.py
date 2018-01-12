#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.
    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
sys.path.append("/home/akshaya/udacity-ml-course/ud120-projects/tools/")
from time import time
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


features_train, features_test, labels_train, labels_test = preprocess()
#To speed up the algorithm by considering only 1% of the total data
##features_train = features_train[:len(features_train)/100] 
##labels_train = labels_train[:len(labels_train)/100] 

########################## SVM #################################


def submitAccuracy():
    return accuracy_score(pred, labels_test)

clf = SVC(kernel="rbf", C=10000.0)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
#to predict the label class of element 20 in the test sample.
##pred = clf.predict(features_test)[20]
##print pred
print "predicting time:", round(time()-t1, 3), "s"
t2 = time()
print "accuracy: ", accuracy_score(pred, labels_test)
print "scoring time:", round(time()-t2, 3), "s"

## To get number of predicted emails written by Chris.  Ans: 877
count = 0
for i in range(1,1700):
    pred = clf.predict(features_test)[i]
    i = i + 1
    count = count + pred
print count

#################################################################
