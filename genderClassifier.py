# CREDITS: SIRAJ RAVAL

from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Data and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
     [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
     [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female',
     'female', 'female', 'male', 'male']


# Classifiers with default hyperparametersl
clf_tree = tree.DecisionTreeClassifier()
clf_perceptron = Perceptron()
clf_svm = SVC()
clf_KNN = KNeighborsClassifier()

# Training all the models using Data X and labels Y
clf_tree.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_svm.fit(X, Y)
clf_KNN.fit(X, Y)

# Testing using the same data
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print("Accuracy of DecisionTree: {}" .format(acc_tree))

pred_perceptron = clf_perceptron.predict(X)
acc_per = accuracy_score(Y, pred_perceptron) * 100
print("Accuracy of Perceptron: {}" .format(acc_per))


pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print("Accuracy of SVM: {}" .format(acc_svm))


pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print("Accuracy of KNN: {}" .format(acc_KNN))

# The best classifier from svm, per, KNN
index = np.argmax([acc_tree, acc_per, acc_svm, acc_KNN])
classifiers = {0: 'DecisionTree', 1: 'Perceptron', 2: 'SVM', 3: 'KNN'}
print('Best gender classifier is {}'.format(classifiers[index]))
