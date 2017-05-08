from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
#by  me
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.datasets import fetch_lfw_pairs

lfw_pairs_train = fetch_lfw_pairs(subset='train')
# test data
lfw_pairs_test = fetch_lfw_pairs(subset='test')

print(lfw_pairs_train.pairs.shape)
n_samples, c, h, w = lfw_pairs_train.pairs.shape

X=lfw_pairs_train.data
target_names=lfw_pairs_train.target_names
n_features=X.shape[1]
n_classes=target_names.shape[0]
Y=lfw_pairs_train.target
X_=lfw_pairs_test.data
Y_=lfw_pairs_test.target


print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)



print("X.shape=" , (X.shape))
print("Y.shape=" , (Y.shape))

n_components = 100
n_estimators = 400
learning_rate = 0.2

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, c, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X)
X_test_pca = pca.transform(X_)
print("done in %0.3fs" % (time() - t0))

print("X_train_pca.shape=" , (X_train_pca.shape))
print("X_test_pca.shape= " , (X_test_pca.shape))

dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
print("begin to train=====>....")

#1
ada_discrete = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME")
# ada_discrete.fit(X_train_pca, Y)
#2
ada_real = AdaBoostClassifier(
    base_estimator=dt_stump,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    algorithm="SAMME.R")
# ada_real.fit(X_train_pca, Y)

#3
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
svm_clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', probability=True), param_grid)
svm_clf = svm_clf.fit(X_train_pca, Y)
print(svm_clf.best_estimator_)

#4
lr_clf = LogisticRegression(random_state=2)
# lr_clf = lr_clf.fit(X_train_pca, Y)

#5

rf_clf = RandomForestClassifier(random_state=12,n_estimators=400,max_depth=5)
# rf_clf = rf_clf.fit(X_train_pca, Y)

#6
gnb_clf = GaussianNB()
# gnb_clf = gnb_clf.fit(X_train_pca, Y)


#7  the vote methed 
end_clf = VotingClassifier(estimators=[('adareal', ada_real), ('svm_clf', svm_clf), ('rf', rf_clf), ('gnb', gnb_clf)],voting='soft')
# end_clf=end_clf.fit(X_train_pca, Y)



#@@@@@@@@@@   change rf_clf  to  the method you wanted     @@@@ 
print("Predicting is or not same:")
t0 = time()
y_pred = svm_clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(Y_, y_pred, target_names=target_names))
print(confusion_matrix(Y_, y_pred, labels=range(n_classes)))
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

ans=0
for i in range(1000):
	# print(Y[i]),
	if y_pred[i]==Y_[i]:
		ans+=1

ans=ans*1.0/1000.0
print("the accuracy is:",ans)
















