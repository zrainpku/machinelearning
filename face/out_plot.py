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

#by  zr
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
ada_discrete.fit(X_train_pca, Y)

fig = plt.figure()
ax = fig.add_subplot(111)

rf_clf= RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=12)
error_rate=[]
for i in range(1, 400 + 1):
        rf_clf.set_params(n_estimators=i)
        rf_clf.fit(X_train_pca, Y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - rf_clf.oob_score_
        error_rate.append(oob_error)
# rf_clf = rf_clf.fit(X_train, y_train)

gnb_clf = GaussianNB()

ada_discrete_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_test_pca)):
    ada_discrete_err[i] = zero_one_loss(y_pred, Y_)

ada_discrete_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_train_pca)):
    ada_discrete_err_train[i] = zero_one_loss(y_pred, Y)

ax.plot(np.arange(n_estimators) + 1, ada_discrete_err,
        label=' AdaBoost Test Error',
        color='red')
ax.plot(np.arange(n_estimators) + 1, ada_discrete_err_train,
        label=' AdaBoost Train Error',
        color='blue')

ax.plot(np.arange(n_estimators) + 1, error_rate,
        label='RandomForestClassifier Train Error',
        color='green')

ax.set_ylim((0.0, 0.6))
ax.set_xlabel('n_estimators')
ax.set_ylabel('error rate')

leg = ax.legend(loc='lower left', fancybox=True)
leg.get_frame().set_alpha(0.7)


plt.show()


