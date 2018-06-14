#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
# grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
# bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
# grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
# bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=3)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

# wrong_preds = []
# for index in range(len(pred)):
#     if pred[index] != labels_test[index]:
#         wrong_preds.append(index)

# wrong_pred_features = []
# wrong_pred_labels = []
# for index in wrong_preds:
#     wrong_pred_features.append(features_test[index])
#     wrong_pred_labels.append(labels_test[index])

# # display the wrong prediction

# ### the training data (features_train, labels_train) have both "fast" and "slow"
# ### points mixed together--separate them so we can give them different colors
# ### in the scatterplot and identify them visually
# wrong_grade_fast = [wrong_pred_features[ii][0] for ii in range(0, len(wrong_pred_features)) if wrong_pred_labels[ii]==0]
# wrong_bumpy_fast = [wrong_pred_features[ii][1] for ii in range(0, len(wrong_pred_features)) if wrong_pred_labels[ii]==0]
# wrong_grade_slow = [wrong_pred_features[ii][0] for ii in range(0, len(wrong_pred_features)) if wrong_pred_labels[ii]==1]
# wrong_bumpy_slow = [wrong_pred_features[ii][1] for ii in range(0, len(wrong_pred_features)) if wrong_pred_labels[ii]==1]


# # #### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(wrong_bumpy_fast, wrong_grade_fast, color = "y", label="false fast")
# plt.scatter(wrong_grade_slow, wrong_bumpy_slow, color = "g", label="false slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print accuracy

try:
    prettyPicture(clf, features_test, labels_test, pred)
except NameError:
    pass






# Learning Curve
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit

# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     print train_scores
#     print test_scores
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")

#     plt.legend(loc="best")
#     return plt

# title = "Learning Curves (Naive Bayes)"
# # Cross validation with 100 iterations to get smoother mean test and train
# # score curves, each time with 20% data randomly selected as a validation set.
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

# estimator = RandomForestClassifier()

# # features_train += features_test
# # labels_train += labels_test

# plot_learning_curve(estimator, title, features_train, labels_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
# plt.show()