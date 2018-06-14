#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def prettyPicture(clf, X_test, y_test, pred):
    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    wrong_preds = []
    for index in range(len(pred)):
        if pred[index] != y_test[index]:
            wrong_preds.append(index)

    wrong_pred_features = []
    wrong_pred_labels = []
    for index in wrong_preds:
        wrong_pred_features.append(X_test[index])
        wrong_pred_labels.append(y_test[index])

    # display the wrong prediction
    ### the training data (features_train, labels_train) have both "fast" and "slow"
    ### points mixed together--separate them so we can give them different colors
    ### in the scatterplot and identify them visually
    wrong_grade_fast = [wrong_pred_features[ii][0] for ii in range(0, len(wrong_pred_features)) if wrong_pred_labels[ii]==0]
    wrong_bumpy_fast = [wrong_pred_features[ii][1] for ii in range(0, len(wrong_pred_features)) if wrong_pred_labels[ii]==0]
    wrong_grade_slow = [wrong_pred_features[ii][0] for ii in range(0, len(wrong_pred_features)) if wrong_pred_labels[ii]==1]
    wrong_bumpy_slow = [wrong_pred_features[ii][1] for ii in range(0, len(wrong_pred_features)) if wrong_pred_labels[ii]==1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
    plt.scatter(wrong_bumpy_fast, wrong_grade_fast, color = "w", label="false fast")
    plt.scatter(wrong_grade_slow, wrong_bumpy_slow, color = "g", label="false slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")

    plt.savefig("test.png")

import base64
import json
import subprocess

def output_image(name, format, bytes):
    image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
    image_end = "END_IMAGE_0238jfw08fjsiufhw8frs"
    data = {}
    data['name'] = name
    data['format'] = format
    data['bytes'] = base64.encodestring(bytes)
    print image_start+json.dumps(data)+image_end
                                    
