#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by Teng on 09/05/2018

# import the sklearn module for GaussianNB
# create classifier
# fit the classifier on the training features and labels
# return the fit classifier

from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):
    clf = GaussianNB()
    clf.fit(features_train, labels_train)

    return clf