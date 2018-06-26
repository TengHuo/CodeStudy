#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by Teng on 24/06/2018

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('./input/train.csv')
images = data.values[:, 1:]
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)
print('Images shape: {}'.format(images.shape))

labels = data.values[:, 0]
onehot_labels = np.zeros([labels.shape[0], 10])
for i in range(labels.shape[0]):
    onehot_labels[i][labels[i]] = 1

labels = onehot_labels.astype(np.uint8)
print('Labels shape: {}'.format(labels.shape))


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = - tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


TEST_SIZE = 2000
test_images = images[:TEST_SIZE]
test_labels = labels[:TEST_SIZE]

train_images = images[TEST_SIZE:]
train_labels = labels[TEST_SIZE:]

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]


def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all training data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(1000):
    batch_xs, batch_ys = next_batch(200)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 50 == 0:
        print('ACCURACY: {}'.format(sess.run(accuracy, feed_dict={x: test_images, y_: test_labels})))

sess.close()
