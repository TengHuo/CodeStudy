#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by Teng on 24/06/2018

import tensorflow as tf
import pandas as pd
import numpy as np
import datetime

BATCH_SIZE = 200


def preprocess_images(images):
    processed_images = images.astype(np.float)
    processed_images = np.multiply(processed_images, 1.0 / 255.0)
    return processed_images


def preprocess_labels(labels):
    one_hot_labels = np.zeros([labels.shape[0], 10])
    for i in range(labels.shape[0]):
        one_hot_labels[i][labels[i]] = 1

    return one_hot_labels.astype(np.uint8)


def add_layer(inputs, in_size, out_size, activate_function=None):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, W) + biases

    if activate_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activate_function(Wx_plus_b)
    return outputs


data = pd.read_csv('./input/train.csv')
submission_data = pd.read_csv('./input/test.csv')

images = preprocess_images(data.values[:, 1:])
submission_images = preprocess_images(submission_data.values)
print('All Images shape: {}'.format(images.shape))
print('Submission Images shape: {}'.format(submission_images.shape))

labels = preprocess_labels(data.values[:, 0])
print('Labels shape: {}'.format(labels.shape))


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# y = tf.nn.softmax(tf.matmul(x, W) + b)

layer1 = add_layer(x, 784, 100, activate_function=tf.nn.softmax)
layer2 = add_layer(layer1, 100, 100, activate_function=tf.nn.softmax)
y = add_layer(layer2, 100, 10, activate_function=tf.nn.softmax)

cross_entropy = - tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

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

t0 = datetime.datetime.now()
for i in range(20000):
    batch_xs, batch_ys = next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
        print('Step: {}, Accuracy: {}'.format(i, sess.run(accuracy, feed_dict={x: test_images, y_: test_labels})))
t1 = datetime.datetime.now()
print('Training time: {} seconds'.format((t1 - t0).seconds))

predicted_labels = sess.run(tf.cast(tf.argmax(sess.run(y, feed_dict={x: submission_images}), 1), tf.int8))
sess.close()

np.savetxt('submission.csv',
           np.c_[range(1, len(submission_images)+1), predicted_labels],
           delimiter=',',
           header='ImageId,Label',
           comments='',
           fmt='%d')

print('done')
