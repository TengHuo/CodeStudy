#!/usr/bin/env python3
# Created by Teng on 25/06/2018

import tensorflow as tf
import pandas as pd
import numpy as np
import datetime

BATCH_SIZE = 50


def preprocess_images(images):
    processed_images = images.astype(np.float)
    processed_images = np.multiply(processed_images, 1.0 / 255.0)
    return processed_images


def preprocess_labels(labels):
    one_hot_labels = np.zeros([labels.shape[0], 10])
    for i in range(labels.shape[0]):
        one_hot_labels[i][labels[i]] = 1

    return one_hot_labels.astype(np.uint8)


data = pd.read_csv('./input/train.csv')
submission_data = pd.read_csv('./input/test.csv')

images = preprocess_images(data.values[:, 1:])
submission_images = preprocess_images(submission_data.values)
print('All Images shape: {}'.format(images.shape))
print('Submission Images shape: {}'.format(submission_images.shape))

labels = preprocess_labels(data.values[:, 0])
print('Labels shape: {}'.format(labels.shape))

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


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 这个函数是干什么的
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(value=0.1, shape=shape)
    return tf.Variable(initial)


# 卷积和池化
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


with tf.device('/cpu:0'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # 把x变成一个4d向量，其第2、第3维对应图片的宽、高，
    # 最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable(([1024]))

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.global_variables_initializer())

    t0 = datetime.datetime.now()
    for i in range(20000):
        batch_xs, batch_ys = next_batch(BATCH_SIZE)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print('step {}, training accuracy {}'.format(i, train_accuracy))
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    t1 = datetime.datetime.now()
    print('Training time: {}'.format((t1 - t0).seconds))

    predicted_labels = sess.run(
        tf.cast(tf.argmax(sess.run(y_conv, feed_dict={x: submission_images, keep_prob: 1.0}), 1), tf.int8))
    sess.close()


np.savetxt('submission.csv',
           np.c_[range(1, len(submission_images)+1), predicted_labels],
           delimiter=',',
           header='ImageId,Label',
           comments='',
           fmt='%d')
print('done')
