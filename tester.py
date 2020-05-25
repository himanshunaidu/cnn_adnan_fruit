##TIMING
import time
start_time = time.time()

##SETUP

import tensorflow as tf

from retriever import IMG_PX_SIZE, classLength

import os
print(os.getcwd())

#ALL HELPER FUNCTIONS

#INIT WEIGHTS

def init_weights(shape):
    init_rand_dist = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init_rand_dist)

#INIT BIAS

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def init_bias_2(shape, value):
    init_bias_vals = tf.constant(value, shape=shape)
    return tf.Variable(init_bias_vals)

#CONV2D

def conv2d(x, W, stridec=[1, 1, 1, 1]):
    # x---> [batch, h, w, channels]
    # W---> [filter_h, filter_w, channel_in, channel_out]
    print(x.shape)
    
    return tf.nn.conv2d(x, W, strides=stridec, padding='SAME')

#POOLING
#Fixed ksize and strides

def max_pool_3by3(x):
    # x---> [batch, h, w, channels]
    print(x.shape)
    
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

#CONVOLUTIONAL LAYER

def convolutional_layer(input_x, shape, stridec=[1, 1, 1, 1]):
    W = init_weights(shape)
    b = init_bias_2([shape[3]], 0.0)
    #must cast b to float32
    return tf.nn.relu(conv2d(input_x, W, stridec=stridec)+b)

def convolutional_layer_2(input_x, shape, stridec=[1, 1, 1, 1]):
    W = init_weights(shape)
    return conv2d(input_x, W, stridec=stridec)

#NORMAL LAYER

def normal_full_layer(input_layer, size):
    print(input_layer.shape)
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias_2([size], 1.0)
    return tf.matmul(input_layer, W)+b

#LRN

def LRN(x, R, alpha, beta, name = None, bias = 1.0):
    """LRN"""
    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
                                              beta = beta, bias = bias, name = name)

#PLACEHOLDER

#x = tf.placeholder(tf.float32, shape=[None, IMG_PX_SIZE*IMG_PX_SIZE])

y_true = tf.placeholder(tf.float32, shape=[None, classLength])

#LAYERS

x_image = tf.placeholder(tf.float32, shape=[None, IMG_PX_SIZE, IMG_PX_SIZE, 3])
#x_image = tf.reshape(x, [-1, IMG_PX_SIZE, IMG_PX_SIZE, 1])

#decoy = convolutional_layer(x_image, shape=[11, 11, 1, 64])

#Take in RGB images
convo1 = convolutional_layer(x_image, shape=[11, 11, 3, 64], stridec=[1, 4, 4, 1])
#lrn1 = LRN(convo1, 2, 2e-05, 0.75, "norm1")
convo1_pool = max_pool_3by3(convo1)

convo2 = convolutional_layer(convo1_pool, shape=[5, 5, 64, 192])
#lrn2 = LRN(convo2, 2, 2e-05, 0.75, "norm1")
convo2_pool = max_pool_3by3(convo2)

convo3 = convolutional_layer(convo2_pool, shape=[5, 5, 192, 384])

convo4 = convolutional_layer(convo3, shape=[3, 3, 384, 256])

convo5 = convolutional_layer(convo4, shape=[3, 3, 256, 256])
convo5_pool = max_pool_3by3(convo5)
print('Convolutional layers end')

#the dimension 8*8*256 might be wrong because of insufficient details of strides and paddings of some layers

convo5_flat = tf.reshape(convo5_pool, [-1, 7*7*256])
full_layer_one = tf.nn.relu(normal_full_layer(convo5_flat, 4096))

#DROPOUT
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

full_layer_two = tf.nn.relu(normal_full_layer(full_one_dropout, 4096))

#DROPOUT 2
full_two_dropout = tf.nn.dropout(full_layer_two, keep_prob=hold_prob)

full_layer_three = tf.nn.relu(normal_full_layer(full_two_dropout, 4096))

y_pred = normal_full_layer(full_layer_three, 100)

#LOSS FUNCTION
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

#OPTIMIZER
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

print('Setup Completed')

##THIS MARKS THE END OF THE SETUP

##DATASET GENERATION

from retriever import getSubFolders, getImageArray, datasetPath, testPath
import numpy as np
import matplotlib.pyplot as plt

subfolders = getSubFolders(datasetPath)
print('Subfolders saved')

#Total train images: 300 * 15 = 4500
#Total test images: 5 * 15 = 75
itest = getImageArray(testPath, subfolders, 140)

print('Dataset Generated')

##THIS MARKS THE END OF THE DATASET GENERATION

##SAVER SETUP

from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

var_export_dir = '<PATH>/TempF2/variables.ckpt'
saver = tf.train.Saver()

test_saver = open('<PATH>/TempF2/test.txt', 'a+')

import logging
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('<PATH>/tensorflow.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

##THIS MARKS THE END OF THE SAVER SETUP

##TRAINING

from utils import nextImageBatch, nextImageRandomBatch, nextFullBatch

steps = 70
accuracy = 0

with tf.Session() as sess:
    sess.run(init)
    actual_steps = 0
    runs = 0
    index = 0

    testlength = int(len(itest)/70)

    saver.restore(sess, var_export_dir)
    
    for i in range((actual_steps-runs), steps):

        matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        test_x, test_y, index = nextImageBatch(itest, testlength, len(subfolders), index)

        acc = tf.reduce_mean(tf.cast(matches, tf.float32))

        print("ACCURACY: ")
        get_acc = sess.run(acc, feed_dict={x_image:test_x, y_true:test_y, hold_prob:1.0})
        print(get_acc)

        test_saver.write(str(get_acc)+'\n')

        accuracy = accuracy + get_acc
        

final_acc = accuracy / steps
print(final_acc)

#Used to check the time required for the testing
time_saver = open('<PATH>/TempF2/time.txt', 'a+')
time_saver.write(str(time.time() - start_time)+'\n')
time_saver.write(str(final_acc))
time_saver.close()

test_saver.close()