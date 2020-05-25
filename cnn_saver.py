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

from retriever import getSubFolders, getImageArray, getImageArrayFull, datasetPath, testPath
import numpy as np
import matplotlib.pyplot as plt

subfolders = getSubFolders(datasetPath)
print('Subfolders saved')

#Total train images: 300 * 15 = 4500
#Total test images: 5 * 15 = 75
itrain = getImageArrayFull(datasetPath, subfolders)
itest = getImageArray(testPath, subfolders, 100)

print('Dataset Generated')

##THIS MARKS THE END OF THE DATASET GENERATION

##SAVER SETUP

from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

var_export_dir = '<PATH>/TempF2/variables.ckpt'
saver = tf.train.Saver()

#export_dir = '<PATH>/TempF2/model1.ckpt'
#builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
#print(builder._export_dir)
#signature = predict_signature_def(inputs={'Image': x_image, 'True': y_true}, outputs={'Pred': y_pred})

step_saver = open('<PATH>/TempF2/steps.txt', 'a+')
acc_saver = open('<PATH>/TempF2/acc.txt', 'a+')

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

#There are 52200 images in training dataset
#If we have each training batch size of 200, we could iterate over the entire dataset in 261 steps
#We want to train over the dataset 10 times
steps = 2610

with tf.Session() as sess:
    sess.run(init)
    actual_steps = 0
    runs = 0
    index = 0
    i1 = 0

    #2 images per class (100 classes)
    trainlength = int(len(itrain)/261)
    testlength = int(len(itest)/50)

    #saver.restore(sess, var_export_dir)
    
    for i in range((actual_steps-runs), steps):

        print('Saving step number');
        step_saver.write(str(i) + '\n')

        step_saver.close()

        step_saver = open('<PATH>/TempF2/steps.txt', 'a+')

        batch_x, batch_y, index = nextImageBatch(itrain, trainlength, len(subfolders), index)
        
        sess.run(train, feed_dict={x_image:batch_x, y_true:batch_y, hold_prob:0.5})
        #print(trainlength)
        
        #print accuracy every few steps
        if (i+1)%10==0:
            i1 = 0
            print("ON STEP {}".format(i))
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            test_x, test_y, i1 = nextImageBatch(itest, testlength, len(subfolders), i1)
            
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            print("ACCURACY: ")
            accuracy = sess.run(acc, feed_dict={x_image:test_x, y_true:test_y, hold_prob:1.0})
            print(accuracy)
            print('\n')

            acc_saver.write(str(i)+'-'+str(accuracy)+'\n')

        if (i+1)%10==0:
            saver.save(sess, var_export_dir)
        
        if ((index+1)%261 == 0):
            index = 0
            i1 = 0
        
    #builder.add_meta_graph_and_variables(sess, tags=["Training"], signature_def_map={'predict': signature})
    #builder.save()

#The file wont be written to until close() is called
step_saver.close()
acc_saver.close()