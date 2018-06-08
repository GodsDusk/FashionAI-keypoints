#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np 
import time 

dropout_rate = 0.2
training = True

def conv_bn_relu(inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv_bn_relu'):

    kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
    conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding='VALID', data_format='NHWC')
    norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = training)

    return norm


def residual(inputs, numOut, name = 'residual_block'):

    convb = conv_block(inputs, numOut)
    skipl = skip_layer(inputs, numOut)
    return tf.add_n([convb, skipl], name = 'res_block')


def conv_block(inputs, numOut, name = 'conv_block'):

    with tf.name_scope('norm_1'):
        norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = training)
        conv_1 = conv2d(norm_1, int(numOut/2), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
    with tf.name_scope('norm_2'):
        norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = training)
        pad = tf.pad(norm_2, np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
        conv_2 = conv2d(pad, int(numOut/2), kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
    with tf.name_scope('norm_3'):
        norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = training)
        conv_3 = conv2d(norm_3, int(numOut), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
    return conv_3




def conv2d(inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv'):

    kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
    conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')

    return conv



def skip_layer(inputs, numOut, name = 'skip_layer'):

    with tf.name_scope(name):
        if inputs.get_shape().as_list()[3] == numOut:
            return inputs
        else:
            conv = conv2d(inputs, numOut, kernel_size=1, strides = 1, name = 'conv')
            return conv 



def hourglass(inputs, n, numOut, name = 'hourglass'):

    with tf.name_scope(name):
        # Upper Branch
        up_1 = residual(inputs, numOut, name = 'up_1')
        # Lower Branch
        low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')
        low_1= residual(low_, numOut, name = 'low_1')
        
        if n > 0:
            low_2 = hourglass(low_1, n-1, numOut, name = 'low_2')
        else:
            low_2 = residual(low_1, numOut, name = 'low_2')
            
        low_3 = residual(low_2, numOut, name = 'low_3')
        # up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3]*2, name = 'upsampling')
        up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(up_1)[1:3], name = 'upsampling')
        # print up_2, up_1
        return tf.add_n([up_2,up_1], name='out_hg')

def net(inputs, outDim, nFeat=512, nStack=4, nLow=4):

    pad1 = tf.pad(inputs, [[0,0],[2,2],[2,2],[0,0]], name='pad_1')
    conv1 = conv_bn_relu(pad1, filters= 64, kernel_size = 6, strides = 2, name = 'conv_256_to_128')
    r1 = residual(conv1, numOut = 128, name = 'r1')
    pool1 = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], padding='VALID')

    r2 = residual(pool1, numOut= int(nFeat/2), name = 'r2')
    r3 = residual(r2, numOut= nFeat, name = 'r3')

    hg = hourglass(r3, nLow, nFeat, 'hourglass')
    drop = tf.layers.dropout(hg, rate = dropout_rate, training = training, name = 'dropout')
    ll = conv_bn_relu(drop, nFeat, 1,1, 'VALID', name = 'conv')
    ll_0 =  conv2d(ll, nFeat, 1, 1, 'VALID', 'll')

    out = conv2d(ll, outDim, 1, 1, 'VALID', 'out')
    tf.add_to_collection('heatmaps', out)
    out_ = conv2d(out, nFeat, 1, 1, 'VALID', 'out_')
    sum_ = tf.add_n([out_, ll, r3], name = 'merge')
    for _ in range(1, nStack -1):
        hg = hourglass(sum_, nLow, nFeat, 'hourglass')
        drop = tf.layers.dropout(hg, rate = dropout_rate, training = training, name = 'dropout')
        ll = conv_bn_relu(drop, nFeat, 1, 1, 'VALID', name= 'conv')
        ll_ = conv2d(ll, nFeat, 1, 1, 'VALID', 'll')
        out = conv2d(ll, outDim, 1, 1, 'VALID', 'out')
        tf.add_to_collection('heatmaps', out)

        out_ = conv2d(out, nFeat, 1, 1, 'VALID', 'out_')
        sum_ = tf.add_n([out_, sum_, ll_0], name= 'merge')


    with tf.name_scope('stage_' + str(nStack -1)):
        hg = hourglass(sum_, nLow, nFeat, 'hourglass')
        drop = tf.layers.dropout(hg, rate = dropout_rate, training = training, name = 'dropout')
        ll = conv_bn_relu(drop, nFeat, 1, 1, 'VALID', 'conv')
        out = conv2d(ll, outDim, 1,1, 'VALID', 'out')

    tf.add_to_collection('heatmaps', out)
    return out  

if __name__ == '__main__':
    
    im = tf.placeholder(tf.float32, shape=[1, 626, 468, 3])
    pred = net(im, 16)
    print pred
    heatmaps = tf.get_collection('heatmaps')
    # print heatmaps
    writer = tf.summary.FileWriter('log', tf.get_default_graph())
    writer.close()