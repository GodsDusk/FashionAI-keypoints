import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import time
import argparse
import Queue
import threading

import numpy as np
import tensorflow as tf 
from tensorflow.python.framework import graph_util 
import matplotlib.pyplot as plt
from scipy.misc import imresize

from stack_hourglass import net
from data_utils import read_data, gen_traindata
from config import *

def data_generator(batch_size):

    data = read_data(batch_size)
    iterator = data.make_initializable_iterator()
    batch_images, batch_outputs, batch_categorys = iterator.get_next() 
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in xrange(800000):
            images, label, category = sess.run([batch_images, batch_outputs, batch_categorys])
            im, hmp, c = gen_traindata(images, label, category)
            message.put([i, im, hmp, c])
        message.put(None)


def train(nstack, save_name='hg'):

    landmark_len = len(landmarks)
   
    inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    heatmap = tf.placeholder(tf.float32, shape=[None, None, None, landmark_len])
    calculable = tf.placeholder(tf.float32, shape=[None, landmark_len])
    pred = net(inputs, landmark_len, nStack=nstack)
    outputs = tf.get_collection('heatmaps')

    c_loss = tf.reduce_mean(tf.stack([tf.nn.l2_loss(calculable*(heatmap - o)) for o in outputs]))

    # loss = tf.reduce_sum(tf.stack([tf.nn.l2_loss(heatmap - o) for o in outputs]))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # optimizer = tf.train.MomentumOptimizer(1e-3, 0.9).minimize(c_loss)
        optimizer = tf.train.AdamOptimizer(1e-3).minimize(c_loss)


    saver = tf.train.Saver(max_to_keep=2)        
    with tf.Session() as sess:
        
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'models516/hg-220000')
        st = time.time()

        while True:

            data = message.get()
            if data is None:
                break
            step, im ,hmp, c = data


            sess.run(optimizer, feed_dict={inputs:im, heatmap:hmp, calculable:c})
            if not step % 2000:
                c_e = sess.run(c_loss, feed_dict={inputs:im, heatmap:hmp, calculable:c})
                print step, c_e, time.time() - st 
                st = time.time()
            if not step % 10000:
                saver.save(sess, 'models/{}/{}'.format(nstack, save_name), global_step=step)

        saver.save(sess, 'models/{}/{}'.format(nstack, save_name))
        
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["stage_3/conv2d/BiasAdd"])
        with tf.gfile.FastGFile('models/{}/model.pb'.format(nstack), mode='wb') as f:
            f.write(constant_graph.SerializeToString())

        # p = sess.run(pred, feed_dict={inputs:im})
        # for i in xrange(landmark_len):
        
        #     plt.imshow(imresize(im[0,:,:,:], 1/4.0))
        #     plt.imshow(p[0,:,:,i], alpha=0.5)
        #     plt.show()            


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, choices=['0', '1', '2', '3'], default='3')
    parser.add_argument('-nstack', type=int, default=4)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    nstack = args.nstack

    batch_size = 1
    message = Queue.Queue(50)

    producer = threading.Thread(target=data_generator, args=(batch_size,))
    consumer = threading.Thread(target = train, args=(nstack,))

    producer.start()
    consumer.start()
    message.join()