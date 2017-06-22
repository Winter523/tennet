import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import numpy as np

# 获取数据集
mnist = input_data.read_data_sets('/path/to/MNIST', one_hot=True)

# 定义网络结构
(m, n) = mnist.train.images.shape
(_, o) = mnist.train.labels.shape

INPUT_NODE = n
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1

CONV1_DEEP = 32
CONV1_SIZE = 5
CONV1_STRIDE = 1
POOL1_SIZE = 2
POOL1_STRIDE = 2

CONV2_DEEP = 64
CONV2_SIZE = 5
CONV2_STRIDE = 1
POOL2_SIZE = 2
POOL2_STRIDE = 2

FC = [512, 10]

name_layers = ['layer1_conv1', 'layer2_pool1', 'layer3_conv2', 'layer4_pool2',
               'layer5_fc1', 'layer6_fc2']

# 定义初始化
PARAMETERS_INITIALIZER = tf.truncated_normal_initializer

# 计算神经网络的前向计算结果
def inference(input_tensor, train, regularizer):
    n_layer = 0

    # layer1_conv1 input = 1@28*28 output = 32@28*28
    with tf.variable_scope(name_layers[n_layer]):
        n_layer += 1
        conv1_weights = tf.get_variable('weights', shape=(CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP),
                                        initializer=PARAMETERS_INITIALIZER(stddev=0.1))
        # if regularizer != None:
        #     tf.add_to_collection('loss',regularizer(conv1_weights))
        conv1_biases = tf.get_variable('biases', shape=(CONV1_DEEP), initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, CONV1_STRIDE, CONV1_STRIDE, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # layer2_pool1 input = 32@28*28 output = 32@14*14
    with tf.variable_scope(name_layers[n_layer]):
        n_layer += 1
        pool1 = tf.nn.max_pool(relu1, ksize=[1, POOL1_SIZE, POOL1_SIZE, 1], strides=[1, POOL1_STRIDE, POOL1_STRIDE, 1],
                               padding='SAME')

    # layer3_conv2 input = 32@14*14 output = 64@14*14
    with tf.variable_scope(name_layers[n_layer]):
        n_layer += 1
        conv2_weights = tf.get_variable('weights', shape=(CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP),
                                        initializer=PARAMETERS_INITIALIZER(stddev=0.1))
        # if regularizer != None:
        #     tf.add_to_collection('loss',regularizer(conv2_weights))
        conv2_biases = tf.get_variable('biases', shape=(CONV2_DEEP), initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, CONV2_STRIDE, CONV2_STRIDE, 1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # layer4_pool2 input = 64@14*14 output = 64@7*7
    with tf.variable_scope(name_layers[n_layer]):
        n_layer += 1
        pool2 = tf.nn.max_pool(relu2, ksize=[1,POOL2_SIZE,POOL2_SIZE,1], strides=[1, POOL2_STRIDE, POOL2_STRIDE, 1], padding='SAME')

    # 第5层不用原论文的处理，直接拉成batch*向量
    # layer5_fc1 input = 64*7*7 = 3136 n_layer = 4
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2,shape=(pool_shape[0], nodes))

    value_layers = []
    value_layers.append(reshaped)
    FC.insert(0,nodes)
    in_dimension = FC[0]
    for i in range(1, len(FC)):
        with tf.variable_scope(name_layers[n_layer]):
            n_layer += 1

            out_dimension = FC[i]
            weights = tf.get_variable('weights', shape=(in_dimension, out_dimension),
                                      initializer=PARAMETERS_INITIALIZER(stddev=0.1))
            biases = tf.get_variable('biases', shape=(1, out_dimension),
                                     initializer=tf.constant_initializer(0.0))
            if regularizer != None:
                tf.add_to_collection('loss', regularizer(weights))
                # print('how many times?')
            value = tf.matmul(value_layers[-1], weights) + biases
            if i == len(FC) - 1:
                # 最后一层配合softmax则放弃激活函数
                value_layers.append(value)
            else:
                if train:
                    value_layers.append(tf.nn.dropout(tf.nn.relu(value), 0.5))
                else:
                    value_layers.append(tf.nn.relu(value))
            in_dimension = out_dimension
    return value_layers[-1]